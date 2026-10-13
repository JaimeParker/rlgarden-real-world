"""Tests for the eval-only real-robot deploy entrypoint (rl_garden_real_world/deploy.py)."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch
import tyro
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

import robot_infra.devices.claims as claims
from robot_infra.devices.errors import DeviceFault

from rl_garden_real_world.deploy import (
    DEVICE_ERROR_EXIT_CODE,
    DeployArgs,
    EpisodeResult,
    build_deploy_env,
    main,
    restore_policy,
    restore_training_args,
    run_deploy,
)
from rl_garden_real_world.envs.real_robot import RealRobotEnvConfig

FAKE_FRANKA = {
    "type": "franka",
    "arm_backend": "fake",
    "gripper": {"backend": "fake"},
}


class _FakeHealth:
    ok = True


class _FakeRobot:
    def health(self):
        return _FakeHealth()


class _FakeEnv(gym.Env):
    """Batch-of-1 torch env shaped like RealRobotVectorEnv; truncates after ``episode_len``."""

    num_envs = 1

    def __init__(self, episode_len: int = 3, fail_on_step: int | None = None) -> None:
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-1, 1, (4,), dtype="float32")}
        )
        self.single_action_space = spaces.Box(-1, 1, (2,), dtype="float32")
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)
        self.robot = _FakeRobot()
        self.episode_len = episode_len
        self.fail_on_step = fail_on_step
        self.total_steps = 0
        self.resets = 0
        self.closed = False
        self._t = 0

    def _obs(self):
        return {"state": torch.zeros(1, 4)}

    def reset(self, *, seed=None, options=None):
        self.resets += 1
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        assert action.shape == (1, 2)
        self.total_steps += 1
        if self.fail_on_step is not None and self.total_steps == self.fail_on_step:
            raise DeviceFault("arm faulted", hint="clear the fault in Desk", device="arm")
        self._t += 1
        return (
            self._obs(),
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([self._t >= self.episode_len]),
            {"rtt_ms": 2.0, "elapsed_steps": self._t},
        )

    def close(self):
        self.closed = True


class _FakePolicy:
    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.evaluated = False

    def eval(self):
        self.evaluated = True

    def predict(self, obs, deterministic=False):
        self.calls.append(deterministic)
        return torch.zeros(1, 2)


def test_run_deploy_runs_episodes_and_prints_summary(capsys):
    env, policy = _FakeEnv(episode_len=3), _FakePolicy()
    args = DeployArgs(checkpoint="unused.pt", episodes=2)
    results = run_deploy(args, env=env, policy=policy)

    assert results == [
        EpisodeResult(steps=3, episode_return=3.0, truncated=True, rtt_ms_mean=2.0, health_ok=True)
    ] * 2
    assert env.resets == 2 and env.closed
    assert policy.evaluated and policy.calls == [True] * 6
    out = capsys.readouterr().out
    assert out.count("[deploy] episode=") == 2
    assert "[deploy] summary episodes=2 mean_steps=3.0 mean_return=3.000" in out


def test_max_steps_per_episode_caps_episode():
    args = DeployArgs(checkpoint="unused.pt", episodes=1, max_steps_per_episode=2, deterministic=False)
    policy = _FakePolicy()
    (result,) = run_deploy(args, env=_FakeEnv(episode_len=10), policy=policy)
    assert result.steps == 2 and result.truncated
    assert policy.calls == [False, False]


def test_device_error_prints_hint_closes_env_and_returns_2(capsys):
    env = _FakeEnv(fail_on_step=2)
    code = main(DeployArgs(checkpoint="unused.pt", episodes=2), env=env, policy=_FakePolicy())
    assert code == DEVICE_ERROR_EXIT_CODE
    assert env.closed
    err = capsys.readouterr().err
    assert "arm faulted" in err and "clear the fault in Desk" in err


def test_main_returns_zero_on_success():
    env = _FakeEnv()
    assert main(DeployArgs(checkpoint="unused.pt", episodes=1), env=env, policy=_FakePolicy()) == 0
    assert env.closed


def test_deploy_args_parse():
    args = tyro.cli(
        DeployArgs,
        args=[
            "--checkpoint", "ckpt.pt",
            "--algo", "rlpd_hybrid",
            "--real_robot.remote", "127.0.0.1:1",
            "--real_robot.num-joints", "6",
            "--episodes", "3",
            "--no-deterministic",
            "--max-steps-per-episode", "50",
        ],
    )
    assert args.checkpoint == "ckpt.pt" and args.algo == "rlpd_hybrid"
    assert args.real_robot.remote == "127.0.0.1:1" and args.real_robot.num_joints == 6
    assert args.episodes == 3 and not args.deterministic and args.max_steps_per_episode == 50
    assert args.device == "cpu"


# -- checkpoint restore against a real (fake-hardware) real_robot env ------------------


@pytest.fixture
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def _deploy_args(checkpoint, **overrides) -> DeployArgs:
    real_robot = RealRobotEnvConfig(
        robot=FAKE_FRANKA, home_joints=[0.0] * 7, reset_settle_s=0.0, control_hz=1000.0
    )
    return DeployArgs(checkpoint=str(checkpoint), real_robot=real_robot, **overrides)


def test_restores_saved_rlpd_checkpoint_and_deploys(tmp_path, _claim_dir, capsys):
    from rl_garden.training.online.rlpd import build_rlpd

    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_0.pt"
    checkpoint.parent.mkdir(parents=True)
    args = _deploy_args(checkpoint, episodes=1, max_steps_per_episode=2)

    # Train-side: an RLPD agent on the same env spaces, saved with its weights perturbed.
    training_args = _scratch_rlpd_args()
    env = build_deploy_env(args, training_args)
    saved = build_rlpd(training_args, env, None, logger=None, checkpoint_dir=None)
    with torch.no_grad():
        for param in saved.policy.parameters():
            param.add_(0.5)
    saved.save(checkpoint)
    env.close()

    restored_args = restore_training_args(args)
    env = build_deploy_env(args, restored_args)
    try:
        policy = restore_policy(args, restored_args, env)
        expected = saved.policy.state_dict()
        actual = policy.state_dict()
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value.to(actual[key].device))
        assert all(p.device.type == "cpu" for p in policy.parameters())
    finally:
        env.close()

    (result,) = run_deploy(args)
    assert result.steps == 2 and result.truncated and result.health_ok is True
    assert "[deploy] summary episodes=1" in capsys.readouterr().out


def _scratch_rlpd_args():
    from rl_garden.training.online.rlpd import RLPDArgs

    return RLPDArgs(num_envs=1, num_eval_envs=1, buffer_size=8, buffer_device="cpu")
