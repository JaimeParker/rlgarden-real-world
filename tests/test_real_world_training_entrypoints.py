"""Tests that training/real_world's thin entrypoints wire the right pieces
together -- env/agent construction and Loop class selection -- without
actually running a robot or a training loop (both SerlActorLoop.run()/
SerlLearnerLoop.run() are monkeypatched to no-ops that record their args)."""
from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden_real_world.training.serl import SerlArgs


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self.single_observation_space = spaces.Box(-1, 1, (4,), dtype="float32")
        self.single_action_space = spaces.Box(-1, 1, (2,), dtype="float32")
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)


class _FakePolicy:
    def eval(self):
        pass


class _FakeAgent:
    def __init__(self):
        self.policy = _FakePolicy()
        self.device = torch.device("cpu")


def test_run_actor_builds_scratch_agent_with_tiny_buffer_and_starts_actor_loop(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )

    def _fake_build_rlpd(args, env, eval_env, logger, checkpoint_dir):
        captured["build_args"] = args
        captured["logger"] = logger
        captured["checkpoint_dir"] = checkpoint_dir
        return _FakeAgent()

    monkeypatch.setattr("rl_garden.training.online.rlpd.build_rlpd", _fake_build_rlpd)

    class _FakeActorLoop:
        def __init__(self, env, policy, sync_client, **kwargs):
            captured["actor_loop_kwargs"] = dict(env=env, policy=policy, sync_client=sync_client, **kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden_real_world.serl.SerlActorLoop", _FakeActorLoop)

    from rl_garden_real_world.training.serl import _run_actor

    args = SerlArgs(role="actor", buffer_size=1_000_000, sync_host="10.0.0.1", sync_port=7000)
    _run_actor(args)

    assert captured["build_args"].buffer_size == 8, "actor's scratch agent must use a tiny buffer"
    assert captured["logger"] is None
    assert captured["checkpoint_dir"] is None
    assert captured["ran"] is True
    assert captured["actor_loop_kwargs"]["sync_client"]._base_url == "http://10.0.0.1:7000"


def test_run_learner_builds_full_agent_and_starts_learner_loop(monkeypatch, tmp_path):
    captured = {}

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )

    def _fake_build_rlpd(args, env, eval_env, logger, checkpoint_dir):
        captured["build_args"] = args
        captured["checkpoint_dir"] = checkpoint_dir
        return _FakeAgent()

    monkeypatch.setattr("rl_garden.training.online.rlpd.build_rlpd", _fake_build_rlpd)

    class _FakeLoggerHandle:
        def close(self):
            captured["logger_closed"] = True

    monkeypatch.setattr(
        "rl_garden.common.Logger.create", lambda **kwargs: _FakeLoggerHandle()
    )

    class _FakeLearnerLoop:
        def __init__(self, agent, host, port, **kwargs):
            captured["learner_loop_kwargs"] = dict(agent=agent, host=host, port=port, **kwargs)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr("rl_garden_real_world.serl.SerlLearnerLoop", _FakeLearnerLoop)

    from rl_garden_real_world.training.serl import _run_learner

    args = SerlArgs(
        role="learner",
        sync_host="0.0.0.0",
        sync_port=6000,
        log_dir=str(tmp_path),
        log_type="none",
    )
    _run_learner(args)

    assert captured["build_args"] is args
    assert captured["ran"] is True
    assert captured["logger_closed"] is True
    assert captured["learner_loop_kwargs"]["host"] == "0.0.0.0"
    assert captured["learner_loop_kwargs"]["port"] == 6000


# -- real_robot backend selection ---------------------------------------------------


def test_serl_args_parse_real_robot_backend():
    import tyro

    args = tyro.cli(
        SerlArgs,
        args=[
            "--env_backend", "real_robot",
            "--real_robot.remote", "127.0.0.1:1",
            "--real_robot.num-joints", "6",
        ],
    )
    assert args.env_backend == "real_robot"
    assert args.real_robot.remote == "127.0.0.1:1"
    assert args.real_robot.num_joints == 6
    assert args.resolve_backend_config() is args.real_robot


def test_serl_default_env_request_is_franka_real():
    import dataclasses

    from rl_garden_real_world.envs.real_robot import RealRobotEnvConfig
    from rl_garden_real_world.training._args import FrankaRealConfig
    from rl_garden_real_world.training.serl import _serl_env_request

    args = SerlArgs()
    assert args.env_backend == "franka_real"
    assert args.real_robot == RealRobotEnvConfig()
    request = _serl_env_request(args, run_name="snapshot")
    snapshot = {
        f.name: getattr(request, f.name)
        for f in dataclasses.fields(request)
        if f.name not in ("observation", "eval_record_dir")
    }
    assert snapshot == {
        "env_id": args.env_id,
        "num_envs": args.num_envs,
        "control_mode": args.control_mode,
        "render_mode": args.render_mode,
        "seed": args.seed,
        "reward_scale": args.reward_scale,
        "reward_bias": args.reward_bias,
        "num_eval_envs": args.num_eval_envs,
        "capture_video": args.capture_video,
        "video_fps": args.video_fps,
        "num_eval_steps": args.num_eval_steps,
        "create_eval_env": True,
        "backend_config": FrankaRealConfig(),
    }
    assert request.observation == args.obs


def test_serl_actor_builds_real_robot_env_with_real_robot_config(monkeypatch):
    from rl_garden_real_world.envs.real_robot import RealRobotEnvConfig

    captured = {}

    def _fake_make_training_envs(backend, req):
        captured["backend"] = backend
        captured["request"] = req
        return _FakeEnv(), None

    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs", _fake_make_training_envs
    )
    monkeypatch.setattr(
        "rl_garden.training.online.rlpd.build_rlpd", lambda *a, **k: _FakeAgent()
    )

    class _FakeActorLoop:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            pass

    monkeypatch.setattr("rl_garden_real_world.serl.SerlActorLoop", _FakeActorLoop)

    from rl_garden_real_world.training.serl import _run_actor

    config = RealRobotEnvConfig(remote="127.0.0.1:1", num_joints=6)
    _run_actor(SerlArgs(role="actor", env_backend="real_robot", real_robot=config))

    assert captured["backend"] == "real_robot"
    assert captured["request"].backend_config is config
