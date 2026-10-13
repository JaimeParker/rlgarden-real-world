"""Tests for the real_robot env backend: config resolution, flattening, registration."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

import robot_infra.devices.claims as claims
from rl_garden.envs.backend_registry import EnvRequest, _validate_env_observation_contract
from rl_garden.observations.config import ObservationConfig
from rl_garden.observations.schema import ObservationContractError

from rl_garden_real_world.envs.backends.real_robot import RealRobotBackend
from rl_garden_real_world.envs.real_robot import RealRobotEnvConfig

FAKE_FRANKA = {
    "type": "franka",
    "arm_backend": "fake",
    "gripper": {"backend": "fake"},
    "cameras": [{"name": "wrist", "backend": "fake", "width": 32, "height": 24, "fps": 60}],
}
STATE_DIM = 7 + 6 + 7 + 7 + 1 + 3 + 3


@pytest.fixture(autouse=True)
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def _req(observation: ObservationConfig | None = None, **config_overrides) -> EnvRequest:
    config = dict(robot=FAKE_FRANKA, reset_settle_s=0.0, home_joints=[0.0] * 7)
    config.update(config_overrides)
    return EnvRequest(
        env_id="unused",
        num_envs=8,
        control_mode="",
        render_mode="rgb_array",
        seed=1,
        num_eval_envs=4,
        observation=observation or ObservationConfig(state=True, rgb=("wrist",)),
        backend_config=RealRobotEnvConfig(**config),
    )


@pytest.fixture
def built_envs():
    envs = []
    yield envs
    for env in envs:
        env.close()


def test_backend_class_contract():
    assert RealRobotBackend.__dict__["api_version"] == 2
    assert RealRobotBackend.config_field == "real_robot"
    for name in ("resolve_config", "make_train_env", "make_eval_env"):
        assert name in RealRobotBackend.__dict__


def test_entry_point_declared_in_pyproject_resolves_to_backend():
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    target = pyproject["project"]["entry-points"]["rlgarden.env_backends"]["real_robot"]
    module = importlib.import_module(target)
    assert module.RealRobotBackend is RealRobotBackend


def test_registered_under_real_robot():
    from rl_garden.envs.backend_registry import _REGISTRY

    assert _REGISTRY["real_robot"] is RealRobotBackend


def test_resolve_config_takes_cameras_and_image_size_from_observation():
    req = _req(ObservationConfig(state=True, rgb=("wrist",), image_size=(16, 16)))
    config = RealRobotBackend.resolve_config(req, is_eval=False)
    assert config.cameras == ["wrist"]
    assert config.image_size == (16, 16)
    assert RealRobotBackend.resolve_config(req, is_eval=True) == config
    assert req.backend_config.cameras == []


@pytest.mark.parametrize(
    "observation, config_overrides, match",
    [
        (ObservationConfig(state=True, depth=("wrist",)), {}, "depth"),
        (ObservationConfig(state=True, extra_state=("obj",)), {}, "extra_state"),
        (ObservationConfig(state=True, rgb=("wrist",), frame_stack=2), {}, "frame_stack"),
        (ObservationConfig(state=True, rgb=("wrist",)), {"cameras": ["front"]}, "conflict"),
        (
            ObservationConfig(state=True, rgb=("wrist",), image_size=(16, 16)),
            {"image_size": (8, 8)},
            "conflict",
        ),
    ],
)
def test_resolve_config_rejects_unsupported_observation(observation, config_overrides, match):
    with pytest.raises(ObservationContractError, match=match):
        RealRobotBackend.resolve_config(_req(observation, **config_overrides), is_eval=False)


def test_resolve_config_requires_real_robot_config():
    req = _req()
    req.backend_config = None
    with pytest.raises(ValueError, match="RealRobotEnvConfig"):
        RealRobotBackend.resolve_config(req, is_eval=False)


@pytest.mark.parametrize("sources", [dict(robot=None), dict(remote="host:1")])
def test_resolve_config_requires_exactly_one_robot_source(sources):
    req = _req()
    req.backend_config = RealRobotEnvConfig(
        **{**dict(robot=FAKE_FRANKA, home_joints=[0.0] * 7), **sources}
    )
    with pytest.raises(ValueError, match="exactly one"):
        RealRobotBackend.resolve_config(req, is_eval=False)


def test_make_env_is_shared_flattened_and_returns_torch(built_envs):
    req = _req()
    env = RealRobotBackend.make_train_env(req)
    built_envs.append(env)
    assert RealRobotBackend.make_eval_env(req) is env
    assert env.num_envs == 1
    assert not env.robot.is_connected
    _validate_env_observation_contract(env, req)
    assert env.single_observation_space["state"].shape == (STATE_DIM,)
    assert env.action_space.shape == (1, 7)

    obs, _ = env.reset(seed=0)
    assert obs["state"].shape == (1, STATE_DIM) and obs["state"].dtype == torch.float32
    assert obs["rgb_wrist"].shape == (1, 24, 32, 3) and obs["rgb_wrist"].dtype == torch.uint8
    # gripper_pos sits after tcp_pose, tcp_vel, joint_pos, joint_vel; reset opens the gripper.
    assert obs["state"][0, 7 + 6 + 7 + 7].item() == 1.0

    obs, reward, terminated, truncated, info = env.step(torch.zeros(1, 7))
    assert obs["state"].shape == (1, STATE_DIM)
    assert reward.shape == (1,) and reward.dtype == torch.float32
    assert terminated.dtype == torch.bool and truncated.dtype == torch.bool
    assert info["elapsed_steps"] == 1

    env.close()
    assert not env.robot.is_connected


def test_state_only_and_rgb_only_requests(built_envs):
    state_req = _req(ObservationConfig(state=True))
    state_env = RealRobotBackend.make_train_env(state_req)
    built_envs.append(state_env)
    _validate_env_observation_contract(state_env, state_req)
    assert list(state_env.single_observation_space.spaces) == ["state"]

    rgb_req = _req(ObservationConfig(state=False, rgb=("wrist",), image_size=(16, 16)))
    rgb_env = RealRobotBackend.make_train_env(rgb_req)
    built_envs.append(rgb_env)
    _validate_env_observation_contract(rgb_env, rgb_req)
    obs, _ = rgb_env.reset()
    assert set(obs) == {"rgb_wrist"}
    assert obs["rgb_wrist"].shape == (1, 16, 16, 3)


def test_state_concatenates_in_config_field_order(built_envs):
    fields = ("gripper_pos", "joint_pos")
    req = _req(ObservationConfig(state=True), state_fields=fields)
    env = RealRobotBackend.make_train_env(req)
    built_envs.append(env)
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["state"][0].numpy(), [1.0] + [0.0] * 7)


def test_reward_scale_bias_wrapper_keeps_contract(built_envs):
    req = _req(ObservationConfig(state=True))
    req.reward_bias = 1.0
    env = RealRobotBackend.make_train_env(req)
    built_envs.append(env)
    _validate_env_observation_contract(env, req)
    env.reset()
    _, reward, *_ = env.step(torch.zeros(1, 7))
    assert reward.tolist() == [1.0]
