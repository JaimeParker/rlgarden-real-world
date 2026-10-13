"""RotvecObsWrapper: converts FrankaRealEnv's quaternion state slice into a
rotation vector, against a fake bridge client (no HTTP/ROS/hardware)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from rl_garden_real_world.envs.franka_real.config import FrankaRealEnvConfig
from rl_garden_real_world.envs.franka_real.env import FrankaRealEnv
from rl_garden_real_world.envs.wrappers.rotvec_obs import RotvecObsWrapper, quat_xyzw_to_rotvec

from test_franka_real_env import _FakeBridgeClient


def _make_wrapped_env(initial_pose=None) -> RotvecObsWrapper:
    bridge = _FakeBridgeClient(initial_pose=initial_pose)
    cfg = FrankaRealEnvConfig(bridge_url="http://unused")
    env = FrankaRealEnv(cfg, bridge_client=bridge, camera_capture=None)
    return RotvecObsWrapper(env)


def test_observation_space_shrinks_state_by_one():
    env = _make_wrapped_env()
    assert env.single_observation_space["state"].shape == (19,)
    assert env.observation_space["state"].shape == (1, 19)


def test_reset_converts_quat_to_rotvec():
    quat = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    pose = [0.5, 0.0, 0.2, *quat]
    env = _make_wrapped_env(initial_pose=pose)

    obs, _ = env.reset()
    assert obs["state"].shape == (1, 19)
    np.testing.assert_allclose(obs["state"][0, :3].numpy(), [0.5, 0.0, 0.2], atol=1e-6)
    expected_rotvec = Rotation.from_quat(quat).as_rotvec()
    np.testing.assert_allclose(obs["state"][0, 3:6].numpy(), expected_rotvec, atol=1e-6)
    # rest of the state (vel/gripper/force/torque) passes through unchanged.
    np.testing.assert_allclose(obs["state"][0, 6:].numpy(), np.zeros(13), atol=1e-6)


def test_step_converts_quat_to_rotvec():
    env = _make_wrapped_env()
    env.reset()
    obs, _, _, _, _ = env.step(torch.zeros(1, 7))
    assert obs["state"].shape == (1, 19)
    np.testing.assert_allclose(obs["state"][0, 3:6].numpy(), [0.0, 0.0, 0.0], atol=1e-6)


class _FakeStateEnv(gym.Env):
    num_envs = 2

    def __init__(self, state: torch.Tensor):
        self._state = state
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, shape=(state.shape[-1],), dtype=np.float32)}
        )

    def reset(self, **kwargs):
        return {"state": self._state}, {}


@pytest.mark.parametrize("quat_slice", [(3, 7), (0, 4), (5, 9)])
def test_custom_quat_slice(quat_slice):
    start, end = quat_slice
    quat = Rotation.from_euler("xyz", [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]]).as_quat()
    state = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12)
    state[:, start:end] = torch.as_tensor(quat, dtype=torch.float32)
    env = RotvecObsWrapper(_FakeStateEnv(state), quat_slice=quat_slice)
    assert env.single_observation_space["state"].shape == (11,)
    obs, _ = env.reset()
    out = obs["state"]
    assert out.shape == (2, 11)
    torch.testing.assert_close(out[:, :start], state[:, :start])
    np.testing.assert_allclose(
        out[:, start : start + 3].numpy(), Rotation.from_quat(quat).as_rotvec(), atol=1e-5
    )
    torch.testing.assert_close(out[:, start + 3 :], state[:, end:])


def test_quat_slice_must_span_four_dims():
    with pytest.raises(ValueError):
        RotvecObsWrapper(_FakeStateEnv(torch.zeros(1, 12)), quat_slice=(3, 6))


def test_quat_xyzw_to_rotvec_matches_scipy():
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(20, 4))
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)  # includes w < 0 cases
    expected = Rotation.from_quat(quats).as_rotvec()
    actual = quat_xyzw_to_rotvec(torch.as_tensor(quats, dtype=torch.float64))
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-8)


def test_quat_xyzw_to_rotvec_identity_and_tiny_angle_are_finite():
    q = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1e-12, 0.0, 0.0, 1.0]])
    out = quat_xyzw_to_rotvec(q)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros(2, 3), atol=1e-6, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quat_xyzw_to_rotvec_preserves_device_and_dtype(dtype):
    q = torch.tensor([[0.0, 0.0, 0.7071, 0.7071]], dtype=dtype)
    out = quat_xyzw_to_rotvec(q)
    assert out.dtype == dtype and out.device == q.device and out.shape == (1, 3)
