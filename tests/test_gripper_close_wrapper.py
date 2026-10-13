"""Tests for GripperCloseEnv."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl_garden_real_world.envs.wrappers.gripper_close import GripperCloseEnv


class _FakeEnv(gym.Env):
    num_envs = 1
    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.full((1, 7), -1.0, dtype=np.float32),
        high=np.full((1, 7), 1.0, dtype=np.float32),
    )

    def __init__(self):
        self.step_actions = []

    def reset(self, **kwargs):
        return torch.zeros(1, 7), {}

    def step(self, action):
        self.step_actions.append(action)
        return (
            torch.ones(1, 7),
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )


def test_action_space_is_6d():
    wrapped = GripperCloseEnv(_FakeEnv())
    assert wrapped.single_action_space.shape == (6,)
    assert wrapped.action_space.shape == (1, 6)


def test_step_zero_pads_gripper_dim_before_forwarding():
    env = _FakeEnv()
    wrapped = GripperCloseEnv(env)
    action = torch.full((1, 6), 0.5)
    wrapped.step(action)
    forwarded = env.step_actions[0]
    assert forwarded.shape == (1, 7)
    torch.testing.assert_close(forwarded[..., :6], action)
    torch.testing.assert_close(forwarded[..., 6], torch.zeros(1))


def test_intervene_action_truncated_when_present():
    class _EnvWithIntervention(_FakeEnv):
        def step(self, action):
            self.step_actions.append(action)
            return (
                torch.ones(1, 7),
                torch.tensor([0.0]),
                torch.tensor([False]),
                torch.tensor([False]),
                {"intervene_action": torch.full((1, 7), 9.0)},
            )

    wrapped = GripperCloseEnv(_EnvWithIntervention())
    _, _, _, _, info = wrapped.step(torch.zeros(1, 6))
    assert info["intervene_action"].shape == (1, 6)


def test_intervene_action_absent_is_untouched():
    wrapped = GripperCloseEnv(_FakeEnv())
    _, _, _, _, info = wrapped.step(torch.zeros(1, 6))
    assert "intervene_action" not in info


def _make_env(dim: int):
    class _Env(_FakeEnv):
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)
        action_space = spaces.Box(
            low=np.full((1, dim), -1.0, dtype=np.float32),
            high=np.full((1, dim), 1.0, dtype=np.float32),
        )

        def step(self, action):
            self.step_actions.append(action)
            return (
                torch.ones(1, 7),
                torch.tensor([0.0]),
                torch.tensor([False]),
                torch.tensor([False]),
                {"intervene_action": torch.arange(dim, dtype=torch.float32)[None]},
            )

    return _Env()


@pytest.mark.parametrize(
    "dim, gripper_dim, closed_value",
    [(7, 0, 0.0), (9, -1, 0.0), (9, 4, -1.0), (9, 0, 0.5)],
)
def test_configurable_gripper_dim_and_closed_value(dim, gripper_dim, closed_value):
    env = _make_env(dim)
    wrapped = GripperCloseEnv(env, gripper_dim=gripper_dim, closed_value=closed_value)
    assert wrapped.single_action_space.shape == (dim - 1,)
    assert wrapped.action_space.shape == (1, dim - 1)

    action = torch.arange(1, dim, dtype=torch.float32)[None] / 10
    _, _, _, _, info = wrapped.step(action)
    forwarded = env.step_actions[0]
    index = gripper_dim % dim
    kept = [i for i in range(dim) if i != index]
    assert forwarded.shape == (1, dim)
    torch.testing.assert_close(forwarded[..., kept], action)
    assert forwarded[0, index].item() == closed_value
    torch.testing.assert_close(
        info["intervene_action"], torch.tensor(kept, dtype=torch.float32)[None]
    )


def test_rejects_non_1d_or_too_small_action_space():
    class _OneDim(_FakeEnv):
        single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    with pytest.raises(ValueError):
        GripperCloseEnv(_OneDim())
    with pytest.raises(ValueError):
        GripperCloseEnv(_FakeEnv(), gripper_dim=7)
