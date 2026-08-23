"""Tests for GraspPenaltyWrapper."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from rl_garden_real_world.envs.wrappers.grasp_penalty import GraspPenaltyWrapper


class _FakeEnv(gym.Env):
    num_envs = 1

    def reset(self, **kwargs):
        return torch.zeros(1, 4), {}

    def step(self, action):
        del action
        return (
            torch.ones(1, 4),
            torch.tensor([1.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )


def _gripper_action(value: float) -> torch.Tensor:
    action = torch.zeros(1, 7)
    action[..., -1] = value
    return action


def test_no_penalty_on_first_step_after_reset():
    wrapped = GraspPenaltyWrapper(_FakeEnv())
    wrapped.reset()
    _, _, _, _, info = wrapped.step(_gripper_action(-1.0))
    assert info["grasp_penalty"].item() == 0.0


def test_penalty_fires_exactly_on_gripper_flip():
    wrapped = GraspPenaltyWrapper(_FakeEnv(), penalty=-0.05)
    wrapped.reset()
    wrapped.step(_gripper_action(-1.0))  # open

    _, _, _, _, info_same = wrapped.step(_gripper_action(-1.0))  # still open
    assert info_same["grasp_penalty"].item() == 0.0

    _, _, _, _, info_flip = wrapped.step(_gripper_action(1.0))  # now closed
    assert info_flip["grasp_penalty"].item() == pytest.approx(-0.05)

    _, _, _, _, info_flip_back = wrapped.step(_gripper_action(-1.0))  # open again
    assert info_flip_back["grasp_penalty"].item() == pytest.approx(-0.05)


def test_honors_intervene_action_over_policy_action():
    class _InterveningEnv(_FakeEnv):
        def step(self, action):
            del action
            return (
                torch.ones(1, 4),
                torch.tensor([1.0]),
                torch.tensor([False]),
                torch.tensor([False]),
                {"intervene_action": _gripper_action(1.0)},
            )

    wrapped = GraspPenaltyWrapper(_InterveningEnv())
    wrapped.reset()
    wrapped.step(_gripper_action(-1.0))  # opens per intervene_action, not the -1.0 policy action

    _, _, _, _, info = wrapped.step(_gripper_action(-1.0))
    # Both steps' executed action (via intervene_action) is 1.0 -> closed,
    # never flips.
    assert info["grasp_penalty"].item() == 0.0


def test_getattr_forwards_to_wrapped_env():
    env = _FakeEnv()
    wrapped = GraspPenaltyWrapper(env)
    assert wrapped.num_envs == 1
