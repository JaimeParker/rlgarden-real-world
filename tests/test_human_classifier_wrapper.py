"""Tests for HumanClassifierWrapper."""
from __future__ import annotations

import gymnasium as gym
import pytest
import torch

from rl_garden_real_world.envs.wrappers.human_classifier import HumanClassifierWrapper


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self, terminated: bool = False, truncated: bool = False):
        self._terminated = terminated
        self._truncated = truncated

    def reset(self, **kwargs):
        return torch.zeros(1, 4), {}

    def step(self, action):
        del action
        return (
            torch.ones(1, 4),
            torch.tensor([0.0]),
            torch.tensor([self._terminated]),
            torch.tensor([self._truncated]),
            {"raw": True},
        )


def test_non_terminal_step_does_not_prompt(monkeypatch):
    def _fail_input(prompt):
        raise AssertionError("input() must not be called on a non-terminal step")

    monkeypatch.setattr("builtins.input", _fail_input)
    wrapped = HumanClassifierWrapper(_FakeEnv())
    _, reward, _, _, info = wrapped.step(torch.zeros(1, 4))
    assert reward.item() == 0.0
    assert "succeed" not in info
    assert info["raw"] is True


def test_terminal_step_prompts_and_sets_succeed_on_success(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    wrapped = HumanClassifierWrapper(_FakeEnv(terminated=True))
    _, reward, terminated, _, info = wrapped.step(torch.zeros(1, 4))
    assert bool(terminated[0])
    assert reward.item() == 1.0
    assert info["succeed"].item() == 1.0


def test_terminal_step_prompts_and_sets_succeed_on_failure(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "0")
    wrapped = HumanClassifierWrapper(_FakeEnv(truncated=True))
    _, reward, _, truncated, info = wrapped.step(torch.zeros(1, 4))
    assert bool(truncated[0])
    assert reward.item() == 0.0
    assert info["succeed"].item() == 0.0


def test_invalid_input_is_reprompted(monkeypatch):
    answers = iter(["maybe", "2", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    wrapped = HumanClassifierWrapper(_FakeEnv(terminated=True))
    _, reward, _, _, info = wrapped.step(torch.zeros(1, 4))
    assert reward.item() == 1.0
    assert info["succeed"].item() == 1.0
