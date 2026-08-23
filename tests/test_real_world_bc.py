"""Tests for the real-world BC entrypoint. Mirrors
test_hil_serl_training_entrypoints.py's style (mock make_training_envs, real
Logger/persist_resolved_config with log_type="none") -- note franka_real
backend registration is a known pre-existing environment quirk (also
affecting several hil_serl entrypoint tests): these pass as part of the full
real_world/tests/ suite (another test module's import registers the backend
as a side effect) but fail if this file is run in isolation.
"""
from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden_real_world.training.bc import RealWorldBCArgs, run_real_world_bc


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self.single_observation_space = spaces.Box(-1, 1, (4,), dtype="float32")
        self.single_action_space = spaces.Box(-1, 1, (3,), dtype="float32")
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)


def _write_demo_pkl(path, n=8):
    transitions = [
        {
            "obs": torch.randn(1, 4),
            "next_obs": torch.randn(1, 4),
            "action": torch.rand(1, 3) * 2 - 1,
            "reward": torch.ones(1),
            "done": torch.zeros(1),
        }
        for _ in range(n)
    ]
    with open(path, "wb") as f:
        pickle.dump(transitions, f)


def _base_args(tmp_path, **overrides) -> RealWorldBCArgs:
    kwargs = dict(
        log_dir=str(tmp_path / "logs"),
        log_type="none",
        num_offline_steps=2,
        checkpoint_freq=0,
        save_final_checkpoint=False,
        batch_size=4,
    )
    kwargs.update(overrides)
    return RealWorldBCArgs(**kwargs)


def test_run_real_world_bc_trains_on_demo_pkls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    demo_path = tmp_path / "demos.pkl"
    _write_demo_pkl(demo_path)

    args = _base_args(tmp_path, demo_dataset_paths=[str(demo_path)])
    run_real_world_bc(args)  # must not raise


def test_run_real_world_bc_raises_when_no_demos_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    args = _base_args(tmp_path, demo_dataset_paths=[str(tmp_path / "nonexistent_*.pkl")])
    try:
        run_real_world_bc(args)
        assert False, "expected ValueError for zero loaded demo transitions"
    except ValueError as e:
        assert "No demo transitions" in str(e)


def test_run_real_world_bc_saves_periodic_and_final_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rl_garden.envs.backend_registry.make_training_envs",
        lambda backend, req: (_FakeEnv(), None),
    )
    demo_path = tmp_path / "demos.pkl"
    _write_demo_pkl(demo_path)

    checkpoint_dir = tmp_path / "ckpt"
    args = _base_args(
        tmp_path,
        demo_dataset_paths=[str(demo_path)],
        num_offline_steps=4,
        checkpoint_freq=2,
        save_final_checkpoint=True,
        checkpoint_dir=str(checkpoint_dir),
    )
    run_real_world_bc(args)

    assert (checkpoint_dir / "checkpoint_2.pt").exists()
    assert (checkpoint_dir / "checkpoint_4.pt").exists()
    assert (checkpoint_dir / "final.pt").exists()
