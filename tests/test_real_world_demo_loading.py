"""Tests for legacy flat-observation wrapping in demo-pkl loading."""
from __future__ import annotations

import pickle

import torch

from rl_garden_real_world.hil_serl.learner_loop import HilSerlLearnerLoop
from rl_garden_real_world.training._demo_loading import as_dict_observation, load_demo_pkls


def _transition(obs, next_obs) -> dict:
    return {
        "obs": obs,
        "next_obs": next_obs,
        "action": torch.zeros(1, 3),
        "reward": torch.zeros(1),
        "done": torch.zeros(1),
    }


def test_as_dict_observation_wraps_flat_tensor_as_state():
    obs = torch.randn(1, 4)
    wrapped = as_dict_observation(obs)
    assert set(wrapped) == {"state"}
    assert wrapped["state"] is obs


def test_as_dict_observation_leaves_dict_untouched():
    obs = {"state": torch.randn(1, 4), "rgb_wrist": torch.zeros(1, 8, 8, 3, dtype=torch.uint8)}
    assert as_dict_observation(obs) is obs


def test_load_demo_pkls_wraps_legacy_flat_obs_and_keeps_dict_obs(tmp_path):
    flat = _transition(torch.randn(1, 4), torch.randn(1, 4))
    dict_obs = {"state": torch.randn(1, 4)}
    dict_next = {"state": torch.randn(1, 4)}
    nested = _transition(dict_obs, dict_next)
    with open(tmp_path / "demo.pkl", "wb") as f:
        pickle.dump([flat, nested], f)

    received = []
    count = load_demo_pkls(
        [str(tmp_path / "*.pkl")], lambda *a, **kw: received.append(a), device="cpu"
    )

    assert count == 2
    (obs0, next0, *_), (obs1, next1, *_) = received
    assert torch.equal(obs0["state"], flat["obs"]) and torch.equal(next0["state"], flat["next_obs"])
    assert set(obs1) == {"state"} and torch.equal(obs1["state"], dict_obs["state"])
    assert torch.equal(next1["state"], dict_next["state"])


def test_hil_serl_add_transition_wraps_legacy_flat_obs():
    received = []
    transition = _transition(torch.randn(1, 4), torch.randn(1, 4))
    HilSerlLearnerLoop._add_transition(lambda *a, **kw: received.append(a), transition, "cpu")
    obs, next_obs = received[0][:2]
    assert torch.equal(obs["state"], transition["obs"])
    assert torch.equal(next_obs["state"], transition["next_obs"])
