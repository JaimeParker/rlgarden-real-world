"""Shared glob+pickle demo-transition loading, factored out of
``HilSerlLearnerLoop._load_demo_datasets`` so the real-world BC entrypoint
(``training/bc.py``) and the online HIL-SERL demo-buffer preload share one
implementation of the pkl-list-of-transition-dicts format instead of two
copies that could silently drift apart.
"""
from __future__ import annotations

import glob
import pickle
from typing import Any, Callable, Mapping, Sequence

import torch

from rl_garden_real_world.action_contract import (
    read_contract_sidecar,
    validate_transitions_against_contract,
)

_TRANSITION_TENSOR_KEYS = ("obs", "next_obs", "action", "reward", "done")


def as_dict_observation(obs: Any) -> Any:
    """Wrap a legacy flat-tensor observation as ``{"state": obs}``.

    rl-garden replay buffers are Dict-only (state-only observations live
    under the ``"state"`` key), but pkls written before that change (and
    state-only demo pkls) store ``obs``/``next_obs`` as bare tensors.
    Dict observations (what ``FrankaRealEnv`` emits) pass through untouched.
    """
    return obs if isinstance(obs, Mapping) else {"state": obs}


def load_demo_pkls(
    paths: Sequence[str],
    adder: Callable[..., None],
    device: torch.device | str,
    expected_contract: Mapping[str, Any] | None = None,
) -> int:
    """Glob each pattern in ``paths``, load every pkl (a pickled list of
    transition dicts), and call ``adder(obs, next_obs, action, reward, done,
    **extra)`` for each transition, moving every tensor value to ``device``
    first. Returns the number of transitions loaded.

    With ``expected_contract`` (``action_contract.DEMO_CONTRACT_KEYS`` of the target env), every
    pkl is checked before any of its transitions is added: a ``<pkl stem>.contract.json``
    sidecar must match on action_mode/arm_action_dim/gripper_in_action and the limits, scales and
    state_fields it records, and without one only the action width is checked (with a warning).
    ``None`` skips all checks."""
    count = 0
    for pattern in paths:
        for pkl_path in sorted(glob.glob(pattern)):
            with open(pkl_path, "rb") as f:
                transitions = pickle.load(f)
            validate_transitions_against_contract(
                transitions,
                expected_contract,
                source=pkl_path,
                saved=read_contract_sidecar(pkl_path),
            )
            for transition in transitions:
                tensors = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in transition.items()
                    if k in _TRANSITION_TENSOR_KEYS
                }
                extra: dict[str, Any] = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in transition.items()
                    if k not in _TRANSITION_TENSOR_KEYS
                }
                adder(
                    as_dict_observation(tensors["obs"]),
                    as_dict_observation(tensors["next_obs"]),
                    tensors["action"],
                    tensors["reward"],
                    tensors["done"],
                    **extra,
                )
                count += 1
    return count
