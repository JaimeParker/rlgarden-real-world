"""Shared glob+pickle demo-transition loading, factored out of
``HilSerlLearnerLoop._load_demo_datasets`` so the real-world BC entrypoint
(``training/bc.py``) and the online HIL-SERL demo-buffer preload share one
implementation of the pkl-list-of-transition-dicts format instead of two
copies that could silently drift apart.
"""
from __future__ import annotations

import glob
import pickle
from typing import Any, Callable, Sequence

import torch

_TRANSITION_TENSOR_KEYS = ("obs", "next_obs", "action", "reward", "done")


def load_demo_pkls(
    paths: Sequence[str],
    adder: Callable[..., None],
    device: torch.device | str,
) -> int:
    """Glob each pattern in ``paths``, load every pkl (a pickled list of
    transition dicts), and call ``adder(obs, next_obs, action, reward, done,
    **extra)`` for each transition, moving every tensor value to ``device``
    first. Returns the number of transitions loaded."""
    count = 0
    for pattern in paths:
        for pkl_path in sorted(glob.glob(pattern)):
            with open(pkl_path, "rb") as f:
                transitions = pickle.load(f)
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
                    tensors["obs"],
                    tensors["next_obs"],
                    tensors["action"],
                    tensors["reward"],
                    tensors["done"],
                    **extra,
                )
                count += 1
    return count
