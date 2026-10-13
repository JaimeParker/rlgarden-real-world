"""Action-space contract shared by intervention, demo/snapshot loading and deploy.

A contract is the dict ``RealRobotEnv.action_contract`` exposes; the demo contract
(``DEMO_CONTRACT_KEYS``) is the subset persisted next to pkl transition files, so a
replayed transition can be checked against the env it is replayed into. Backends
without a contract (``franka_real``) yield ``None`` and keep their legacy behavior.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

# What a normalized action physically means: these must match for a replayed transition.
_PHYSICAL_KEYS = ("joint_limits", "xyz_scale_m", "rot_scale_rad", "joint_scale_rad", "state_fields")
DEMO_CONTRACT_KEYS = ("action_mode", "arm_action_dim", "gripper_in_action", "num_joints") + _PHYSICAL_KEYS
_COMPARED_KEYS = ("action_mode", "arm_action_dim", "gripper_in_action") + _PHYSICAL_KEYS
# Pre-contract convention: a 6-D cartesian twist plus a trailing gripper dimension.
LEGACY_ACTION_CONTRACT: Mapping[str, Any] = {
    "action_mode": "cartesian_delta",
    "arm_action_dim": 6,
    "gripper_in_action": True,
}
_SIDECAR_SUFFIX = ".contract.json"


def get_env_action_contract(env: Any) -> Optional[dict]:
    """``env.action_contract`` through any wrapper chain, or ``None`` if the backend has none."""
    getter = getattr(env, "get_wrapper_attr", None)
    try:
        contract = getter("action_contract") if getter is not None else env.action_contract
    except AttributeError:
        return None
    return dict(contract) if isinstance(contract, Mapping) else None


def action_dim_of(contract: Mapping[str, Any]) -> int:
    return int(contract["arm_action_dim"]) + int(bool(contract["gripper_in_action"]))


def demo_contract_of(env: Any) -> Optional[dict]:
    """The persistable subset of ``env``'s contract, or ``None`` for contract-less backends."""
    contract = get_env_action_contract(env)
    if contract is None:
        return None
    return {key: contract[key] for key in DEMO_CONTRACT_KEYS if key in contract}


def describe_contract(contract: Mapping[str, Any]) -> str:
    return (
        f"action_mode={contract['action_mode']!r}, arm_action_dim={contract['arm_action_dim']}, "
        f"gripper_in_action={contract['gripper_in_action']} "
        f"(action dim {action_dim_of(contract)})"
    )


def contract_sidecar_path(pkl_path: str | Path) -> Path:
    return Path(pkl_path).with_suffix(_SIDECAR_SUFFIX)


def write_contract_sidecar(pkl_path: str | Path, contract: Mapping[str, Any]) -> None:
    contract_sidecar_path(pkl_path).write_text(
        json.dumps({key: contract[key] for key in DEMO_CONTRACT_KEYS if key in contract}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _contract_values_match(saved: Any, expected: Any) -> bool:
    if saved is None or expected is None:
        return saved is None and expected is None
    if isinstance(saved, str) or isinstance(expected, str):
        return saved == expected
    if isinstance(saved, (list, tuple)) and any(isinstance(item, str) for item in saved):
        return list(saved) == list(expected)  # state_fields (a JSON list vs the env's tuple)
    if isinstance(saved, bool) or isinstance(expected, bool):
        return saved == expected
    saved_array, expected_array = np.asarray(saved, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return saved_array.shape == expected_array.shape and bool(np.allclose(saved_array, expected_array))


def read_contract_sidecar(pkl_path: str | Path) -> Optional[dict]:
    path = contract_sidecar_path(pkl_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_transitions_against_contract(
    transitions: Sequence[Mapping[str, Any]],
    expected: Optional[Mapping[str, Any]],
    *,
    source: str,
    saved: Optional[Mapping[str, Any]] = None,
) -> None:
    """Raise ``ValueError`` if ``transitions`` were not recorded for the ``expected`` action space.

    ``saved`` is the sidecar metadata of ``source`` (``None`` for pkls written before contracts
    existed: only the action width is checked, and a warning notes the missing metadata). Keys
    the env contract has but an older sidecar lacks (limits, scales, state_fields) are skipped
    with a warning.
    """
    if expected is None:
        return
    if saved is not None:
        compared = [key for key in _COMPARED_KEYS if key in expected]
        missing = [key for key in compared if key not in saved]
        mismatched = [
            key
            for key in compared
            if key not in missing and not _contract_values_match(saved[key], expected[key])
        ]
        if mismatched:
            details = "; ".join(
                f"{key}: recorded={saved[key]!r}, env={expected[key]!r}" for key in mismatched
            )
            raise ValueError(
                f"{source}: recorded action contract does not match the env's "
                f"(differs in {mismatched}): {details}. recorded: {describe_contract(saved)}; "
                f"env: {describe_contract(expected)}"
            )
        if missing:
            warnings.warn(
                f"{source}'s action-contract metadata predates {missing}; those were not checked",
                stacklevel=2,
            )
    expected_dim = action_dim_of(expected)
    for index, transition in enumerate(transitions):
        actual_dim = int(torch.as_tensor(transition["action"]).shape[-1])
        if actual_dim != expected_dim:
            raise ValueError(
                f"{source}: transition {index} has action dim {actual_dim}, but the env expects "
                f"{expected_dim}: {describe_contract(expected)}"
            )
    if saved is None:
        warnings.warn(
            f"{source} has no action-contract metadata ({contract_sidecar_path(source).name} "
            f"missing); only the action dim ({expected_dim}) was checked, so a same-width but "
            "different action_mode would go unnoticed",
            stacklevel=2,
        )
