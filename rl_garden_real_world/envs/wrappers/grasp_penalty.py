"""Gripper-flip reward-shaping wrapper (HIL-SERL's ``grasp_penalty``): flags
``info["grasp_penalty"]`` whenever the gripper action flips open<->closed
relative to the previous step, for ``RLPDHybrid``'s discrete critic target to
add on top of the raw environment reward (``use_grasp_penalty=True``).

Ported from ``3rd_party/hil-serl/examples/experiments/usb_pickup_insertion/
wrapper.py``'s task-specific ``GripperPenaltyWrapper`` -- the one HIL-SERL's
own ``train_rlpd.py``/``sac_hybrid_single.py`` actually consume via
``info["grasp_penalty"]`` (an additive side channel), not the base
``franka_env/envs/wrappers.py::GripperPenaltyWrapper``, which instead mutates
``reward`` directly and is never wired into the real training loop.

Must wrap *outside* ``TeleopInterventionWrapper`` in the wrapper stack (i.e.
applied after it) so ``info.get("intervene_action")`` -- the actually-executed
action under human intervention -- is already present when this wrapper's
``step()`` runs.
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import torch


class GraspPenaltyWrapper(gym.Wrapper):
    def __init__(
        self, env: gym.Env, penalty: float = -0.05, gripper_action_idx: int = -1
    ) -> None:
        super().__init__(env)
        self.penalty = penalty
        self.gripper_action_idx = gripper_action_idx
        self._last_gripper_open: Optional[torch.Tensor] = None  # per-env bool

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_gripper_open = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        executed_action = info.get("intervene_action", action)
        is_open = executed_action[..., self.gripper_action_idx] <= 0
        if self._last_gripper_open is None:
            flipped = torch.zeros_like(is_open, dtype=torch.bool)
        else:
            flipped = is_open != self._last_gripper_open
        self._last_gripper_open = is_open
        info = dict(info)
        info["grasp_penalty"] = torch.where(
            flipped, torch.full_like(reward, self.penalty), torch.zeros_like(reward)
        )
        return obs, reward, terminated, truncated, info
