"""Gripper-always-closed action wrapper (HIL-SERL's ``GripperCloseEnv``,
``3rd_party/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py``):
removes the gripper dimension (default: 7D -> 6D) for tasks that never
open/close the gripper, re-inserting ``closed_value`` there before
forwarding to the wrapped env.

Lives in ``real_world/`` rather than ``rl_garden/envs/wrappers/``: the 7D,
last-dim-is-gripper action shape it asserts is ``FrankaRealEnv``'s own
convention (also the convention ``RLPDHybridPolicy``'s hybrid
continuous+discrete action head was built to match -- see its module
docstring), not a generic rl-garden convention -- no ManiSkill or RoboTwin
sim task uses it (RoboTwin's own gripper handling is an unrelated dual-arm,
continuous-delta scheme). Unlike ``MultiStageBinaryRewardClassifierWrapper``/
``ActionChunkWrapper``/``ImageFrameStackWrapper``, this wrapper has no
sim-reusable, shape-agnostic form -- it's specific to the real-robot
single-Franka-arm action contract.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space


class GripperCloseEnv(gym.ActionWrapper):
    """Hides the gripper dimension (``gripper_dim``, default last) from the
    policy and always forwards ``closed_value`` for it. The base env's gripper
    threshold decides open/close, so 0.0 means "closed" under the Franka
    [-1, 1] convention; a continuous gripper passes its own closed action value.
    """

    def __init__(
        self, env: gym.Env, gripper_dim: int = -1, closed_value: float = 0.0
    ) -> None:
        super().__init__(env)
        single = env.single_action_space
        if len(single.shape) != 1 or single.shape[0] < 2:
            raise ValueError(
                f"GripperCloseEnv requires a 1-D action space with >= 2 dims, got {single.shape}."
            )
        full_dim = single.shape[0]
        if not -full_dim <= gripper_dim < full_dim:
            raise ValueError(f"gripper_dim={gripper_dim} out of range for {full_dim}-D action space.")
        self._gripper_dim = gripper_dim % full_dim
        self._closed_value = closed_value
        self._kept_dims = [i for i in range(full_dim) if i != self._gripper_dim]
        self.single_action_space = spaces.Box(
            low=single.low[self._kept_dims], high=single.high[self._kept_dims], dtype=single.dtype
        )
        self.action_space = batch_space(self.single_action_space, env.num_envs)

    def __getattr__(self, name: str):
        # See RewardClassifierWrapper -- gymnasium.Wrapper (>=1.0) no longer
        # forwards arbitrary attributes to self.env.
        return getattr(self.env, name)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        full = action.new_full((*action.shape[:-1], len(self._kept_dims) + 1), self._closed_value)
        full[..., self._kept_dims] = action
        return full

    def step(self, action: torch.Tensor):
        # Overrides step() directly (not just action()) since
        # info["intervene_action"] also needs truncating to match the
        # shrunk action space -- gym.ActionWrapper's default step() only
        # transforms the action, not info.
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        if "intervene_action" in info:
            info = dict(info)
            info["intervene_action"] = info["intervene_action"][..., self._kept_dims]
        return obs, reward, terminated, truncated, info
