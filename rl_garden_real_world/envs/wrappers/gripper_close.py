"""Gripper-always-closed action wrapper (HIL-SERL's ``GripperCloseEnv``,
``3rd_party/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py``):
truncates a 7D action space (6D pose delta + 1D gripper) to 6D for tasks
that never open/close the gripper, zero-padding the gripper dimension back
before forwarding to the wrapped env.

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
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        single = env.single_action_space
        assert single.shape == (7,), (
            f"GripperCloseEnv requires a 7D action space, got {single.shape}."
        )
        self.single_action_space = spaces.Box(
            low=single.low[:6], high=single.high[:6], dtype=single.dtype
        )
        self.action_space = batch_space(self.single_action_space, env.num_envs)

    def __getattr__(self, name: str):
        # See RewardClassifierWrapper -- gymnasium.Wrapper (>=1.0) no longer
        # forwards arbitrary attributes to self.env.
        return getattr(self.env, name)

    def action(self, action: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(*action.shape[:-1], 1, device=action.device, dtype=action.dtype)
        return torch.cat([action, pad], dim=-1)

    def step(self, action: torch.Tensor):
        # Overrides step() directly (not just action()) since
        # info["intervene_action"] also needs truncating to match the
        # shrunk action space -- gym.ActionWrapper's default step() only
        # transforms the action, not info.
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        if "intervene_action" in info:
            info = dict(info)
            info["intervene_action"] = info["intervene_action"][..., :6]
        return obs, reward, terminated, truncated, info
