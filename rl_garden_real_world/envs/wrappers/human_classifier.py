"""Human-judged success wrapper (HIL-SERL's ``HumanClassifierWrapper``,
``3rd_party/hil-serl/serl_robot_infra/franka_env/envs/wrappers.py``): blocks
on a keyboard prompt at episode end for a human to label success/failure, an
alternative to a learned reward classifier for tasks where training one
isn't worth it yet. Blocking on ``input()`` only makes sense in a live
teleop/eval session, so this lives in ``real_world/`` rather than
``rl_garden/envs/wrappers/`` (unlike ``RewardClassifierWrapper``, which is
hardware-agnostic).
"""
from __future__ import annotations

import gymnasium as gym
import torch


class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, prompt: str = "Success? (1/0): ") -> None:
        super().__init__(env)
        self.prompt = prompt

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        if bool(terminated) or bool(truncated):
            while True:
                answer = input(self.prompt).strip()
                if answer in ("0", "1"):
                    break
            success = answer == "1"
            reward = torch.full_like(reward, float(success))
            info["succeed"] = reward
        return obs, reward, terminated, truncated, info
