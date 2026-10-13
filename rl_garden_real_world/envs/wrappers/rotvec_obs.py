"""Converts the quaternion slice of a Franka real-robot env's ``state``
observation into a 3D rotation vector, matching the rotation representation
the action side already uses (``FrankaRealEnv.step()``'s rotvec deltas).
Optional and off by default -- HIL-SERL converts to Euler via
``Quat2EulerWrapper`` instead, but rotvec avoids Euler's gimbal-lock/wraparound
special-casing and stays consistent with this repo's own action convention.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

# Default quaternion position in the flat "state" layout of Franka-style envs:
# pose (xyz[0:3] + quat_xyzw[3:7]) + vel(6) + gripper_pos(1) + force(3) +
# torque(3). Other robots pass their own ``quat_slice``.
_DEFAULT_QUAT_SLICE = (3, 7)
_SMALL_ANGLE_EPS = 1e-8


def quat_xyzw_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Pure-torch xyzw quaternion -> rotation vector (``(..., 4) -> (..., 3)``).

    Matches ``scipy.spatial.transform.Rotation.as_rotvec``: ``q`` and ``-q``
    give the same rotation (the shortest-angle representative, angle in
    ``[0, pi]``). Input need not be exactly unit norm. Output keeps the input's
    device and dtype.
    """
    xyz, w = q[..., :3], q[..., 3:4]
    # Canonicalise to w >= 0 so the angle stays in [0, pi].
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    xyz, w = xyz * sign, w * sign
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    small = sin_half < _SMALL_ANGLE_EPS
    # angle / sin(angle/2) -> 2 / w as the angle -> 0; avoids dividing by ~0.
    scale = torch.where(small, 2.0 / w.clamp_min(_SMALL_ANGLE_EPS), angle / sin_half.clamp_min(_SMALL_ANGLE_EPS))
    return xyz * scale


class RotvecObsWrapper(gym.ObservationWrapper):
    """Shrinks ``obs["state"]`` by one dim by replacing the 4D xyzw quaternion
    at ``quat_slice`` (default ``(3, 7)``) with its 3D rotation vector."""

    def __init__(
        self, env: gym.Env, quat_slice: tuple[int, int] = _DEFAULT_QUAT_SLICE
    ) -> None:
        super().__init__(env)
        start, end = quat_slice
        if end - start != 4:
            raise ValueError(f"quat_slice must span 4 dims, got {quat_slice}.")
        self._quat_start, self._quat_end = start, end
        state_space = env.single_observation_space["state"]
        new_state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_space.shape[0] - 1,),
            dtype=state_space.dtype,
        )
        new_spaces = dict(env.single_observation_space.spaces)
        new_spaces["state"] = new_state_space
        self.single_observation_space = spaces.Dict(new_spaces)
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)

    def __getattr__(self, name: str):
        # See RewardClassifierWrapper -- gymnasium.Wrapper (>=1.0) no longer
        # forwards arbitrary attributes to self.env, but this repo's env
        # backend contract (num_envs, single_observation_space, ...) relies
        # on direct attribute access.
        return getattr(self.env, name)

    def observation(self, obs: dict) -> dict:
        state = obs["state"]
        before = state[..., : self._quat_start]
        rotvec = quat_xyzw_to_rotvec(state[..., self._quat_start : self._quat_end])
        after = state[..., self._quat_end :]
        return {**obs, "state": torch.cat([before, rotvec, after], dim=-1)}
