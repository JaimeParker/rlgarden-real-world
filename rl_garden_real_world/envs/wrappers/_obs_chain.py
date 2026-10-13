"""Observation-wrapper chain shared by HIL-SERL training and deploy.

Both must stack frames and convert quaternions identically, or a deployed policy sees a
different observation layout than it trained on.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import gymnasium as gym

from rl_garden_real_world.envs.real_robot.config import RealRobotEnvConfig

_LOGGER = logging.getLogger(__name__)
_TCP_POSE_FIELD = "tcp_pose"
_TCP_POSE_QUAT_OFFSET = 3
_QUAT_DIM = 4


def tcp_pose_quat_slice(config: RealRobotEnvConfig) -> Optional[tuple[int, int]]:
    """Where the xyzw quaternion sits in the flat ``state`` vector, or ``None`` without ``tcp_pose``."""
    from rl_garden_real_world.envs.real_robot.env import _state_field_shape

    offset = 0
    for field in config.state_fields:
        if field == _TCP_POSE_FIELD:
            start = offset + _TCP_POSE_QUAT_OFFSET
            return (start, start + _QUAT_DIM)
        offset += _state_field_shape(field, config.num_joints)[0]
    return None


def apply_obs_chain(
    env: gym.Env,
    *,
    frame_stack: int,
    image_keys: Sequence[str],
    convert_obs_rotation: bool,
    quat_slice: Optional[tuple[int, int]],
) -> gym.Env:
    """Frame stacking (innermost) then rotvec conversion, each only when requested.

    ``image_keys`` empty means every non-``state`` observation key is a camera.
    """
    from rl_garden.envs.wrappers.frame_stack import ImageFrameStackWrapper
    from rl_garden_real_world.envs.wrappers.rotvec_obs import RotvecObsWrapper

    if frame_stack > 1:
        # FrankaRealEnv/RealRobotVectorEnv have no _init_raw_obs/update_obs_space (ManiSkill-only
        # APIs) and their camera keys are not necessarily "rgb*"/"depth*"-prefixed, so
        # image_keys must be passed explicitly.
        stack_keys = tuple(image_keys) or tuple(
            k for k in env.single_observation_space.spaces if k != "state"
        )
        env = ImageFrameStackWrapper(env, frame_stack=frame_stack, image_keys=stack_keys)

    if convert_obs_rotation:
        if quat_slice is None:
            _LOGGER.warning(
                "convert_obs_rotation skipped: state_fields has no tcp_pose, so there is no "
                "quaternion to convert"
            )
        else:
            env = RotvecObsWrapper(env, quat_slice=quat_slice)
    return env
