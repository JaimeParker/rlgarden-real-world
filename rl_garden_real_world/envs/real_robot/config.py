"""Config for the generic robot_infra-backed real-robot env.

Pure dataclass (no robot_infra/scipy/torch imports) so it can be nested in CLI
args and inspected without any robot dependency installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ACTION_MODES = ("cartesian_delta", "joint_delta", "joint_abs")
STATE_FIELDS = (
    "tcp_pose",
    "tcp_vel",
    "joint_pos",
    "joint_vel",
    "gripper_pos",
    "tcp_force",
    "tcp_torque",
)
_SAFETY_BOX_LENGTH = 6


@dataclass
class RealRobotEnvConfig:
    # Exactly one of: a local robot_infra RobotConfig dict (must carry "type"),
    # or the address of a robot_infra daemon. Checked by ``require_robot_source``
    # when the env is built, so the default-constructed config (CLI group) stays valid.
    robot: Optional[dict] = None
    remote: Optional[str] = None
    token: Optional[str] = None
    num_joints: int = 7
    # Concatenated in this order into the flat backend ``state`` vector.
    state_fields: tuple[str, ...] = STATE_FIELDS
    cameras: list[str] = field(default_factory=list)
    # (H, W) every camera frame is resized to; None keeps the camera's native size.
    image_size: Optional[tuple[int, int]] = None
    action_mode: Literal["cartesian_delta", "joint_delta", "joint_abs"] = "cartesian_delta"
    control_hz: float = 10.0
    xyz_scale_m: float = 0.02
    rot_scale_rad: float = 0.1
    joint_scale_rad: float = 0.05
    # [xmin, ymin, zmin, xmax, ymax, zmax]; clips the cartesian target position.
    safety_box: Optional[list[float]] = None
    # One [low, high] pair per joint; clips joint targets, required by joint_abs.
    joint_limits: Optional[list[list[float]]] = None
    home_joints: Optional[list[float]] = None
    gripper_in_action: bool = True
    max_episode_steps: Optional[int] = None
    reset_settle_s: float = 0.5

    def __post_init__(self) -> None:
        self.state_fields = tuple(self.state_fields)
        self.cameras = list(self.cameras)
        if self.image_size is not None:
            self.image_size = tuple(self.image_size)
        if self.action_mode not in ACTION_MODES:
            raise ValueError(
                f"action_mode {self.action_mode!r} invalid; choose from {list(ACTION_MODES)}"
            )
        if self.num_joints <= 0:
            raise ValueError(f"num_joints must be positive, got {self.num_joints}")
        if self.joint_limits is not None:
            shape_ok = len(self.joint_limits) == self.num_joints and all(
                len(pair) == 2 for pair in self.joint_limits
            )
            if not shape_ok:
                raise ValueError(
                    f"joint_limits must have shape ({self.num_joints}, 2) "
                    "([[low, high], ...] per joint)"
                )
            if any(low > high for low, high in self.joint_limits):
                raise ValueError("joint_limits entries must satisfy low <= high")
        if self.action_mode == "joint_abs" and self.joint_limits is None:
            raise ValueError(
                "action_mode 'joint_abs' maps [-1, 1] onto joint_limits; set joint_limits"
            )
        if self.safety_box is not None and len(self.safety_box) != _SAFETY_BOX_LENGTH:
            raise ValueError(
                "safety_box must be [xmin, ymin, zmin, xmax, ymax, zmax], "
                f"got {len(self.safety_box)} entries"
            )
        unknown = [f for f in self.state_fields if f not in STATE_FIELDS]
        if unknown:
            raise ValueError(f"unknown state_fields {unknown}; choose from {list(STATE_FIELDS)}")
        if not self.state_fields:
            raise ValueError("state_fields must not be empty")
        if self.home_joints is not None and len(self.home_joints) != self.num_joints:
            raise ValueError(
                f"home_joints must have {self.num_joints} entries, got {len(self.home_joints)}"
            )
        if self.image_size is not None and (
            len(self.image_size) != 2 or min(self.image_size) <= 0
        ):
            raise ValueError(f"image_size must be a positive (H, W), got {self.image_size}")
        if self.remote is not None and self.cameras and self.image_size is None:
            raise ValueError(
                "remote robots report camera sizes only after connect; set image_size "
                "so the observation space is known at construction"
            )
        if self.control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {self.control_hz}")
        if self.max_episode_steps is not None and self.max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be positive or None, got {self.max_episode_steps}"
            )

    def require_robot_source(self) -> None:
        if (self.robot is None) == (self.remote is None):
            raise ValueError(
                "set exactly one of `robot` (local RobotConfig dict) or `remote` "
                "(robot_infra daemon address)"
            )
