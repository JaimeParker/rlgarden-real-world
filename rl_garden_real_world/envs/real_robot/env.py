"""RealRobotEnv: a numpy gym.Env over any robot_infra ``Robot`` (local or daemon-served).

Observation is a nested dict ``{"state": {field: float32}, "frames": {cam: uint8 HxWx3}}``;
the rl-garden backend (``envs/backends/real_robot.py``) flattens it to ``state``/``rgb_<cam>``
torch tensors. Action is ``[-1, 1]^D``: 6 (xyz + rotvec) for ``cartesian_delta`` or
``num_joints`` for the joint modes, plus one gripper dimension when ``gripper_in_action``.

Unlike ``FrankaRealEnv``, this env paces itself to ``control_hz``: each ``step()`` sleeps the
remainder of the control period since the previous step's command was sent.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from rl_garden_real_world.envs.real_robot.config import RealRobotEnvConfig

_CARTESIAN_ACTION_DIM = 6
_GRIPPER_OPEN_COMMAND = 1.0


def _state_field_shape(field: str, num_joints: int) -> tuple[int, ...]:
    return {
        "tcp_pose": (7,),
        "tcp_vel": (6,),
        "joint_pos": (num_joints,),
        "joint_vel": (num_joints,),
        "gripper_pos": (1,),
        "tcp_force": (3,),
        "tcp_torque": (3,),
    }[field]


def _build_robot(config: RealRobotEnvConfig):
    """Construct (never connect) the robot described by ``config``."""
    if config.robot is not None:
        import robot_infra.devices.backends  # noqa: F401  (registers backends; SDK imports are lazy)
        from robot_infra.devices import Robot

        return Robot.from_dict(config.robot)
    from robot_infra.server import RemoteRobot, RemoteRobotConfig

    return RemoteRobot(RemoteRobotConfig(address=config.remote, token=config.token))


class RealRobotEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: RealRobotEnvConfig, *, robot: Any = None) -> None:
        self.config = config
        self.robot = robot if robot is not None else _build_robot(config)
        self.task_description: str = ""
        self.render_mode = "rgb_array"

        self._frame_sizes = {cam: self._frame_size(cam) for cam in config.cameras}
        num_joints = config.num_joints
        state_space = spaces.Dict(
            [
                (
                    field,
                    spaces.Box(
                        -np.inf, np.inf, shape=_state_field_shape(field, num_joints), dtype=np.float32
                    ),
                )
                for field in config.state_fields
            ]
        )
        frame_space = spaces.Dict(
            [
                (cam, spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8))
                for cam, (h, w) in self._frame_sizes.items()
            ]
        )
        self.observation_space = spaces.Dict([("state", state_space), ("frames", frame_space)])
        arm_dim = _CARTESIAN_ACTION_DIM if config.action_mode == "cartesian_delta" else num_joints
        action_dim = arm_dim + (1 if config.gripper_in_action else 0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)

        self._joint_limits = (
            np.asarray(config.joint_limits, dtype=np.float64)
            if config.joint_limits is not None
            else None
        )
        self._period_s = 1.0 / config.control_hz
        self._validated = False
        self._last_raw: Optional[dict] = None
        self._last_obs: Optional[dict] = None
        self._last_send_time: Optional[float] = None
        self._elapsed_steps = 0

    def _frame_size(self, cam: str) -> tuple[int, int]:
        if self.config.image_size is not None:
            return self.config.image_size
        camera_configs = {c.name: c for c in getattr(self.robot.config, "cameras", [])}
        if cam not in camera_configs:
            raise ValueError(
                f"camera {cam!r} is not in the robot config (cameras: {sorted(camera_configs)}); "
                "add it there or set image_size"
            )
        return camera_configs[cam].height, camera_configs[cam].width

    # -- connection ----------------------------------------------------------------

    def _connect_and_validate(self) -> None:
        if self._validated:
            return
        if not self.robot.is_connected:
            self.robot.connect()
        try:
            self._validate_robot()
        except BaseException:
            self.robot.disconnect()
            raise
        self._validated = True

    def _validate_robot(self) -> None:
        from robot_infra.devices.errors import DeviceFault

        robot, config = self.robot, self.config

        def fault(message: str, hint: str) -> DeviceFault:
            return DeviceFault(message, hint=hint, device=robot.name)

        arm = robot.arm
        if arm.num_joints != config.num_joints:
            raise fault(
                f"robot arm has {arm.num_joints} joints, env config expects {config.num_joints}",
                f"set num_joints={arm.num_joints}",
            )
        available = set(arm.SUPPORTED_STATE_FIELDS)
        if robot.gripper is not None:
            available.add("gripper_pos")
        required = list(config.state_fields)
        required.append("tcp_pose" if config.action_mode == "cartesian_delta" else "joint_pos")
        missing = sorted({f for f in required if f not in available})
        if missing:
            raise fault(
                f"robot does not report state fields {missing} (available: {sorted(available)})",
                "drop them from state_fields (Piper: tcp_pose, joint_pos, joint_vel, gripper_pos)"
                " or pick an action_mode the robot supports",
            )
        if config.gripper_in_action and robot.gripper is None:
            raise fault("robot has no gripper", "set gripper_in_action=False")
        unknown_cams = [cam for cam in config.cameras if cam not in robot.cameras]
        if unknown_cams:
            raise fault(
                f"robot has no cameras {unknown_cams} (available: {sorted(robot.cameras)})",
                "fix the camera names in the env config",
            )

    def _home_joints(self) -> np.ndarray:
        from robot_infra.devices.errors import DeviceFault

        home = self.config.home_joints
        if home is None:
            home = getattr(self.robot.config, "home_joints", None)
        if home is None:
            raise DeviceFault(
                "no home joint configuration for reset",
                hint="set home_joints in the env config",
                device=self.robot.name,
            )
        return np.asarray(home, dtype=np.float64)

    def _send_and_observe(self, action: dict) -> dict:
        """One command + observation; a single round trip on a ``RemoteRobot``."""
        step = getattr(self.robot, "step", None)
        if step is not None:
            return step(action)
        self.robot.send_action(action)
        return self.robot.get_observation()

    # -- gym API ---------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        del options
        from robot_infra.devices.arm import ControlMode

        self._connect_and_validate()
        arm = self.robot.arm
        arm.set_control_mode(ControlMode.IDLE)
        arm.move_to_joints(self._home_joints(), blocking=True)
        if self.robot.gripper is not None:
            # Through the action path, not gripper.open(): a binary gripper skips a command equal
            # to the last one it was sent, so a direct open() would desync that memory.
            self._send_and_observe({"gripper": {"gripper": np.array([_GRIPPER_OPEN_COMMAND])}})
        if self.config.reset_settle_s > 0:
            time.sleep(self.config.reset_settle_s)
        servo = (
            ControlMode.CARTESIAN_SERVO
            if self.config.action_mode == "cartesian_delta"
            else ControlMode.JOINT_SERVO
        )
        arm.set_control_mode(servo)
        raw = self.robot.get_observation()
        self._elapsed_steps = 0
        self._last_send_time = None
        return self._observe(raw), {}

    def step(self, action):
        if self._last_raw is None:
            raise RuntimeError("RealRobotEnv.step() called before reset()")
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")

        command: dict[str, Any] = {"arm": self._arm_command(action)}
        if self.config.gripper_in_action:
            command["gripper"] = {"gripper": np.array([(action[-1] + 1.0) / 2.0])}

        if self._last_send_time is not None:
            remaining = self._period_s - (time.monotonic() - self._last_send_time)
            if remaining > 0:
                time.sleep(remaining)
        self._last_send_time = time.monotonic()
        raw = self._send_and_observe(command)

        self._elapsed_steps += 1
        obs = self._observe(raw)
        truncated = (
            self.config.max_episode_steps is not None
            and self._elapsed_steps >= self.config.max_episode_steps
        )
        client = getattr(self.robot, "client", None)
        info = {
            "tcp_pose": np.asarray(raw["arm"]["tcp_pose"], dtype=np.float32).copy()
            if "tcp_pose" in raw["arm"]
            else None,
            "rtt_ms": getattr(client, "last_rtt_ms", None),
            "elapsed_steps": self._elapsed_steps,
        }
        return obs, 0.0, False, bool(truncated), info

    def _arm_command(self, action: np.ndarray) -> dict[str, np.ndarray]:
        config = self.config
        arm_state = self._last_raw["arm"]
        if config.action_mode == "cartesian_delta":
            from scipy.spatial.transform import Rotation

            pose = np.asarray(arm_state["tcp_pose"], dtype=np.float64)
            xyz = pose[:3] + action[:3] * config.xyz_scale_m
            if config.safety_box is not None:
                box = np.asarray(config.safety_box, dtype=np.float64)
                xyz = np.clip(xyz, box[:3], box[3:])
            rotation = Rotation.from_rotvec(action[3:6] * config.rot_scale_rad) * Rotation.from_quat(
                pose[3:7]
            )
            return {"tcp_pose": np.concatenate([xyz, rotation.as_quat()])}

        num_joints = config.num_joints
        if config.action_mode == "joint_delta":
            q = np.asarray(arm_state["joint_pos"], dtype=np.float64) + (
                action[:num_joints] * config.joint_scale_rad
            )
        else:  # joint_abs; joint_limits is guaranteed by the config
            low, high = self._joint_limits[:, 0], self._joint_limits[:, 1]
            q = low + (action[:num_joints] + 1.0) / 2.0 * (high - low)
        if self._joint_limits is not None:
            q = np.clip(q, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return {"joint_pos": q}

    def _observe(self, raw: dict) -> dict:
        self._last_raw = raw
        state = {}
        for field in self.config.state_fields:
            source = raw["gripper"] if field == "gripper_pos" else raw["arm"]
            state[field] = np.asarray(source[field], dtype=np.float32).reshape(
                self.observation_space["state"][field].shape
            )
        frames = {
            cam: self._fit_frame(np.asarray(raw["cameras"][cam]["color"], dtype=np.uint8), size)
            for cam, size in self._frame_sizes.items()
        }
        self._last_obs = {"state": state, "frames": frames}
        return self._last_obs

    @staticmethod
    def _fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        height, width = size
        if frame.shape[:2] == (height, width):
            return frame
        try:
            from PIL import Image
        except ImportError:
            from robot_infra.devices.errors import SdkMissing

            raise SdkMissing(
                f"resizing camera frames {frame.shape[:2]} -> {size} needs Pillow",
                hint="pip install pillow",
                device="RealRobotEnv",
            ) from None
        return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))

    def render(self):
        if self._last_obs is None or not self._last_obs["frames"]:
            return None
        return next(iter(self._last_obs["frames"].values()))

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()
        self._validated = False
        self._last_raw = None


class RealRobotVectorEnv(gym.Env):
    """Batch-of-1 torch view of a ``RealRobotEnv`` satisfying rl-garden's observation vocabulary.

    ``state`` is ``config.state_fields`` concatenated in config order (``(1, D)`` float32) and
    each camera becomes ``rgb_<cam>`` (``(1, H, W, 3)`` uint8). Tensors live on CPU. Like
    ``FrankaRealEnv`` there is no autoreset: the caller resets on ``terminated``/``truncated``
    (``ActorLoop`` does), so the robot is never homed twice per episode boundary.
    """

    num_envs = 1

    def __init__(self, env: RealRobotEnv, *, include_state: bool = True) -> None:
        self.env = env
        self.include_state = include_state
        entries: list[tuple[str, spaces.Space]] = [
            (f"rgb_{cam}", env.observation_space["frames"][cam]) for cam in env.config.cameras
        ]
        if include_state:
            state_dim = sum(
                env.observation_space["state"][field].shape[0] for field in env.config.state_fields
            )
            entries.append(
                ("state", spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32))
            )
        self.single_observation_space = spaces.Dict(entries)
        self.single_action_space = env.action_space
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)
        self.metadata = env.metadata

    @property
    def robot(self):
        return self.env.robot

    @property
    def task_description(self) -> str:
        return self.env.task_description

    def _to_torch(self, obs: dict) -> dict[str, torch.Tensor]:
        out = {
            f"rgb_{cam}": torch.from_numpy(np.require(obs["frames"][cam], requirements=["C", "W"])).unsqueeze(0)
            for cam in self.env.config.cameras
        }
        if self.include_state:
            state = np.concatenate([obs["state"][field] for field in self.env.config.state_fields])
            out["state"] = torch.from_numpy(state.astype(np.float32, copy=False)).unsqueeze(0)
        return out

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_torch(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action).reshape(-1))
        return (
            self._to_torch(obs),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([terminated], dtype=torch.bool),
            torch.tensor([truncated], dtype=torch.bool),
            info,
        )

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
