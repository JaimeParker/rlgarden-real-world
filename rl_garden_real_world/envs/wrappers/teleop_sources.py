"""Adapts a robot_infra ``Teleop`` part (local ``Teleop`` or ``RemoteTeleop``) to the
``reset()``/``poll() -> TeleOpSample`` source that ``TeleopInterventionWrapper`` consumes.

The teleop's arm action is mapped into the env's ``[-1, 1]`` action space:

- ``cartesian_delta``: unchanged (the teleop already emits a normalized twist).
- ``joint_abs``: leader joints normalized onto ``joint_limits`` (``2*(q-low)/(high-low)-1``).
- ``joint_delta``: ``(q_leader - q_follower) / joint_scale_rad``, where ``q_follower`` is the
  ``joint_pos`` of the env's latest observation.

A teleop gripper target in ``[0, 1]`` becomes ``2*g-1`` (inverse of ``RealRobotEnv``'s
``(a+1)/2``). A teleop without a gripper on an env with one holds the gripper action last sent
to the env (``last_gripper_action_getter``, kept by ``TeleopInterventionWrapper``):
``RealRobotEnv`` sends an absolute width target every step, so there is no no-op value; a fixed
"open" would drop a grasped object the moment a human takes over the arm, and the measured width
would too (a binary gripper closed on a wide object measures above its open threshold).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence, Union

import numpy as np

from robot_infra.teleop.utils.telo_op_control_twist import TeleOpSample

_CARTESIAN_ARM_DIM = 6
# Comfortably above the local Teleop's fault latency (5 failed polls x 10 ms backoff).
DEFAULT_READ_TIMEOUT_S = 1.0


class TeleopContract(NamedTuple):
    """What a teleop device declares: its arm action space, arm width and gripper channel."""

    action_mode: str
    arm_action_dim: int
    has_gripper: bool


def teleop_contract_of(teleop: Any) -> TeleopContract:
    """Declaration of a ``Teleop`` or ``RemoteTeleop`` (``action_space``/``has_gripper``/``action_dim``)."""
    action_mode = getattr(teleop, "action_space", None)
    if action_mode is None:  # robot_infra predating the Teleop.action_space/has_gripper properties
        return TeleopContract(str(teleop.ACTION_SPACE), int(teleop.action_dim), bool(teleop.HAS_GRIPPER))
    return TeleopContract(str(action_mode), int(teleop.action_dim), bool(teleop.has_gripper))


TeleopResolver = Callable[[], Optional[Any]]


class TeleopPartSource:
    """``teleop`` is the part itself, or a zero-arg resolver (``lambda: robot.teleop``) for a
    robot whose teleop only exists once connected (``RemoteRobot``). With a resolver that yields
    nothing yet, ``teleop_contract`` declares the device; every newly resolved part (also after a
    reconnect) is checked against the declaration and a mismatch raises ``DeviceFault``.

    ``action_mode``/``arm_action_dim``/``has_gripper`` describe the actions this source emits:
    ``has_gripper`` is true whenever a gripper dimension is emitted, including the held gripper
    action padded in for a gripper-less teleop on a gripper env.
    """

    def __init__(
        self,
        teleop: Union[Any, TeleopResolver],
        *,
        env_contract: Mapping[str, Any],
        teleop_contract: Optional[Sequence] = None,
        joint_limits: Optional[Sequence[Sequence[float]]] = None,
        joint_scale_rad: Optional[float] = None,
        last_observation_getter: Optional[Callable[[], Optional[dict]]] = None,
        last_gripper_action_getter: Optional[Callable[[], float]] = None,
        read_timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    ) -> None:
        self._resolve: TeleopResolver = teleop if callable(teleop) else (lambda: teleop)
        resolved = self._resolve()
        declared = None if teleop_contract is None else TeleopContract(*teleop_contract)
        if resolved is None:
            if declared is None:
                raise ValueError(
                    "the robot's teleop part is not available before the robot connects "
                    "(RemoteRobot builds it from the daemon's describe); declare it with "
                    "teleop_contract=(action_mode, arm_action_dim, has_gripper) "
                    "(hil_serl: --robot-teleop-action-mode/--robot-teleop-has-gripper)"
                )
            self._contract = declared
        else:
            self._contract = teleop_contract_of(resolved)
            if declared is not None and declared != self._contract:
                raise ValueError(
                    f"declared teleop {declared} does not match the robot's teleop part "
                    f"{self._contract}"
                )
        self._verified: Any = resolved

        self._env_gripper = bool(env_contract["gripper_in_action"])
        self._pads_gripper = self._env_gripper and not self._contract.has_gripper
        self.action_mode = self._contract.action_mode
        self.arm_action_dim = self._contract.arm_action_dim
        self.has_gripper = self._contract.has_gripper or self._env_gripper
        self._read_timeout_s = read_timeout_s
        self._last_observation = last_observation_getter
        self._last_gripper_action = last_gripper_action_getter
        state_fields = tuple(env_contract.get("state_fields") or ())
        self._joint_limits: Optional[np.ndarray] = None
        self._joint_scale_rad = joint_scale_rad
        if env_contract.get("action_mode", self.action_mode) != self.action_mode:
            return  # the wrapper rejects the mode mismatch, naming both sides

        if self.action_mode == "joint_abs":
            if joint_limits is None:
                raise ValueError("a joint_abs teleop source needs joint_limits ([[low, high]] per joint)")
            self._joint_limits = np.asarray(joint_limits, dtype=np.float64)
            if self._joint_limits.shape != (self.arm_action_dim, 2):
                raise ValueError(
                    f"joint_limits has shape {self._joint_limits.shape}, expected "
                    f"({self.arm_action_dim}, 2)"
                )
            if not np.all(self._joint_limits[:, 1] > self._joint_limits[:, 0]):
                raise ValueError(f"joint_abs needs high > low for every joint, got {joint_limits}")
        if self.action_mode == "joint_delta":
            if joint_scale_rad is None or last_observation_getter is None:
                raise ValueError(
                    "a joint_delta teleop source needs joint_scale_rad and last_observation_getter "
                    "(the follower's joint_pos)"
                )
            self._require_state_field(state_fields, "joint_pos", "joint_delta intervention")
        if self._pads_gripper and last_gripper_action_getter is None:
            raise ValueError(
                "a teleop without a gripper on an env with gripper_in_action holds the last sent "
                "gripper action, so it needs last_gripper_action_getter"
            )

    @staticmethod
    def _require_state_field(state_fields: tuple, field: str, purpose: str) -> None:
        if state_fields and field not in state_fields:
            raise ValueError(
                f"{purpose} reads {field!r} from the env observation, but state_fields is "
                f"{list(state_fields)}; add {field!r} to --real_robot.state_fields"
            )

    @classmethod
    def for_env(cls, env: Any, **kwargs: Any) -> "TeleopPartSource":
        """Source over the teleop part of the robot behind ``env``'s ``RealRobotEnv``."""
        from rl_garden_real_world.action_contract import get_env_action_contract

        real_env = _find_last_observation_owner(env)
        contract = get_env_action_contract(env)
        if real_env is None or contract is None:
            raise ValueError(
                "device='robot' needs a real_robot backend env (RealRobotEnv with a robot_infra "
                "robot); franka_real has no teleop part"
            )
        robot = real_env.robot
        kwargs.setdefault("last_observation_getter", lambda: real_env.last_observation)
        return cls(lambda: robot.teleop, env_contract=contract, **kwargs)

    # -- source protocol -----------------------------------------------------------

    def reset(self) -> None:
        """Drops events latched so far (the wrapper calls it after the env reset, so presses
        during homing are discarded). No-op while the robot is not connected."""
        from robot_infra.devices.errors import DeviceNotConnected

        teleop = self._resolve()
        if teleop is None:
            return
        self._check_declaration(teleop)
        try:
            teleop.reset()
        except DeviceNotConnected:
            return

    def poll(self) -> TeleOpSample:
        """One teleop read mapped into the env action space. Read failures propagate: silently
        dropping an intervention would put policy actions into the demo buffer."""
        from robot_infra.devices.errors import DeviceFault

        teleop = self._resolve()
        if teleop is None:
            raise DeviceFault(
                "the robot has no teleop part (not connected, or no teleop: section in its yaml)",
                hint="add a teleop: section to the robot yaml and reset the env before stepping",
                device="teleop",
            )
        self._check_declaration(teleop)
        sample = teleop.read(self._read_timeout_s)
        if sample.action_space != self.action_mode:
            raise DeviceFault(
                f"teleop sample has action_space {sample.action_space!r}, expected {self.action_mode!r}",
                device=getattr(teleop, "name", "teleop"),
            )
        arm = np.clip(self._map_arm(np.asarray(sample.action, dtype=np.float64)), -1.0, 1.0)
        parts = [arm]
        gripper = float("nan")
        if self._env_gripper:
            gripper = self._gripper_action(sample, teleop)
            parts.append(np.array([gripper]))
        intervening = bool(sample.intervening)
        return TeleOpSample(
            action=np.concatenate(parts).astype(np.float32),
            twist=arm.astype(np.float32),
            gripper=gripper,
            bind_pressed=intervening,
            episode_end=bool(sample.events.get("episode_end", False)),
            intervened=intervening,
        )

    # -- mapping -------------------------------------------------------------------

    def _check_declaration(self, teleop: Any) -> None:
        if teleop is self._verified:
            return
        from robot_infra.devices.errors import DeviceFault

        actual = teleop_contract_of(teleop)
        if actual != self._contract:
            raise DeviceFault(
                f"the robot's teleop part declares {actual}, but this run was built for "
                f"{self._contract}",
                hint="fix the robot yaml's teleop: section or --robot-teleop-action-mode/"
                "--robot-teleop-has-gripper",
                device=getattr(teleop, "name", "teleop"),
            )
        self._verified = teleop

    def _map_arm(self, q: np.ndarray) -> np.ndarray:
        if self.action_mode == "cartesian_delta":
            return q
        if self.action_mode == "joint_abs":
            low, high = self._joint_limits[:, 0], self._joint_limits[:, 1]
            return 2.0 * (q - low) / (high - low) - 1.0
        follower = np.asarray(self._latest_state()["joint_pos"], dtype=np.float64)
        return (q - follower) / self._joint_scale_rad

    def _gripper_action(self, sample: Any, teleop: Any) -> float:
        if self._pads_gripper:
            return float(self._last_gripper_action())
        if sample.gripper is None:
            from robot_infra.devices.errors import DeviceFault

            raise DeviceFault(
                "teleop declares a gripper but its sample carries none",
                device=getattr(teleop, "name", "teleop"),
            )
        return float(np.clip(2.0 * float(sample.gripper) - 1.0, -1.0, 1.0))

    def _latest_state(self) -> dict:
        observation = self._last_observation()
        if observation is None:
            raise RuntimeError("teleop source polled before the env was reset (no observation yet)")
        return observation["state"]


def _find_last_observation_owner(env: Any) -> Optional[Any]:
    """Walk ``.env`` down the wrapper chain to the ``RealRobotEnv`` (whose ``last_observation``
    is the raw nested dict, before frame stacking or rotvec conversion)."""
    node, seen = env, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(getattr(type(node), "last_observation", None), property):
            return node
        node = vars(node).get("env") if hasattr(node, "__dict__") else None
    return None
