"""Human-in-the-loop intervention wrapper (the design doc's "SpaceMouse
intervention" component).

Built on rl-garden's own teleop bridge
(``robot_infra/teleop/utils/telo_op_control_twist.py``'s
``EETwistTeleOpWrapper``) for the ``"pico"`` device, and on
``robot_infra/teleop/spacemouse/SpaceMouseTeleOpWrapper`` (ported from
HIL-SERL) for ``"spacemouse"`` -- the two devices work on fundamentally
different mechanisms (a ZMQ-fed absolute pose vs. a locally HID-polled rate
device), so they're two separate device-source classes rather than one
class force-fit to both, but both produce the same
``TeleOpSample``-shaped output this wrapper consumes. The 7D
``TeleOpSample.action`` (6D EE twist + gripper) matches ``FrankaRealEnv``'s
action convention exactly, so no conversion is needed beyond dtype/device
and adding the batch-of-1 leading dim.

``"robot"`` uses the teleop part of the ``RealRobotEnv``'s robot (the robot yaml's
``teleop:`` section, local or daemon-served) through ``TeleopPartSource``, which maps
cartesian or joint teleop actions into the env's action space. ``teleop_kwargs`` then go to
``TeleopPartSource.for_env`` (``teleop_contract``, ``joint_limits``, ``joint_scale_rad``,
``read_timeout_s``); a teleop read failure propagates out of ``step()``. The wrapper keeps the
gripper action last sent to the env (+1, open, after a reset), which a gripper-less teleop holds.

An ``episode_end`` event (operator button) ends the episode: ``step()`` returns
``truncated=True`` with ``info["human_episode_end"]`` (as in HIL-SERL), so actor loops reset
without treating it as terminal.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

import gymnasium as gym
import torch

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper
from rl_garden_real_world.action_contract import (
    LEGACY_ACTION_CONTRACT,
    describe_contract,
    get_env_action_contract,
)

TeleopSource = Union[EETwistTeleOpWrapper, "SpaceMouseTeleOpWrapper"]  # noqa: F821
# RealRobotEnv opens the gripper on reset.
_GRIPPER_ACTION_AFTER_RESET = 1.0


class TeleopInterventionWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        teleop: Optional[TeleopSource] = None,
        device: Literal["pico", "spacemouse", "robot"] = "pico",
        **teleop_kwargs: Any,
    ) -> None:
        super().__init__(env)
        # Set before the source is built: __getattr__ forwards unknown attributes to the env.
        self._last_gripper_action = _GRIPPER_ACTION_AFTER_RESET
        if teleop is not None:
            self.teleop = teleop
        elif device == "pico":
            self.teleop = EETwistTeleOpWrapper(device="pico", **teleop_kwargs)
        elif device == "spacemouse":
            from robot_infra.teleop.spacemouse import SpaceMouseTeleOpWrapper

            self.teleop = SpaceMouseTeleOpWrapper(**teleop_kwargs)
        elif device == "robot":
            from rl_garden_real_world.envs.wrappers.teleop_sources import TeleopPartSource

            teleop_kwargs.setdefault("last_gripper_action_getter", lambda: self._last_gripper_action)
            self.teleop = TeleopPartSource.for_env(env, **teleop_kwargs)
        else:
            raise ValueError(f"device must be 'pico', 'spacemouse' or 'robot', got {device!r}.")
        self._action_dim = self._validate_against_env(env, self.teleop)

    @staticmethod
    def _validate_against_env(env: gym.Env, teleop: Any) -> Optional[int]:
        """Checks the teleop source's declared action space against the env's contract.

        Sources declare it through optional ``action_mode``/``arm_action_dim``/``has_gripper``
        attributes; the two cartesian devices declare nothing and get the defaults. Compares
        ``action_mode`` and not just widths: a 6-joint ``joint_delta`` arm plus gripper is 7-D,
        the same width as a cartesian twist plus gripper.
        """
        contract = get_env_action_contract(env) or LEGACY_ACTION_CONTRACT
        teleop_mode = getattr(teleop, "action_mode", "cartesian_delta")
        teleop_arm_dim = int(getattr(teleop, "arm_action_dim", 6))
        teleop_has_gripper = bool(getattr(teleop, "has_gripper", True))
        teleop_dim = teleop_arm_dim + int(teleop_has_gripper)
        if teleop_mode != contract["action_mode"]:
            raise ValueError(
                f"teleop source produces action_mode={teleop_mode!r} ({teleop_dim}-D) but the env "
                f"expects action_mode={contract['action_mode']!r} ({describe_contract(contract)}). "
                "Use --real_robot.action_mode cartesian_delta, or a teleop source that declares "
                "a joint action space."
            )
        if not contract["gripper_in_action"]:
            raise ValueError(
                "hil_serl currently requires gripper_in_action=true (RLPDHybrid treats the last "
                "action dim as the discrete gripper); non-hybrid algorithms without a gripper "
                f"are not supported yet. Env contract: {describe_contract(contract)}"
            )
        action_space = getattr(env, "single_action_space", None)
        if action_space is not None and teleop_dim != action_space.shape[-1]:
            raise ValueError(
                f"teleop source produces {teleop_dim}-D actions (arm {teleop_arm_dim} + gripper "
                f"{int(teleop_has_gripper)}, action_mode={teleop_mode!r}) but the env action space "
                f"is {action_space.shape[-1]}-D ({describe_contract(contract)})"
            )
        return None if action_space is None else int(action_space.shape[-1])

    def __getattr__(self, name: str):
        # gymnasium.Wrapper (>=1.0) no longer forwards arbitrary attributes to
        # ``self.env`` -- but this repo's env-backend contract (num_envs,
        # single_observation_space, ...) relies on direct attribute access,
        # not ``get_wrapper_attr()``, so this wrapper must still be
        # transparent to algorithm code built against the unwrapped env.
        return getattr(self.env, name)

    def reset(self, **kwargs):
        # Env first: events latched while the robot homes are then dropped by the teleop reset.
        result = self.env.reset(**kwargs)
        self.teleop.reset()
        self._last_gripper_action = _GRIPPER_ACTION_AFTER_RESET
        return result

    def step(self, action: torch.Tensor):
        sample = self.teleop.poll()
        info_extra = {}
        if sample.intervened:
            device = action.device if isinstance(action, torch.Tensor) else torch.device("cpu")
            action = torch.as_tensor(sample.action, device=device, dtype=torch.float32).reshape(
                1, -1 if self._action_dim is None else self._action_dim
            )
            info_extra["intervene_action"] = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        # The env clips actions to [-1, 1], so this is the gripper action it actually sent.
        self._last_gripper_action = float(
            torch.as_tensor(action).reshape(-1)[-1].clamp(-1.0, 1.0)
        )
        info = {**info, **info_extra, "human_episode_end": sample.episode_end}
        if sample.episode_end:
            truncated = (
                torch.ones_like(truncated, dtype=torch.bool)
                if isinstance(truncated, torch.Tensor)
                else True
            )
        return obs, reward, terminated, truncated, info
