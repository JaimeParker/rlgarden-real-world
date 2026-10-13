"""Generic robot_infra real-robot env backend — registered as ``"real_robot"``.

``req.backend_config`` is a ``RealRobotEnvConfig``. Like ``franka_real``, this backend
ignores ``req.num_envs``/``req.num_eval_envs`` (a real robot is a batch of exactly one).
Train and eval share one env instance per request: two envs would mean two connections
(and two device claims) on the same robot.

``req.observation`` is honored or rejected: ``rgb`` names the cameras (``rgb_<cam>`` keys),
``state`` toggles the flat ``state`` key, ``image_size`` resizes frames; depth, extra_state
and frame stacking raise ``ObservationContractError``.
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import Any, ClassVar

from rl_garden.envs.backend_registry import (
    EnvBackend,
    EnvRequest,
    register_env_backend,
)
from rl_garden.observations.schema import ObservationContractError


class RealRobotBackend(EnvBackend):
    api_version = 2
    config_field = "real_robot"

    # id(EnvRequest) -> env, dropped when the request is garbage-collected.
    _envs_by_request: ClassVar[dict[int, Any]] = {}

    @classmethod
    def resolve_config(cls, req: EnvRequest, *, is_eval: bool):
        del is_eval
        from rl_garden_real_world.envs.real_robot.config import RealRobotEnvConfig

        config = req.backend_config
        if not isinstance(config, RealRobotEnvConfig):
            raise ValueError(
                "real_robot backend needs req.backend_config to be a RealRobotEnvConfig "
                f"(with `robot` or `remote` set), got {type(config).__name__}"
            )
        observation = req.observation
        unsupported = {
            "depth": observation.depth,
            "extra_state": observation.extra_state,
            "frame_stack>1": observation.frame_stack > 1,
        }
        requested = [name for name, value in unsupported.items() if value]
        if requested:
            raise ObservationContractError(
                f"real_robot backend cannot produce {requested}; it supports state and rgb only"
            )
        cameras = list(observation.rgb)
        if config.cameras and config.cameras != cameras:
            raise ObservationContractError(
                f"real_robot config cameras {config.cameras} conflict with requested rgb "
                f"cameras {cameras}"
            )
        image_size = config.image_size
        if observation.image_size is not None:
            if image_size is not None and tuple(image_size) != tuple(observation.image_size):
                raise ObservationContractError(
                    f"real_robot config image_size {image_size} conflicts with requested "
                    f"image_size {observation.image_size}"
                )
            image_size = observation.image_size
        return dataclasses.replace(config, cameras=cameras, image_size=image_size)

    @classmethod
    def _make_env(cls, req: EnvRequest):
        from rl_garden_real_world.envs.real_robot import RealRobotEnv, RealRobotVectorEnv

        env = RealRobotVectorEnv(
            RealRobotEnv(cls.resolve_config(req, is_eval=False)),
            include_state=req.observation.state,
        )
        if req.reward_scale != 1.0 or req.reward_bias != 0.0:
            from rl_garden.envs.wrappers.reward_transform import RewardScaleBiasWrapper

            env = RewardScaleBiasWrapper(env, req.reward_scale, req.reward_bias)
        return env

    @classmethod
    def _shared_env(cls, req: EnvRequest):
        key = id(req)
        env = cls._envs_by_request.get(key)
        if env is None:
            env = cls._make_env(req)
            cls._envs_by_request[key] = env
            weakref.finalize(req, cls._envs_by_request.pop, key, None)
        return env

    @classmethod
    def make_train_env(cls, req: EnvRequest):
        return cls._shared_env(req)

    @classmethod
    def make_eval_env(cls, req: EnvRequest):
        return cls._shared_env(req)


register_env_backend("real_robot", RealRobotBackend)
