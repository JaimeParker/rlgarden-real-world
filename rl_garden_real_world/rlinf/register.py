"""Gymnasium registration of ``RealRobotEnv`` for RLinf's real-world env worker.

RLinf's ``RealWorldEnv._create_env`` calls
``gym.make(cfg.init_params.id, override_cfg=<dict>, worker_info=..., robot_info=...,
env_idx=..., env_cfg=...)``; with ``init_params.id`` set to
``"rl_garden_real_world.rlinf.register:RLGardenReal-v0"`` gymnasium imports this module (which
registers the id) and calls ``make_rlinf_env``. RLinf then batches the observation by
``observation_space`` (``{"state": {field: float32}, "frames": {cam: uint8 HxWx3}}``), takes
``frames[main_image_key]`` as the main image, and counts a step as successful iff its reward is
exactly 1.0. See ``configs/rlinf/realworld_rlgarden_franka.yaml``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rl_garden_real_world.envs.real_robot import (
    RealRobotEnv,
    RealRobotEnvConfig,
    RealRobotVectorEnv,
)

ENV_ID = "RLGardenReal-v0"
REWARD_MODES = ("none", "classifier")

logger = logging.getLogger(__name__)


def _first_scalar(value: Any) -> Any:
    """The only element of a python scalar or a batch-of-1 CPU tensor/array."""
    return np.asarray(value).reshape(-1)[0]


class RLinfEnvAdapter(gym.Wrapper):
    """Presents ``RealRobotEnv``'s unbatched nested-dict interface to RLinf.

    ``env`` is the stepping chain: the ``RealRobotEnv`` itself, or a batch-of-1 torch stack
    over it (``RealRobotVectorEnv`` + ``RewardClassifierWrapper``) when a reward classifier
    supplies the reward. Either way the observation returned is the ``RealRobotEnv``'s own
    nested numpy observation, and reward/termination flags are plain python scalars.
    """

    def __init__(
        self,
        env: gym.Env,
        real_env: RealRobotEnv,
        *,
        task_description: str,
        main_image_key: Optional[str],
    ) -> None:
        super().__init__(env)
        self.real_env = real_env
        self.task_description = task_description
        self.observation_space = real_env.observation_space
        self.action_space = real_env.action_space
        cameras = list(real_env.observation_space["frames"].spaces)
        if main_image_key not in cameras:
            raise ValueError(
                f"main_image_key {main_image_key!r} is not a camera of this env "
                f"(cameras: {cameras}); set override_cfg.main_image_key (and the RLinf env "
                "main_image_key) to one of real_robot.cameras"
            )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        _, info = self.env.reset(seed=seed, options=options)
        return self.real_env.last_observation, info

    def step(self, action):
        # RealRobotEnv tolerates stepping past ``terminated``; RLinf relies on that (it owns
        # truncation and auto-reset, and may ignore terminations).
        _, reward, terminated, truncated, info = self.env.step(np.asarray(action).reshape(-1))
        info = {
            key: value.detach().cpu().numpy() if hasattr(value, "detach") else value
            for key, value in info.items()
        }
        return (
            self.real_env.last_observation,
            float(_first_scalar(reward)),
            bool(_first_scalar(terminated)),
            bool(_first_scalar(truncated)),
            info,
        )

    def close(self) -> None:
        self.env.close()


def _wrap_reward_classifier(real_env: RealRobotEnv, classifier_cfg: dict) -> gym.Env:
    """The ``hil_serl._build_env`` classifier stack: the classifier sees the rl-garden backend
    vocabulary (``state`` + ``rgb_<cam>``), so its ``image_keys`` are ``rgb_<cam>`` names."""
    from rl_garden.common.utils import get_device
    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden.models.reward.success.model import load_classifier_fn

    classifier_cfg = dict(classifier_cfg)
    checkpoint = classifier_cfg.pop("checkpoint", None)
    if checkpoint is None:
        raise ValueError("reward: classifier needs override_cfg.classifier.checkpoint")
    vector_env = RealRobotVectorEnv(real_env)
    image_keys = tuple(classifier_cfg.pop("image_keys", None) or ()) or tuple(
        key for key in vector_env.single_observation_space.spaces if key != "state"
    )
    classifier_fn = load_classifier_fn(
        checkpoint,
        vector_env.single_observation_space,
        image_keys,
        device=get_device(classifier_cfg.pop("device", "auto")),
    )
    # Remaining keys (threshold, terminate_on_success) are RewardClassifierWrapper kwargs.
    return RewardClassifierWrapper(vector_env, classifier_fn, **classifier_cfg)


def make_rlinf_env(
    override_cfg: Optional[dict] = None,
    worker_info: Any = None,
    robot_info: Any = None,
    env_idx: int = 0,
    env_cfg: Any = None,
    **kwargs: Any,
) -> gym.Env:
    """Entry point RLinf's ``RealWorldEnv`` calls (via ``gym.make``) once per env worker.

    ``override_cfg`` is the plain-dict ``env.*.override_cfg``: ``real_robot`` holds the
    ``RealRobotEnvConfig`` fields; ``task_description``, ``main_image_key`` and ``reward``
    (``"none"`` -> 0.0, ``"classifier"`` -> reward-classifier success 1.0 with
    ``classifier: {checkpoint, image_keys, threshold, terminate_on_success, device}``).
    """
    del env_cfg, kwargs
    override_cfg = dict(override_cfg or {})
    if env_idx != 0:
        raise ValueError(
            f"env_idx={env_idx}: one real robot per worker; set env total_num_envs to the "
            "number of env workers (one RLGardenReal-v0 env each)"
        )
    logger.debug("RLGardenReal-v0 worker_info=%r robot_info=%r (ignored)", worker_info, robot_info)
    reward_mode = override_cfg.get("reward", "none")
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward {reward_mode!r} invalid; choose from {list(REWARD_MODES)}")
    if "real_robot" not in override_cfg:
        raise ValueError("override_cfg.real_robot (RealRobotEnvConfig fields) is required")

    real_env = RealRobotEnv(RealRobotEnvConfig(**override_cfg["real_robot"]))
    env: gym.Env = real_env
    if reward_mode == "classifier":
        env = _wrap_reward_classifier(real_env, override_cfg.get("classifier") or {})
    return RLinfEnvAdapter(
        env,
        real_env,
        task_description=override_cfg.get("task_description", ""),
        main_image_key=override_cfg.get("main_image_key"),
    )


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=f"{__name__}:make_rlinf_env")
