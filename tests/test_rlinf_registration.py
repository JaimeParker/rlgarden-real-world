"""RLinf contract for ``RLGardenReal-v0``: built exactly as RLinf's ``RealWorldEnv._create_env``
builds it (``gym.make(id, override_cfg=..., worker_info=..., robot_info=..., env_idx=...,
env_cfg=...)``), over robot_infra fake devices."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

import robot_infra.devices.backends  # noqa: F401  (registers the fake backends)
import robot_infra.devices.claims as claims

from rl_garden_real_world.envs.real_robot import RealRobotEnvConfig
from rl_garden_real_world.rlinf import register as rlinf_register

ENV_ID = "rl_garden_real_world.rlinf.register:RLGardenReal-v0"
FRAME_HW = (24, 32)
EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "configs/rlinf/realworld_rlgarden_franka.yaml"


@pytest.fixture(autouse=True)
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def _override_cfg(**overrides) -> dict:
    cfg = {
        "task_description": "pick up the cube",
        "main_image_key": "wrist",
        "reward": "none",
        "real_robot": {
            "robot": {
                "type": "franka",
                "arm_backend": "fake",
                "gripper": {"backend": "fake"},
                "cameras": [
                    {"name": "wrist", "backend": "fake", "width": FRAME_HW[1], "height": FRAME_HW[0]}
                ],
            },
            "cameras": ["wrist"],
            "home_joints": [0.0] * 7,
            "reset_settle_s": 0.0,
            "control_hz": 1000.0,
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def make_env():
    envs: list[gym.Env] = []

    def make(override_cfg=None, env_idx: int = 0) -> gym.Env:
        env = gym.make(
            ENV_ID,
            override_cfg=_override_cfg() if override_cfg is None else override_cfg,
            worker_info=None,
            robot_info=None,
            env_idx=env_idx,
            env_cfg=None,
        )
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def test_register_module_reimport_is_idempotent():
    module = importlib.import_module("rl_garden_real_world.rlinf.register")
    importlib.reload(module)
    importlib.import_module("rl_garden_real_world.rlinf.register")
    assert "RLGardenReal-v0" in gym.registry


def test_reset_and_step_match_rlinf_observation_contract(make_env):
    env = make_env()
    obs, info = env.reset()
    assert isinstance(info, dict)
    assert set(obs) == {"state", "frames"}
    assert isinstance(obs["state"], dict) and obs["state"]
    for value in obs["state"].values():
        assert value.dtype == np.float32 and value.ndim == 1
    assert obs["frames"]["wrist"].dtype == np.uint8
    assert obs["frames"]["wrist"].shape == (*FRAME_HW, 3)
    assert env.observation_space.contains(obs)
    assert env.action_space.shape == (7,)

    for _ in range(3):
        result = env.step(np.zeros((1, 7), dtype=np.float32))  # RLinf's per-env action row
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, float) and reward == 0.0
        assert terminated is False and truncated is False
        assert env.observation_space.contains(obs)

    assert env.get_wrapper_attr("task_description") == "pick up the cube"


def test_missing_main_image_key_lists_cameras(make_env):
    with pytest.raises(ValueError, match=r"main_image_key 'front'.*cameras: \['wrist'\]"):
        make_env(_override_cfg(main_image_key="front"))
    with pytest.raises(ValueError, match="main_image_key None"):
        cfg = _override_cfg()
        del cfg["main_image_key"]
        make_env(cfg)


def test_nonzero_env_idx_is_rejected(make_env):
    with pytest.raises(ValueError, match="one real robot per worker"):
        make_env(env_idx=1)


def test_unknown_reward_mode_is_rejected(make_env):
    with pytest.raises(ValueError, match="reward 'dense' invalid"):
        make_env(_override_cfg(reward="dense"))


def test_step_after_terminated_keeps_working(make_env, monkeypatch):
    env = make_env()
    env.reset()
    real_env = env.unwrapped
    original_step = real_env.step
    calls = {"n": 0}

    def terminate_once(action):
        obs, reward, terminated, truncated, info = original_step(action)
        calls["n"] += 1
        return obs, reward, calls["n"] == 1 or terminated, truncated, info

    monkeypatch.setattr(real_env, "step", terminate_once)
    action = np.zeros(7, dtype=np.float32)
    assert env.step(action)[2] is True
    obs, reward, terminated, truncated, _ = env.step(action)
    assert terminated is False and reward == 0.0
    assert env.observation_space.contains(obs)


def test_classifier_reward_is_exactly_one_on_success(make_env, monkeypatch):
    seen = {}

    def fake_load_classifier_fn(checkpoint, observation_space, image_keys, device="cpu"):
        seen.update(checkpoint=checkpoint, image_keys=tuple(image_keys), spaces=set(observation_space.spaces))
        steps = {"n": 0}

        def classifier_fn(obs):
            assert obs["rgb_wrist"].shape == (1, *FRAME_HW, 3)
            steps["n"] += 1
            return torch.tensor([0.9 if steps["n"] == 2 else 0.1])

        return classifier_fn

    monkeypatch.setattr(
        "rl_garden.models.reward.success.model.load_classifier_fn", fake_load_classifier_fn
    )
    env = make_env(
        _override_cfg(reward="classifier", classifier={"checkpoint": "/ckpt.pt", "device": "cpu"})
    )
    assert seen == {
        "checkpoint": "/ckpt.pt",
        "image_keys": ("rgb_wrist",),
        "spaces": {"rgb_wrist", "state"},
    }
    obs, _ = env.reset()
    assert set(obs) == {"state", "frames"}
    action = np.zeros(7, dtype=np.float32)
    rewards, terminations = [], []
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert env.observation_space.contains(obs)
        assert isinstance(info["classifier_success_prob"], np.ndarray)
        rewards.append(reward)
        terminations.append(terminated)
    assert rewards == [0.0, 1.0, 0.0]
    # terminate_on_success (the RewardClassifierWrapper default); stepping on afterwards works.
    assert terminations == [False, True, False]


def test_classifier_reward_requires_checkpoint(make_env):
    with pytest.raises(ValueError, match="classifier.checkpoint"):
        make_env(_override_cfg(reward="classifier"))


def test_example_yaml_override_cfg_builds_real_robot_config():
    env_block = yaml.safe_load(EXAMPLE_YAML.read_text())
    assert env_block["env_type"] == "real"
    assert env_block["init_params"]["id"] == ENV_ID
    override_cfg = env_block["override_cfg"]
    assert override_cfg["main_image_key"] == env_block["main_image_key"]
    assert override_cfg["reward"] in rlinf_register.REWARD_MODES
    config = RealRobotEnvConfig(**override_cfg["real_robot"])
    assert env_block["main_image_key"] in config.cameras


def test_rlinf_real_world_env_drives_registered_env(monkeypatch):
    pytest.importorskip("rlinf")
    from omegaconf import OmegaConf
    from rlinf.envs.real.env import RealWorldEnv

    # realworld_setup() kills roscore/rosmaster processes on the host; not in a test.
    monkeypatch.setattr(RealWorldEnv, "realworld_setup", staticmethod(lambda: None))
    cfg = OmegaConf.create(
        {
            "env_type": "real",
            "seed": 0,
            "group_size": 1,
            "auto_reset": True,
            "ignore_terminations": True,
            "use_fixed_reset_state_ids": False,
            "max_episode_steps": 100,
            "main_image_key": "wrist",
            "video_cfg": {"save_video": False},
            "init_params": {"id": ENV_ID, "num_envs": None},
            "override_cfg": copy.deepcopy(_override_cfg()),
        }
    )
    env = RealWorldEnv(cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    try:
        obs, _ = env.reset()
        assert tuple(obs["main_images"].shape) == (1, *FRAME_HW, 3)
        assert obs["task_descriptions"] == ["pick up the cube"]
        obs, reward, terminated, truncated, infos = env.step(np.zeros((1, 7), dtype=np.float32))
        assert tuple(reward.shape) == (1,) and float(reward[0]) == 0.0
        assert tuple(obs["main_images"].shape) == (1, *FRAME_HW, 3)
    finally:
        env.close()
