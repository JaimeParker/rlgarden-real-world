"""hil_serl action contract: env exposure, teleop validation, obs offsets, demo/snapshot checks, deploy."""
from __future__ import annotations

import dataclasses
import json
import pickle
import warnings

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

import robot_infra.devices.claims as claims

from rl_garden_real_world.action_contract import demo_contract_of, read_contract_sidecar
from rl_garden_real_world.envs.real_robot import RealRobotEnv, RealRobotEnvConfig
from rl_garden_real_world.envs.real_robot.env import RealRobotVectorEnv
from rl_garden_real_world.envs.wrappers._obs_chain import apply_obs_chain, tcp_pose_quat_slice
from rl_garden_real_world.envs.wrappers.grasp_penalty import GraspPenaltyWrapper
from rl_garden_real_world.envs.wrappers.rotvec_obs import RotvecObsWrapper
from rl_garden_real_world.envs.wrappers.teleop_intervention import TeleopInterventionWrapper
from rl_garden_real_world.training._demo_loading import load_demo_pkls


# -- fakes ----------------------------------------------------------------------------


class _ContractEnv(gym.Env):
    num_envs = 1

    def __init__(self, action_mode, arm_action_dim, gripper_in_action=True, num_joints=7):
        dim = arm_action_dim + int(gripper_in_action)
        self.single_action_space = spaces.Box(-1, 1, (dim,), dtype="float32")
        self.action_contract = {
            "action_mode": action_mode,
            "arm_action_dim": arm_action_dim,
            "gripper_in_action": gripper_in_action,
            "num_joints": num_joints,
        }
        self.step_actions = []

    def reset(self, **kwargs):
        return torch.zeros(1, 4), {}

    def step(self, action):
        self.step_actions.append(action)
        return torch.ones(1, 4), torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {}


class _Sample:
    def __init__(self, action, intervened=True):
        self.action = action
        self.intervened = intervened
        self.episode_end = False


class _Teleop:
    def __init__(self, samples=(), **declared):
        self._samples = list(samples)
        for key, value in declared.items():
            setattr(self, key, value)

    def reset(self):
        pass

    def poll(self):
        return self._samples.pop(0)


# -- env exposes its contract ----------------------------------------------------------


def _real_env(**overrides) -> RealRobotVectorEnv:
    config = RealRobotEnvConfig(robot={"type": "unused"}, **overrides)
    return RealRobotVectorEnv(RealRobotEnv(config, robot=object()))


@pytest.mark.parametrize(
    "overrides, arm_dim, total",
    [
        (dict(), 6, 7),
        (dict(action_mode="joint_delta"), 7, 8),
        (dict(action_mode="joint_delta", num_joints=6), 6, 7),
        (dict(gripper_in_action=False), 6, 6),
    ],
)
def test_real_robot_env_exposes_action_contract(overrides, arm_dim, total):
    env = _real_env(**overrides)
    assert env.arm_action_dim == arm_dim and env.single_action_space.shape == (total,)
    contract = env.get_wrapper_attr("action_contract")
    assert contract == env.env.action_contract
    assert contract["action_mode"] == env.action_mode == env.env.config.action_mode
    assert contract["gripper_in_action"] is env.gripper_in_action
    assert contract["arm_action_dim"] == arm_dim
    assert set(contract) >= {
        "num_joints", "joint_limits", "xyz_scale_m", "rot_scale_rad", "joint_scale_rad", "state_fields",
    }


# -- teleop validation ----------------------------------------------------------------


def test_teleop_accepts_cartesian_delta_with_gripper():
    env = _ContractEnv("cartesian_delta", 6)
    wrapped = TeleopInterventionWrapper(env, teleop=_Teleop([_Sample(np.full(7, 3.0))]))
    wrapped.step(torch.zeros(1, 7))
    assert env.step_actions[0].shape == (1, 7)


def test_teleop_without_contract_env_uses_legacy_cartesian_convention():
    class _Bare(gym.Env):
        num_envs = 1

    TeleopInterventionWrapper(_Bare(), teleop=_Teleop())


def test_teleop_rejects_franka_joint_delta_env_and_names_both_modes():
    env = _ContractEnv("joint_delta", 7)  # 8-D
    with pytest.raises(ValueError) as error:
        TeleopInterventionWrapper(env, teleop=_Teleop())
    message = str(error.value)
    assert "cartesian_delta" in message and "joint_delta" in message
    assert "7-D" in message and "8" in message
    assert "--real_robot.action_mode cartesian_delta" in message


def test_teleop_rejects_piper_joint_delta_even_though_widths_match():
    env = _ContractEnv("joint_delta", 6, num_joints=6)  # 7-D, same width as cartesian + gripper
    assert env.single_action_space.shape == (7,)
    with pytest.raises(ValueError, match="joint_delta"):
        TeleopInterventionWrapper(env, teleop=_Teleop())


def test_teleop_rejects_env_without_gripper_in_action():
    env = _ContractEnv("cartesian_delta", 6, gripper_in_action=False)
    with pytest.raises(ValueError, match="gripper_in_action=true.*non-hybrid"):
        TeleopInterventionWrapper(env, teleop=_Teleop())


def test_teleop_rejects_same_mode_width_mismatch():
    env = _ContractEnv("cartesian_delta", 6)
    with pytest.raises(ValueError, match="6-D actions"):
        TeleopInterventionWrapper(env, teleop=_Teleop(has_gripper=False))


def test_declared_joint_abs_teleop_is_accepted_and_intervene_action_has_env_shape():
    env = _ContractEnv("joint_abs", 6, num_joints=6)
    teleop = _Teleop(
        [_Sample(np.full(7, 0.5, dtype=np.float32))],
        action_mode="joint_abs", arm_action_dim=6, has_gripper=True,
    )
    wrapped = TeleopInterventionWrapper(env, teleop=teleop)
    _, _, _, _, info = wrapped.step(torch.zeros(1, 7))
    assert info["intervene_action"].shape == (1, 7)
    assert env.step_actions[0].shape == (1, 7)


def test_teleop_over_real_robot_vector_env_checks_action_mode():
    TeleopInterventionWrapper(_real_env(), teleop=_Teleop())
    with pytest.raises(ValueError, match="joint_delta"):
        TeleopInterventionWrapper(_real_env(action_mode="joint_delta", num_joints=6), teleop=_Teleop())
    with pytest.raises(ValueError, match="gripper_in_action"):
        TeleopInterventionWrapper(_real_env(gripper_in_action=False), teleop=_Teleop())


# -- observation offsets ---------------------------------------------------------------


def test_tcp_pose_quat_slice_follows_state_field_offsets():
    config = RealRobotEnvConfig(
        remote="h:1", num_joints=6, state_fields=("joint_pos", "tcp_pose", "gripper_pos")
    )
    assert tcp_pose_quat_slice(config) == (9, 13)
    assert tcp_pose_quat_slice(dataclasses.replace(config, state_fields=("tcp_pose", "joint_pos"))) == (3, 7)
    assert tcp_pose_quat_slice(dataclasses.replace(config, state_fields=("joint_pos",))) is None


class _StateEnv(gym.Env):
    num_envs = 1

    def __init__(self, state_dim):
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (state_dim,), dtype="float32")}
        )
        self.observation_space = batch_space(self.single_observation_space, 1)


def test_rotvec_obs_uses_computed_offset_and_drops_one_dim():
    config = RealRobotEnvConfig(
        remote="h:1", num_joints=6, state_fields=("joint_pos", "tcp_pose", "gripper_pos")
    )
    env = apply_obs_chain(
        _StateEnv(14), frame_stack=1, image_keys=(), convert_obs_rotation=True,
        quat_slice=tcp_pose_quat_slice(config),
    )
    assert isinstance(env, RotvecObsWrapper)
    assert env.single_observation_space["state"].shape == (13,)
    state = torch.zeros(1, 14)
    state[:, 9:13] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # identity quaternion at joint_pos(6)+xyz(3)
    state[:, 6:9] = 5.0
    converted = env.observation({"state": state})["state"]
    assert converted.shape == (1, 13)
    assert torch.all(converted[:, 6:9] == 5.0) and torch.allclose(converted[:, 9:12], torch.zeros(1, 3))


def test_obs_chain_warns_and_skips_without_tcp_pose(caplog):
    with caplog.at_level("WARNING"):
        env = apply_obs_chain(
            _StateEnv(6), frame_stack=1, image_keys=(), convert_obs_rotation=True, quat_slice=None
        )
    assert not isinstance(env, RotvecObsWrapper)
    assert "convert_obs_rotation skipped" in caplog.text


# -- grasp penalty / algorithm smoke ---------------------------------------------------


@pytest.mark.parametrize("width", [7, 8])
def test_grasp_penalty_reads_last_dim_for_joint_action_widths(width):
    class _Env(gym.Env):
        num_envs = 1

        def reset(self, **kwargs):
            return torch.zeros(1, 4), {}

        def step(self, action):
            return torch.ones(1, 4), torch.tensor([1.0]), torch.tensor([False]), torch.tensor([False]), {}

    wrapped = GraspPenaltyWrapper(_Env(), penalty=-0.05)
    wrapped.reset()
    open_action, closed_action = torch.zeros(1, width), torch.zeros(1, width)
    open_action[..., -1], closed_action[..., -1] = -1.0, 1.0
    open_action[..., 0] = 1.0  # a non-gripper dim flipping must not matter
    wrapped.step(open_action)
    _, _, _, _, info_same = wrapped.step(open_action)
    _, _, _, _, info_flip = wrapped.step(closed_action)
    assert info_same["grasp_penalty"].item() == 0.0
    assert info_flip["grasp_penalty"].item() == pytest.approx(-0.05)


class _SmokeVecEnv:
    num_envs = 1

    def __init__(self, action_dim):
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)}
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, 1)
        self.action_space = batch_space(self.single_action_space, 1)

    def reset(self, seed=None):
        return {"state": torch.zeros(1, 4)}, {}

    def step(self, actions):
        assert actions.shape == (1, self.single_action_space.shape[0])
        return (
            {"state": torch.randn(1, 4)}, torch.ones(1),
            torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {},
        )

    def close(self):
        return None


@pytest.mark.parametrize("action_dim", [7, 8], ids=["piper_6+1", "franka_7+1"])
def test_build_rlpd_hybrid_trains_one_update_on_joint_action_widths(action_dim):
    from rl_garden.training.online.rlpd_hybrid import RLPDHybridArgs, build_rlpd_hybrid

    args = RLPDHybridArgs(
        num_envs=1, buffer_size=64, buffer_device="cpu", batch_size=8, learning_starts=1,
        hidden_dim=8, discrete_hidden_dim=8,
    )
    agent = build_rlpd_hybrid(args, _SmokeVecEnv(action_dim), None, logger=None, checkpoint_dir=None)
    agent.learn(total_timesteps=16)
    assert agent._global_step >= 16
    action = agent.policy.predict({"state": torch.zeros(1, 4, device=agent.device)}, deterministic=True)
    assert action.shape == (1, action_dim)


# -- demo / snapshot contract ----------------------------------------------------------

_CART = {"action_mode": "cartesian_delta", "arm_action_dim": 6, "gripper_in_action": True, "num_joints": 7}
_PIPER_JOINT = {"action_mode": "joint_delta", "arm_action_dim": 6, "gripper_in_action": True, "num_joints": 6}


def _write_pkl(path, action_dim, count=2, meta=None):
    transitions = [
        {
            "obs": {"state": torch.zeros(1, 4)}, "next_obs": {"state": torch.zeros(1, 4)},
            "action": torch.zeros(1, action_dim), "reward": torch.zeros(1), "done": torch.zeros(1),
        }
        for _ in range(count)
    ]
    with open(path, "wb") as f:
        pickle.dump(transitions, f)
    if meta is not None:
        path.with_suffix(".contract.json").write_text(json.dumps(meta), encoding="utf-8")


def test_demo_contract_of_env_subset_or_none():
    assert demo_contract_of(_ContractEnv("cartesian_delta", 6)) == _CART
    assert demo_contract_of(gym.Env()) is None


def test_load_demo_pkls_rejects_wrong_action_dim(tmp_path):
    _write_pkl(tmp_path / "a.pkl", action_dim=8)
    with pytest.raises(ValueError, match="action dim 8.*expects 7"):
        load_demo_pkls([str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=_CART)


def test_load_demo_pkls_rejects_same_width_different_action_mode(tmp_path):
    _write_pkl(tmp_path / "a.pkl", action_dim=7, meta=_PIPER_JOINT)
    with pytest.raises(ValueError) as error:
        load_demo_pkls([str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=_CART)
    assert "joint_delta" in str(error.value) and "cartesian_delta" in str(error.value)


def test_load_demo_pkls_accepts_matching_meta_without_warning(tmp_path):
    _write_pkl(tmp_path / "a.pkl", action_dim=7, meta=_CART)
    added = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        count = load_demo_pkls(
            [str(tmp_path / "*.pkl")], lambda *a, **k: added.append(a), "cpu", expected_contract=_CART
        )
    assert count == 2 == len(added)


def test_load_demo_pkls_warns_for_legacy_pkl_without_meta_but_loads(tmp_path):
    _write_pkl(tmp_path / "a.pkl", action_dim=7)
    with pytest.warns(UserWarning, match="no action-contract metadata"):
        count = load_demo_pkls(
            [str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=_CART
        )
    assert count == 2


def test_load_demo_pkls_without_expected_contract_is_unchanged(tmp_path):
    _write_pkl(tmp_path / "a.pkl", action_dim=99)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_demo_pkls([str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu") == 2


def _learner_agent(action_dim=3):
    from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid

    class _Env(_SmokeVecEnv):
        pass

    agent = RLPDHybrid(
        env=_Env(action_dim), device="cpu", buffer_device="cpu", buffer_size=32, batch_size=4,
        learning_starts=1, training_freq=4, eval_freq=0, log_freq=0, net_arch=[8], discrete_hidden_dim=8,
    )
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    return agent


_SMALL_CART = {"action_mode": "cartesian_delta", "arm_action_dim": 2, "gripper_in_action": True, "num_joints": 7}
_SMALL_JOINT = {"action_mode": "joint_delta", "arm_action_dim": 2, "gripper_in_action": True, "num_joints": 2}


def _learner_transition(intervened):
    return {
        "obs": {"state": torch.zeros(1, 4)}, "next_obs": {"state": torch.zeros(1, 4)},
        "action": torch.zeros(1, 3), "reward": torch.ones(1), "done": torch.zeros(1, dtype=torch.bool),
        "intervened": intervened,
    }


def test_learner_snapshots_carry_contract_and_reload_validates_it(tmp_path):
    from rl_garden_real_world.hil_serl import HilSerlLearnerLoop

    loop = HilSerlLearnerLoop(
        _learner_agent(), "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2,
        action_contract=_SMALL_CART,
    )
    loop._on_transition(_learner_transition(False))
    loop._on_transition(_learner_transition(True))
    (online_pkl,) = (tmp_path / "buffer").glob("*.pkl")
    assert read_contract_sidecar(online_pkl) == _SMALL_CART

    reloaded = HilSerlLearnerLoop(
        _learner_agent(), "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2,
        action_contract=_SMALL_CART,
    )
    assert reloaded.received_transitions == 2

    with pytest.raises(ValueError, match="joint_delta"):
        HilSerlLearnerLoop(
            _learner_agent(), "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2,
            action_contract=_SMALL_JOINT,
        )
    wrong_dim = dict(_SMALL_CART, arm_action_dim=6)
    with pytest.raises(ValueError, match="arm_action_dim"):
        HilSerlLearnerLoop(
            _learner_agent(), "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=2,
            action_contract=wrong_dim,
        )


def test_learner_demo_dataset_paths_are_validated(tmp_path):
    from rl_garden_real_world.hil_serl import HilSerlLearnerLoop

    _write_pkl(tmp_path / "demo.pkl", action_dim=8)
    with pytest.raises(ValueError, match="action dim 8"):
        HilSerlLearnerLoop(
            _learner_agent(), "127.0.0.1", 0, checkpoint_dir=str(tmp_path / "ckpt"),
            demo_dataset_paths=[str(tmp_path / "demo.pkl")], action_contract=_SMALL_CART,
        )


# -- deploy contract -------------------------------------------------------------------

FAKE_FRANKA = {"type": "franka", "arm_backend": "fake", "gripper": {"backend": "fake"}}


@pytest.fixture
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def _training(real_robot=None, **extra):
    import types

    return types.SimpleNamespace(real_robot=real_robot, **extra)


def _deploy_args(**real_robot_overrides):
    from rl_garden_real_world.deploy import DeployArgs

    return DeployArgs(
        checkpoint="unused.pt",
        real_robot=RealRobotEnvConfig(robot=FAKE_FRANKA, **real_robot_overrides),
    )


def test_deploy_cli_contract_conflict_with_training_raises_listing_both_sides():
    from rl_garden_real_world.deploy import resolve_deploy_robot_config

    trained = RealRobotEnvConfig(remote="h:1", action_mode="joint_delta", num_joints=7)
    args = _deploy_args(action_mode="joint_abs", joint_limits=[[-1.0, 1.0]] * 7)
    with pytest.raises(ValueError) as error:
        resolve_deploy_robot_config(args, _training(trained))
    message = str(error.value)
    assert "action_mode" in message and "joint_delta" in message and "joint_abs" in message


def test_deploy_default_cli_adopts_training_contract_and_keeps_connection_fields():
    from rl_garden_real_world.deploy import resolve_deploy_robot_config

    trained = RealRobotEnvConfig(
        remote="train-host:1", token="secret", action_mode="joint_delta", num_joints=6,
        gripper_in_action=True, state_fields=("joint_pos", "tcp_pose"), control_hz=5.0,
        xyz_scale_m=0.01,
    )
    args = _deploy_args(home_joints=[0.0] * 6, num_joints=6)
    resolved = resolve_deploy_robot_config(args, _training(trained))
    assert resolved.action_mode == "joint_delta" and resolved.num_joints == 6
    assert resolved.state_fields == ("joint_pos", "tcp_pose")
    assert resolved.control_hz == 5.0 and resolved.xyz_scale_m == 0.01
    assert resolved.robot == FAKE_FRANKA and resolved.remote is None and resolved.token is None
    assert resolved.home_joints == [0.0] * 6


def test_deploy_cli_agreeing_with_training_is_accepted():
    from rl_garden_real_world.deploy import resolve_deploy_robot_config

    trained = RealRobotEnvConfig(remote="h:1", action_mode="joint_delta")
    resolved = resolve_deploy_robot_config(_deploy_args(action_mode="joint_delta"), _training(trained))
    assert resolved.action_mode == "joint_delta"


def test_deploy_without_training_real_robot_falls_back_to_cli_with_warning(caplog):
    from rl_garden_real_world.deploy import resolve_deploy_robot_config

    args = _deploy_args(action_mode="joint_delta")
    with caplog.at_level("WARNING"):
        assert resolve_deploy_robot_config(args, _training(None)) is args.real_robot
    assert "no real_robot section" in caplog.text


def test_restore_training_args_reads_real_robot_and_rotvec_from_persisted_config(tmp_path):
    """persist_resolved_config -> config.json -> restore_training_args round trip."""
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden_real_world.deploy import DeployArgs, restore_training_args
    from rl_garden_real_world.training.hil_serl import HilSerlArgs

    trained = RealRobotEnvConfig(
        remote="h:1", token="t", action_mode="joint_delta", num_joints=6, image_size=(16, 16),
        state_fields=("joint_pos", "tcp_pose"), joint_limits=[[-1.0, 1.0]] * 6,
    )
    persist_resolved_config(
        HilSerlArgs(env_backend="real_robot", real_robot=trained, convert_obs_rotation=True),
        training_phase="real_world", algorithm="hil_serl", run_name="run", log_dir=str(tmp_path),
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_0.pt"
    checkpoint.parent.mkdir()
    torch.save({"metadata": {}}, checkpoint)

    restored = restore_training_args(DeployArgs(checkpoint=str(checkpoint), algo="rlpd_hybrid"))
    assert restored.real_robot == trained
    assert restored.convert_obs_rotation is True
    assert restored.obs.frame_stack == 1


def test_deploy_env_replays_rotvec_so_observation_matches_training_agent(tmp_path, _claim_dir):
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden.training.online.rlpd import RLPDArgs, build_rlpd
    from rl_garden_real_world.deploy import (
        DeployArgs, build_deploy_env, restore_policy, restore_training_args,
    )
    from rl_garden_real_world.training.hil_serl import HilSerlArgs

    trained = RealRobotEnvConfig(robot=FAKE_FRANKA, home_joints=[0.0] * 7, reset_settle_s=0.0)
    persist_resolved_config(
        HilSerlArgs(env_backend="real_robot", real_robot=trained, convert_obs_rotation=True),
        training_phase="real_world", algorithm="hil_serl", run_name="run", log_dir=str(tmp_path),
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_0.pt"
    checkpoint.parent.mkdir()
    # Deploy CLI knows only the connection (default contract fields).
    args = DeployArgs(checkpoint=str(checkpoint), real_robot=trained)

    training_args = RLPDArgs(num_envs=1, num_eval_envs=1, buffer_size=8, buffer_device="cpu")
    training_args.real_robot = trained
    training_args.convert_obs_rotation = True
    train_env = build_deploy_env(args, training_args)
    assert train_env.single_observation_space["state"].shape == (33,)  # 34 raw - 1 (quat -> rotvec)
    saved = build_rlpd(training_args, train_env, None, logger=None, checkpoint_dir=None)
    saved.save(checkpoint)
    train_env.close()

    restored = restore_training_args(args)
    env = build_deploy_env(args, restored)
    try:
        assert env.single_observation_space["state"].shape == (33,)
        restore_policy(args, restored, env)  # no shape mismatch in agent.load
    finally:
        env.close()


# -- sidecar carries limits/scales/state_fields (M1) ------------------------------------


def _joint_abs_contract(**overrides):
    config = dict(action_mode="joint_abs", joint_limits=[[-1.0, 1.0]] * 7)
    config.update(overrides)
    return demo_contract_of(_real_env(**config))


def test_sidecar_from_real_env_contract_round_trips_without_warning(tmp_path):
    from rl_garden_real_world.action_contract import write_contract_sidecar

    contract = _joint_abs_contract()
    assert set(contract) >= {"joint_limits", "xyz_scale_m", "rot_scale_rad", "joint_scale_rad", "state_fields"}
    _write_pkl(tmp_path / "a.pkl", action_dim=8)
    write_contract_sidecar(tmp_path / "a.pkl", contract)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        count = load_demo_pkls(
            [str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=contract
        )
    assert count == 2


@pytest.mark.parametrize(
    "field, recorded, replayed",
    [
        ("joint_limits", dict(joint_limits=[[-1.0, 1.0]] * 7), dict(joint_limits=[[-2.0, 2.0]] * 7)),
        ("joint_scale_rad", dict(joint_scale_rad=0.05), dict(joint_scale_rad=0.1)),
        ("xyz_scale_m", dict(xyz_scale_m=0.02), dict(xyz_scale_m=0.05)),
        (
            "state_fields",
            dict(state_fields=("joint_pos", "gripper_pos")),
            dict(state_fields=("gripper_pos", "joint_pos")),
        ),
    ],
)
def test_sidecar_physical_field_mismatch_raises_naming_both_values(tmp_path, field, recorded, replayed):
    from rl_garden_real_world.action_contract import write_contract_sidecar

    _write_pkl(tmp_path / "a.pkl", action_dim=8)
    write_contract_sidecar(tmp_path / "a.pkl", _joint_abs_contract(**recorded))
    with pytest.raises(ValueError) as error:
        load_demo_pkls(
            [str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu",
            expected_contract=_joint_abs_contract(**replayed),
        )
    message = str(error.value)
    assert f"{field}: recorded=" in message and "env=" in message


def test_sidecar_float_scales_compare_with_tolerance(tmp_path):
    from rl_garden_real_world.action_contract import write_contract_sidecar

    _write_pkl(tmp_path / "a.pkl", action_dim=8)
    write_contract_sidecar(tmp_path / "a.pkl", _joint_abs_contract(joint_scale_rad=0.05))
    load_demo_pkls(
        [str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu",
        expected_contract=_joint_abs_contract(joint_scale_rad=0.05 + 1e-12),
    )


def test_old_sidecar_without_physical_fields_is_checked_on_its_keys_and_warns(tmp_path):
    cart = demo_contract_of(_real_env())
    _write_pkl(tmp_path / "a.pkl", action_dim=7, meta=_CART)
    with pytest.warns(UserWarning, match="predates"):
        count = load_demo_pkls(
            [str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=cart
        )
    assert count == 2
    _write_pkl(tmp_path / "a.pkl", action_dim=7, meta=_PIPER_JOINT)
    with pytest.raises(ValueError, match="joint_delta"):
        load_demo_pkls([str(tmp_path / "*.pkl")], lambda *a, **k: None, "cpu", expected_contract=cart)


# -- deploy restores classifier_image_keys (M2) -----------------------------------------


class _TwoCameraEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        image = spaces.Box(0, 255, (8, 8, 3), dtype=np.uint8)
        self.single_observation_space = spaces.Dict(
            {"state": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32), "wrist": image, "front": image}
        )
        self.single_action_space = spaces.Box(-1, 1, (7,), dtype=np.float32)


def test_deploy_restores_classifier_image_keys_so_frame_stacking_matches_training(tmp_path):
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden_real_world.deploy import DeployArgs, restore_training_args
    from rl_garden_real_world.training.hil_serl import HilSerlArgs

    trained_args = HilSerlArgs(
        env_backend="real_robot",
        real_robot=RealRobotEnvConfig(remote="h:1", image_size=(8, 8)),
        classifier_image_keys=("wrist",),
    )
    persist_resolved_config(
        trained_args, training_phase="real_world", algorithm="hil_serl", run_name="run", log_dir=str(tmp_path)
    )
    checkpoint = tmp_path / "run" / "checkpoints" / "checkpoint_0.pt"
    checkpoint.parent.mkdir()
    torch.save({"metadata": {}}, checkpoint)

    restored = restore_training_args(DeployArgs(checkpoint=str(checkpoint), algo="rlpd_hybrid"))
    assert tuple(restored.classifier_image_keys) == ("wrist",)

    def stacked_space(image_keys):
        env = apply_obs_chain(
            _TwoCameraEnv(), frame_stack=3, image_keys=image_keys, convert_obs_rotation=False, quat_slice=None
        )
        return {key: space.shape for key, space in env.single_observation_space.spaces.items()}

    deployed = stacked_space(restored.classifier_image_keys)
    assert deployed == stacked_space(trained_args.classifier_image_keys)
    assert deployed["wrist"] == (3, 8, 8, 3) and deployed["front"] == (8, 8, 3)


# -- deploy env requires the joint_abs step limit (H1) ----------------------------------


@pytest.mark.parametrize("device_step_limit", [False, True])
def test_deploy_joint_abs_requires_device_joint_step_limit(_claim_dir, device_step_limit):
    from robot_infra.devices.errors import DeviceFault
    from rl_garden.training.online.rlpd import RLPDArgs
    from rl_garden_real_world.deploy import DeployArgs, build_deploy_env

    robot = dict(FAKE_FRANKA)
    if device_step_limit:
        robot["max_delta_per_step"] = {"joint_rad": 0.5}
    connection = dict(robot=robot, home_joints=[0.0] * 7, reset_settle_s=0.0)
    trained = RealRobotEnvConfig(**connection, action_mode="joint_abs", joint_limits=[[-1.0, 1.0]] * 7)
    training_args = RLPDArgs(num_envs=1, num_eval_envs=1, buffer_size=8, buffer_device="cpu")
    training_args.real_robot = trained
    args = DeployArgs(checkpoint="unused.pt", real_robot=RealRobotEnvConfig(**connection))

    env = build_deploy_env(args, training_args)
    try:
        if device_step_limit:
            env.reset()
        else:
            with pytest.raises(DeviceFault, match="max_delta_per_step"):
                env.reset()
    finally:
        env.close()


def test_deploy_main_reports_missing_joint_step_limit_as_device_error(_claim_dir, capsys):
    from rl_garden.training.online.rlpd import RLPDArgs
    from rl_garden_real_world.deploy import DEVICE_ERROR_EXIT_CODE, DeployArgs, build_deploy_env, main

    connection = dict(robot=FAKE_FRANKA, home_joints=[0.0] * 7, reset_settle_s=0.0)
    training_args = RLPDArgs(num_envs=1, num_eval_envs=1, buffer_size=8, buffer_device="cpu")
    training_args.real_robot = RealRobotEnvConfig(
        **connection, action_mode="joint_abs", joint_limits=[[-1.0, 1.0]] * 7
    )
    args = DeployArgs(checkpoint="unused.pt", real_robot=RealRobotEnvConfig(**connection), episodes=1)
    env = build_deploy_env(args, training_args)

    class _UnusedPolicy:
        def eval(self):
            pass

    assert main(args, env=env, policy=_UnusedPolicy()) == DEVICE_ERROR_EXIT_CODE
    assert "max_delta_per_step" in capsys.readouterr().err
