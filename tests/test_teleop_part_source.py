"""TeleopPartSource + TeleopInterventionWrapper(device="robot") over robot_infra fake devices:
a fake franka with a scripted fake teleop part, both local and through RemoteRobot + LocalClient."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import robot_infra.devices.backends  # noqa: F401  (registers the fake backends)
import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceFault, GripperConfig
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.devices.teleop import TeleopConfig
from robot_infra.server import RemoteRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient

from rl_garden_real_world.envs.real_robot import RealRobotEnv, RealRobotEnvConfig, RealRobotVectorEnv
from rl_garden_real_world.envs.wrappers.teleop_intervention import TeleopInterventionWrapper
from rl_garden_real_world.envs.wrappers.teleop_sources import TeleopPartSource

ZERO_HOME = [0.0] * 7
FRAME_HW = (24, 32)
JOINT_LIMITS = [[-2.0, 2.0]] * 7
LEADER_Q = [0.5, -1.0, 0.0, 1.0, -2.0, 2.0, 0.25]
FAST = dict(control_hz=1000.0)


@pytest.fixture(autouse=True)
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


def _teleop_config(**options) -> TeleopConfig:
    return TeleopConfig(name="leader", backend="fake", options=options)


def _fake_franka(teleop: TeleopConfig, gripper_mode: str = "binary") -> FrankaRobot:
    return FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake", mode=gripper_mode),
            cameras=[CameraConfig(name="wrist", backend="fake", width=32, height=24, fps=60)],
            teleop=teleop,
            # joint_abs requires it; loose enough never to move a target in these tests.
            max_delta_per_step={"joint_rad": 10.0},
        )
    )


class _Rig:
    def __init__(self, mode: str, teleop: TeleopConfig, gripper_mode: str = "binary") -> None:
        self.mode = mode
        self.service = None
        if mode == "local":
            self.robot = _fake_franka(teleop, gripper_mode)
            self.device_robot = self.robot
        else:
            self.service = RobotService(_fake_franka(teleop, gripper_mode), watchdog_s=30.0)
            self.robot = RemoteRobot.from_client(LocalClient(self.service))
            self.device_robot = self.service.robot
        options = teleop.options
        mode_ = options.get("action_space", "cartesian_delta")
        self.teleop_contract = (
            mode_,
            6 if mode_ == "cartesian_delta" else options["num_joints"],
            options.get("has_gripper", True),
        )
        self.envs: list[RealRobotEnv] = []

    def env(self, **overrides) -> RealRobotEnv:
        values = dict(cameras=[], home_joints=ZERO_HOME, reset_settle_s=0.0, **FAST)
        if self.mode == "local":
            values["robot"] = {"type": "franka", "arm_backend": "fake"}
        else:
            values["remote"] = "local"
        values.update(overrides)
        env = RealRobotEnv(RealRobotEnvConfig(**values), robot=self.robot)
        self.envs.append(env)
        return env

    def wrapped(self, *, teleop_contract=None, **overrides) -> tuple[TeleopInterventionWrapper, RealRobotEnv]:
        real_env = self.env(**overrides)
        if teleop_contract is None and self.mode == "remote":
            teleop_contract = self.teleop_contract
        kwargs = {"joint_limits": real_env.config.joint_limits, "joint_scale_rad": real_env.config.joint_scale_rad}
        if teleop_contract is not None:
            kwargs["teleop_contract"] = teleop_contract
        wrapper = TeleopInterventionWrapper(RealRobotVectorEnv(real_env), device="robot", **kwargs)
        return wrapper, real_env

    def close(self) -> None:
        for env in self.envs:
            env.close()
        if self.robot.is_connected:
            self.robot.disconnect()
        if self.service is not None:
            self.service.close()


@pytest.fixture(params=["local", "remote"])
def make_rig(request):
    rigs: list[_Rig] = []

    def make(gripper_mode: str = "binary", **options) -> _Rig:
        rig = _Rig(request.param, _teleop_config(**options), gripper_mode)
        rigs.append(rig)
        return rig

    yield make
    for rig in rigs:
        rig.close()


def _zeros(env) -> torch.Tensor:
    return torch.zeros(1, env.single_action_space.shape[0])


# -- cartesian: pass-through, replacement, events, gripper ---------------------------

CARTESIAN_SCRIPT = [
    {"action": [0.0] * 6, "active": False},
    {"action": [0.5, -0.5, 0.0, 0.0, 0.1, 0.0], "gripper": 0.0, "active": True},
    {"action": [0.0] * 6, "gripper": 0.25, "active": True, "events": {"episode_end": True}, "hold": 2},
]


def test_cartesian_passthrough_replacement_episode_end_and_gripper(make_rig):
    rig = make_rig(script=CARTESIAN_SCRIPT, on_end="idle")
    env, _ = rig.wrapped()
    env.reset()
    policy = torch.full((1, 7), 0.1)

    _, _, _, _, info = env.step(policy)
    assert "intervene_action" not in info and info["human_episode_end"] is False

    _, _, _, _, info = env.step(policy)
    assert info["intervene_action"].shape == (1, 7)
    np.testing.assert_allclose(
        info["intervene_action"].numpy()[0], [0.5, -0.5, 0.0, 0.0, 0.1, 0.0, -1.0], atol=1e-6
    )
    assert info["human_episode_end"] is False

    _, _, _, truncated, info = env.step(policy)
    assert info["intervene_action"][0, -1].item() == pytest.approx(-0.5)  # 2 * 0.25 - 1
    assert info["human_episode_end"] is True
    assert bool(truncated) is True  # the operator ended the episode
    _, _, _, truncated, info = env.step(policy)  # same scripted step held: the event fires only once
    assert "intervene_action" in info and info["human_episode_end"] is False
    assert bool(truncated) is False


def test_reset_tolerates_unconnected_robot_then_resets_teleop(make_rig, monkeypatch):
    rig = make_rig(script=CARTESIAN_SCRIPT)
    env, _ = rig.wrapped()
    assert not rig.robot.is_connected
    env.reset()  # the env connects the robot, then the teleop part is reset
    assert rig.robot.is_connected

    calls = []
    teleop = rig.robot.teleop
    original = teleop.reset
    monkeypatch.setattr(teleop, "reset", lambda: (calls.append(1), original())[1])
    env.reset()
    assert calls == [1]


def test_robot_observation_carries_teleop_but_env_observation_does_not(make_rig):
    rig = make_rig(script=CARTESIAN_SCRIPT)
    env, real_env = rig.wrapped()
    obs, _ = env.reset()
    assert "teleop" in rig.robot.get_observation()
    assert set(obs) == {"state"}
    assert set(real_env.last_observation) == {"state", "frames"}


# -- action-space mismatches ----------------------------------------------------------


def test_joint_abs_teleop_on_cartesian_env_rejected_naming_both(make_rig):
    rig = make_rig(action_space="joint_abs", num_joints=7)
    with pytest.raises(ValueError, match="joint_abs.*cartesian_delta"):
        rig.wrapped()


def test_piper_joint_delta_env_rejects_cartesian_teleop_of_same_width(make_rig):
    rig = make_rig()  # cartesian twist + gripper: 7-D, the same width as 6 joints + gripper
    with pytest.raises(ValueError, match="cartesian_delta.*joint_delta"):
        rig.wrapped(action_mode="joint_delta", num_joints=6, home_joints=[0.0] * 6)


def test_remote_robot_without_declared_teleop_is_rejected():
    service = RobotService(_fake_franka(_teleop_config()), watchdog_s=30.0)
    robot = RemoteRobot.from_client(LocalClient(service))
    try:
        env = RealRobotEnv(
            RealRobotEnvConfig(remote="local", cameras=[], home_joints=ZERO_HOME), robot=robot
        )
        with pytest.raises(ValueError, match="teleop_contract"):
            TeleopInterventionWrapper(RealRobotVectorEnv(env), device="robot")
    finally:
        service.close()


def test_declared_teleop_contradicting_connected_part_faults_before_any_step():
    service = RobotService(_fake_franka(_teleop_config(has_gripper=False)), watchdog_s=30.0)
    robot = RemoteRobot.from_client(LocalClient(service))
    env = RealRobotEnv(
        RealRobotEnvConfig(remote="local", cameras=[], home_joints=ZERO_HOME, **FAST), robot=robot
    )
    try:
        wrapper = TeleopInterventionWrapper(
            RealRobotVectorEnv(env), device="robot", teleop_contract=("cartesian_delta", 6, True)
        )
        # The teleop part is reset right after the env connects the robot, so the mismatch
        # surfaces in reset().
        with pytest.raises(DeviceFault, match="declares"):
            wrapper.reset()
        assert env._elapsed_steps == 0
    finally:
        env.close()
        service.close()


# -- Step B: joint teleop sources ---------------------------------------------------


def test_joint_abs_intervention_is_leader_normalized_onto_limits(make_rig):
    rig = make_rig(
        action_space="joint_abs",
        num_joints=7,
        script=[{"action": LEADER_Q, "gripper": 1.0, "active": True}],
    )
    env, _ = rig.wrapped(action_mode="joint_abs", joint_limits=JOINT_LIMITS)
    env.reset()
    _, _, _, _, info = env.step(_zeros(env))
    low, high = np.array(JOINT_LIMITS).T
    expected = 2.0 * (np.array(LEADER_Q) - low) / (high - low) - 1.0
    np.testing.assert_allclose(info["intervene_action"].numpy()[0, :7], expected, atol=1e-6)
    assert info["intervene_action"][0, 7].item() == pytest.approx(1.0)


def test_joint_delta_intervention_is_leader_minus_follower_over_scale_clipped(make_rig):
    leader = [0.02, -0.03, 0.0, 0.5, -0.5, 0.01, 0.0]
    rig = make_rig(
        action_space="joint_delta",
        num_joints=7,
        script=[
            {"action": [0.0] * 7, "active": False},
            {"action": leader, "active": True},
        ],
    )
    env, real_env = rig.wrapped(action_mode="joint_delta")
    env.reset()
    policy = torch.zeros(1, 8)
    policy[0, 0] = 0.4  # move the follower off home so q_follower != 0
    env.step(policy)
    follower = real_env.last_observation["state"]["joint_pos"].astype(np.float64)
    assert follower[0] != 0.0

    _, _, _, _, info = env.step(torch.zeros(1, 8))
    scale = real_env.config.joint_scale_rad
    expected = np.clip((np.array(leader) - follower) / scale, -1.0, 1.0)
    np.testing.assert_allclose(info["intervene_action"].numpy()[0, :7], expected, atol=1e-5)


def _device_joint_target(rig) -> np.ndarray:
    return np.asarray(rig.device_robot.arm._joint_target, dtype=np.float64)


def test_joint_abs_leader_joints_reach_the_follower_arm_as_servo_targets(make_rig):
    leader_2 = [0.3, -1.5, 0.1, 1.2, -1.9, 1.9, 0.0]
    rig = make_rig(
        action_space="joint_abs",
        num_joints=7,
        script=[
            {"action": LEADER_Q, "gripper": 1.0, "active": True},
            {"action": leader_2, "active": True},
        ],
    )
    env, _ = rig.wrapped(action_mode="joint_abs", joint_limits=JOINT_LIMITS)
    env.reset()
    for leader in (LEADER_Q, leader_2):
        env.step(_zeros(env))
        np.testing.assert_allclose(_device_joint_target(rig), leader, atol=1e-6)


def test_joint_delta_leader_joints_reach_the_follower_arm_as_servo_targets(make_rig):
    leader_1 = [0.02, -0.03, 0.0, 0.04, -0.01, 0.01, 0.0]
    leader_2 = [0.03, -0.045, 0.01, 0.06, -0.015, 0.015, 0.02]  # within joint_scale_rad of follower
    rig = make_rig(
        action_space="joint_delta",
        num_joints=7,
        script=[{"action": leader_1, "active": True}, {"action": leader_2, "active": True}],
    )
    env, real_env = rig.wrapped(action_mode="joint_delta")
    env.reset()
    for leader in (leader_1, leader_2):
        follower = real_env.last_observation["state"]["joint_pos"].astype(np.float64)
        assert np.all(np.abs(np.array(leader) - follower) < real_env.config.joint_scale_rad)
        env.step(_zeros(env))
        np.testing.assert_allclose(_device_joint_target(rig), leader, atol=1e-6)


_GRIPPERLESS_SCRIPT = [
    {"action": [0.0] * 6, "active": False},
    {"action": [0.1] + [0.0] * 5, "active": True},
]


@pytest.mark.parametrize("gripper_mode", ["binary", "continuous"])
def test_teleop_without_gripper_holds_the_last_sent_gripper_action(make_rig, gripper_mode):
    rig = make_rig(gripper_mode=gripper_mode, has_gripper=False, script=_GRIPPERLESS_SCRIPT)
    env, real_env = rig.wrapped()
    env.reset()
    gripper = rig.device_robot.gripper
    # Closing on a wide object stalls the fingers above the binary open threshold (0.5).
    commands = []

    def close_on_wide_object():
        commands.append("close")
        gripper._width = 0.6

    def move_onto_wide_object(width_norm):
        commands.append(("move", width_norm))
        gripper._width = max(float(width_norm), 0.6)

    gripper.close = close_on_wide_object
    gripper.move = move_onto_wide_object
    original_open = gripper.open
    gripper.open = lambda: (commands.append("open"), original_open())[1]
    close = torch.zeros(1, 7)
    close[0, -1] = -1.0
    env.step(close)  # the policy closes the gripper
    # The measured-width hold would send 2 * 0.6 - 1 = 0.2, i.e. width 0.6: open / move(0.6).
    assert real_env.last_observation["state"]["gripper_pos"][0] == pytest.approx(0.6)
    commands.clear()

    _, _, _, _, info = env.step(torch.zeros(1, 7))
    assert info["intervene_action"][0, -1].item() == pytest.approx(-1.0)
    assert "open" not in commands
    if gripper_mode == "continuous":
        assert commands == [("move", pytest.approx(0.0))]  # (-1 + 1) / 2


def test_teleop_without_gripper_holds_open_after_reset(make_rig):
    rig = make_rig(has_gripper=False, script=_GRIPPERLESS_SCRIPT, on_end="idle")
    env, _ = rig.wrapped()
    env.reset()
    close = torch.zeros(1, 7)
    close[0, -1] = -1.0
    env.step(close)  # the policy closes the gripper ...
    env.reset()  # ... the reset reopens it, then the human takes over
    _, _, _, _, info = env.step(close)
    assert info["intervene_action"][0, -1].item() == pytest.approx(1.0)


def test_episode_end_pressed_while_resetting_is_dropped(make_rig):
    rig = make_rig(on_end="idle")
    env, real_env = rig.wrapped()
    env.reset()
    device_teleop = rig.device_robot.teleop
    original_reset = real_env.reset

    def reset_with_press(**kwargs):
        result = original_reset(**kwargs)
        device_teleop.push({"events": {"episode_end": True}})  # pressed while homing
        return result

    real_env.reset = reset_with_press
    env.reset()
    _, _, _, truncated, info = env.step(torch.zeros(1, 7))
    assert info["human_episode_end"] is False and bool(truncated) is False


def test_joint_sources_require_their_mapping_inputs():
    class _JointTeleop:
        action_space, has_gripper, action_dim = "joint_abs", True, 7

    contract = {"gripper_in_action": True, "state_fields": ("joint_pos", "gripper_pos")}
    with pytest.raises(ValueError, match="joint_limits"):
        TeleopPartSource(_JointTeleop(), env_contract=contract)
    with pytest.raises(ValueError, match="high > low"):
        TeleopPartSource(_JointTeleop(), env_contract=contract, joint_limits=[[0.0, 0.0]] * 7)
    _JointTeleop.action_space = "joint_delta"
    with pytest.raises(ValueError, match="joint_scale_rad"):
        TeleopPartSource(_JointTeleop(), env_contract=contract, joint_scale_rad=0.05)
    with pytest.raises(ValueError, match="joint_pos"):
        TeleopPartSource(
            _JointTeleop(),
            env_contract={**contract, "state_fields": ("gripper_pos",)},
            joint_scale_rad=0.05,
            last_observation_getter=lambda: None,
        )


# -- read failures ----------------------------------------------------------------------


def test_teleop_read_failure_raises_device_fault_without_a_transition(make_rig):
    rig = make_rig(fail_after=1, script=[{"action": [0.0] * 6, "active": False}])
    env, real_env = rig.wrapped()
    env.reset()
    env.step(torch.zeros(1, 7))
    elapsed = real_env._elapsed_steps
    with pytest.raises(DeviceFault):
        env.step(torch.zeros(1, 7))
    assert real_env._elapsed_steps == elapsed


# -- end to end: actor loop pushes the scripted intervention pattern -----------------


class _ZeroPolicy:
    def eval(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def predict(self, obs, deterministic=False):
        return torch.zeros(1, 8)


class _RecordingSyncClient:
    def __init__(self):
        self.pushed = []

    def start(self):
        pass

    def stop(self):
        pass

    def push_transition(self, transition):
        self.pushed.append(transition)

    def latest_policy_params(self):
        return None


class _StateVecEnv:
    num_envs = 1

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        from gymnasium import spaces

        self.single_observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.single_action_space = spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.single_action_space.low[None],
            high=self.single_action_space.high[None],
            dtype=np.float32,
        )

    def reset(self, seed=None):
        return torch.zeros(1, self.single_observation_space.shape[0]), {}

    def step(self, actions):
        raise AssertionError("the learner never steps the env")

    def close(self) -> None:
        return None


def test_remote_actor_loop_pushes_scripted_interventions_and_learner_routes_them(tmp_path):
    from rl_garden.algorithms.rlpd_hybrid import RLPDHybrid
    from rl_garden_real_world.hil_serl import HilSerlActorLoop, HilSerlLearnerLoop

    script = [
        {"action": [0.0] * 7, "active": False, "hold": 2},
        {"action": LEADER_Q, "gripper": 0.0, "active": True, "hold": 3},
        {"action": [0.0] * 7, "active": False},
        {"action": LEADER_Q, "active": True},
    ]
    expected = [False, False, True, True, True, False, True, False]
    rig = _Rig("remote", _teleop_config(action_space="joint_abs", num_joints=7, script=script, on_end="idle"))
    try:
        env, _ = rig.wrapped(
            action_mode="joint_abs", joint_limits=JOINT_LIMITS, state_fields=("joint_pos", "gripper_pos")
        )
        sync_client = _RecordingSyncClient()
        HilSerlActorLoop(env, _ZeroPolicy(), sync_client, control_hz=1_000_000.0).run(
            total_steps=len(expected)
        )
    finally:
        rig.close()

    assert [t["intervened"] for t in sync_client.pushed] == expected
    low, high = np.array(JOINT_LIMITS).T
    leader_action = 2.0 * (np.array(LEADER_Q) - low) / (high - low) - 1.0
    np.testing.assert_allclose(sync_client.pushed[2]["action"].numpy()[0, :7], leader_action, atol=1e-6)

    agent = RLPDHybrid(
        env=_StateVecEnv(obs_dim=8, action_dim=8),
        device="cpu",
        buffer_device="cpu",
        buffer_size=32,
        batch_size=4,
        learning_starts=1,
        training_freq=4,
        eval_freq=0,
        log_freq=0,
        net_arch=[8],
        discrete_hidden_dim=8,
    )
    agent.init_demo_buffer(buffer_size=16, demo_data_ratio=0.5)
    loop = HilSerlLearnerLoop(agent, "127.0.0.1", 0, checkpoint_dir=str(tmp_path), buffer_period=1000)
    for transition in sync_client.pushed:
        loop._on_transition(
            {**transition, "obs": transition["obs"]["state"], "next_obs": transition["next_obs"]["state"]}
        )
    assert len(agent.replay_buffer) == len(expected)
    assert len(agent.offline_replay_buffer) == sum(expected)
