"""RealRobotEnv over robot_infra fake devices, both local and through RemoteRobot + LocalClient."""
from __future__ import annotations

import time

import numpy as np
import pytest

import robot_infra.devices.backends  # noqa: F401  (registers the fake backends)
import robot_infra.devices.claims as claims
from robot_infra.devices import CameraConfig, DeviceFault, GripperConfig
from robot_infra.devices.robots.franka import FrankaConfig, FrankaRobot
from robot_infra.server import RemoteRobot
from robot_infra.server.service import RobotService
from robot_infra.server.transport import LocalClient

from rl_garden_real_world.envs.real_robot import RealRobotEnv, RealRobotEnvConfig

ZERO_HOME = [0.0] * 7
FRAME_HW = (24, 32)


def _fake_franka() -> FrankaRobot:
    return FrankaRobot(
        FrankaConfig(
            arm_backend="fake",
            gripper=GripperConfig(backend="fake"),
            cameras=[CameraConfig(name="wrist", backend="fake", width=32, height=24, fps=60)],
        )
    )


@pytest.fixture(autouse=True)
def _claim_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "_DEFAULT_LOCK_DIR", tmp_path / "claims")


class _Rig:
    """A robot to inject plus the fake arm/gripper that actually execute its commands."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.service = None
        if mode == "local":
            self.robot = _fake_franka()
            self.device_robot = self.robot
        else:
            self.service = RobotService(_fake_franka(), watchdog_s=30.0)
            self.robot = RemoteRobot.from_client(LocalClient(self.service))
            self.device_robot = self.service.robot
        self.envs: list[RealRobotEnv] = []

    @property
    def arm(self):
        return self.device_robot.arm

    def config(self, **overrides) -> RealRobotEnvConfig:
        values = dict(cameras=["wrist"], home_joints=ZERO_HOME, reset_settle_s=0.0)
        if self.mode == "local":
            values["robot"] = {"type": "franka", "arm_backend": "fake"}
        else:
            values.update(remote="local", image_size=FRAME_HW)
        values.update(overrides)
        return RealRobotEnvConfig(**values)

    def env(self, **overrides) -> RealRobotEnv:
        env = RealRobotEnv(self.config(**overrides), robot=self.robot)
        self.envs.append(env)
        return env

    def spy_commands(self, monkeypatch) -> list[dict]:
        """Record every action dict the env sends to the robot."""
        sent: list[dict] = []
        name = "send_action" if self.mode == "local" else "step"
        original = getattr(self.robot, name)

        def spy(action):
            sent.append(action)
            return original(action)

        monkeypatch.setattr(self.robot, name, spy)
        return sent

    def close(self) -> None:
        for env in self.envs:
            env.close()
        if self.robot.is_connected:
            self.robot.disconnect()
        if self.service is not None:
            self.service.close()


@pytest.fixture(params=["local", "remote"])
def rig(request):
    rig = _Rig(request.param)
    yield rig
    rig.close()


def _is_subsequence(expected: list[str], log: list[str]) -> bool:
    it = iter(log)
    return all(item in it for item in expected)


# -- config -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(action_mode="velocity"), "action_mode"),
        (dict(num_joints=0), "num_joints"),
        (dict(joint_limits=[[-1.0, 1.0]] * 6), "joint_limits"),
        (dict(joint_limits=[[-1.0, 1.0, 2.0]] * 7), "joint_limits"),
        (dict(safety_box=[0.0] * 5), "safety_box"),
        (dict(state_fields=("tcp_pose", "ee_wrench")), "state_fields"),
        (dict(action_mode="joint_abs"), "joint_limits"),
        (dict(home_joints=[0.0] * 6), "home_joints"),
    ],
)
def test_config_validation_errors(kwargs, match):
    if "remote" not in kwargs and kwargs:
        kwargs = {"robot": {"type": "franka"}, **kwargs}
    with pytest.raises(ValueError, match=match):
        RealRobotEnvConfig(**kwargs)


def test_config_default_constructs_and_parses_from_empty_cli():
    import tyro

    assert RealRobotEnvConfig().robot is None
    assert tyro.cli(RealRobotEnvConfig, args=[]) == RealRobotEnvConfig()


@pytest.mark.parametrize(
    "kwargs", [dict(), dict(robot={"type": "franka"}, remote="host:1")]
)
def test_env_requires_exactly_one_robot_source(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        RealRobotEnv(RealRobotEnvConfig(**kwargs), robot=object())


def test_config_remote_with_cameras_requires_image_size():
    with pytest.raises(ValueError, match="image_size"):
        RealRobotEnvConfig(remote="host:1", cameras=["wrist"])
    RealRobotEnvConfig(remote="host:1", cameras=["wrist"], image_size=(16, 16))


# -- spaces -------------------------------------------------------------------------


def test_spaces_for_seven_joint_cartesian(rig):
    env = rig.env()
    assert env.action_space.shape == (7,)
    assert env.action_space.dtype == np.float32
    state = env.observation_space["state"]
    assert list(state.spaces) == list(env.config.state_fields)
    assert state["tcp_pose"].shape == (7,)
    assert state["joint_pos"].shape == (7,)
    assert state["gripper_pos"].shape == (1,)
    assert state["tcp_pose"].dtype == np.float32
    assert np.all(np.isinf(state["tcp_pose"].high))
    frame = env.observation_space["frames"]["wrist"]
    assert frame.shape == (*FRAME_HW, 3) and frame.dtype == np.uint8


def test_spaces_for_joint_delta_and_no_gripper(rig):
    assert rig.env(action_mode="joint_delta").action_space.shape == (8,)
    assert rig.env(gripper_in_action=False).action_space.shape == (6,)


def test_construction_does_not_connect(rig):
    rig.env()
    assert not rig.robot.is_connected


# -- reset ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_mode, servo", [("cartesian_delta", "enter:cartesian_servo"), ("joint_delta", "enter:joint_servo")]
)
def test_reset_idles_homes_opens_and_enters_servo(rig, action_mode, servo):
    env = rig.env(action_mode=action_mode)
    env.reset()
    env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    rig.arm.call_log.clear()
    rig.device_robot.gripper.close()

    obs, info = env.reset()

    assert _is_subsequence(["stop", "enter:idle", "move_to_joints", servo], rig.arm.call_log)
    assert rig.arm.call_log.index("move_to_joints") < rig.arm.call_log.index(servo)
    assert rig.arm.control_mode.value == servo.split(":")[1]
    assert rig.device_robot.gripper.width == 1.0
    np.testing.assert_allclose(obs["state"]["joint_pos"], ZERO_HOME)
    assert info == {}


def test_reset_rejects_joint_count_mismatch_with_hint(rig):
    env = rig.env(num_joints=6, home_joints=None)
    with pytest.raises(DeviceFault) as excinfo:
        env.reset()
    assert "num_joints=7" in excinfo.value.hint
    assert not rig.robot.is_connected


def test_reset_rejects_unknown_camera_with_hint(rig):
    env = rig.env(cameras=["front"], image_size=FRAME_HW)
    with pytest.raises(DeviceFault, match="front") as excinfo:
        env.reset()
    assert excinfo.value.hint


def test_reset_without_any_home_joints_raises_hint():
    service = RobotService(_fake_franka(), watchdog_s=30.0)
    robot = RemoteRobot.from_client(LocalClient(service))
    env = RealRobotEnv(
        RealRobotEnvConfig(remote="local", reset_settle_s=0.0), robot=robot
    )
    try:
        with pytest.raises(DeviceFault) as excinfo:
            env.reset()
        assert "home_joints" in excinfo.value.hint
    finally:
        env.close()
        service.close()


def test_local_reset_falls_back_to_robot_home_joints():
    robot = _fake_franka()
    env = RealRobotEnv(
        RealRobotEnvConfig(robot={"type": "franka"}, reset_settle_s=0.0), robot=robot
    )
    try:
        obs, _ = env.reset()
        np.testing.assert_allclose(obs["state"]["joint_pos"], robot.config.home_joints, atol=1e-6)
    finally:
        env.close()


# -- step -------------------------------------------------------------------------


def test_cartesian_step_target(rig, monkeypatch):
    env = rig.env()
    env.reset()
    sent = rig.spy_commands(monkeypatch)

    env.step(np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float32))

    command = sent[-1]
    pose = np.asarray(command["arm"]["tcp_pose"])
    np.testing.assert_allclose(pose[:3], [env.config.xyz_scale_m, 0.0, 0.0])
    np.testing.assert_allclose(pose[3:], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
    assert float(np.asarray(command["gripper"]["gripper"])[0]) == 1.0
    np.testing.assert_allclose(rig.arm._pose_target[:3], [env.config.xyz_scale_m, 0.0, 0.0])


def test_cartesian_rotation_composes_delta_on_current(rig, monkeypatch):
    from scipy.spatial.transform import Rotation

    env = rig.env(rot_scale_rad=0.5)
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.array([0, 0, 1, 0, 0, 1, -1], dtype=np.float32))
    pose = np.asarray(sent[-1]["arm"]["tcp_pose"])
    expected = Rotation.from_rotvec([0, 0, 0.5]).as_quat()
    np.testing.assert_allclose(pose[3:], expected, atol=1e-9)
    assert float(np.asarray(sent[-1]["gripper"]["gripper"])[0]) == 0.0


def test_cartesian_target_clipped_to_safety_box(rig, monkeypatch):
    env = rig.env(safety_box=[-0.005, -1.0, -1.0, 0.005, 1.0, 0.01])
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sent[-1]["arm"]["tcp_pose"])[:3], [0.005, 0.0, 0.01])
    env.step(np.array([-1, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    assert np.asarray(sent[-1]["arm"]["tcp_pose"])[0] == pytest.approx(-0.005)


def test_out_of_range_action_is_clipped(rig, monkeypatch):
    env = rig.env()
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.array([5, 0, 0, 0, 0, 0, 3], dtype=np.float32))
    assert np.asarray(sent[-1]["arm"]["tcp_pose"])[0] == pytest.approx(env.config.xyz_scale_m)
    assert float(np.asarray(sent[-1]["gripper"]["gripper"])[0]) == 1.0


def test_joint_delta_clipped_to_joint_limits(rig, monkeypatch):
    env = rig.env(action_mode="joint_delta", joint_limits=[[-0.01, 0.01]] * 7)
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.array([1, -1, 0.1, 0, 0, 0, 0, 1], dtype=np.float32))
    q = np.asarray(sent[-1]["arm"]["joint_pos"])
    np.testing.assert_allclose(q, [0.01, -0.01, 0.005, 0, 0, 0, 0])
    np.testing.assert_allclose(rig.arm._joint_target, q)


def test_joint_delta_without_limits_is_unclipped(rig, monkeypatch):
    env = rig.env(action_mode="joint_delta")
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.ones(8, dtype=np.float32))
    np.testing.assert_allclose(np.asarray(sent[-1]["arm"]["joint_pos"]), [env.config.joint_scale_rad] * 7)


def test_joint_abs_maps_unit_range_onto_limits(rig, monkeypatch):
    limits = [[-1.0, 1.0], [0.0, 2.0], [-2.0, -1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    env = rig.env(action_mode="joint_abs", joint_limits=limits)
    env.reset()
    sent = rig.spy_commands(monkeypatch)
    env.step(np.array([-1, 0, 1, 0.5, -0.5, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(
        np.asarray(sent[-1]["arm"]["joint_pos"]), [-1.0, 1.0, -1.0, 0.75, 0.25, 0.5, 0.5]
    )


def test_step_before_reset_raises(rig):
    with pytest.raises(RuntimeError, match="reset"):
        rig.env().step(np.zeros(7, dtype=np.float32))


def test_step_paces_to_control_hz(rig):
    env = rig.env(control_hz=50.0)
    env.reset()
    action = np.zeros(7, dtype=np.float32)
    start = time.monotonic()
    env.step(action)
    env.step(action)
    assert time.monotonic() - start >= 0.02


def test_truncated_at_max_episode_steps(rig):
    env = rig.env(max_episode_steps=2)
    env.reset()
    action = np.zeros(7, dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)
    assert (reward, terminated, truncated, info["elapsed_steps"]) == (0.0, False, False, 1)
    _, _, _, truncated, info = env.step(action)
    assert truncated is True and info["elapsed_steps"] == 2
    env.reset()
    _, _, _, truncated, _ = env.step(action)
    assert truncated is False


def test_step_info_carries_tcp_pose_and_rtt(rig):
    env = rig.env()
    env.reset()
    _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert info["tcp_pose"].shape == (7,) and info["tcp_pose"].dtype == np.float32
    if rig.mode == "remote":
        assert isinstance(info["rtt_ms"], float)
    else:
        assert info["rtt_ms"] is None


def test_observation_dtypes_and_shapes(rig):
    env = rig.env()
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["state"]["gripper_pos"].tolist() == [1.0]
    obs, *_ = env.step(np.zeros(7, dtype=np.float32))
    assert env.observation_space.contains(obs)
    for field, value in obs["state"].items():
        assert value.dtype == np.float32, field
    assert obs["frames"]["wrist"].shape == (*FRAME_HW, 3)
    assert obs["frames"]["wrist"].dtype == np.uint8
    assert env.render() is obs["frames"]["wrist"]


def test_frames_resized_to_image_size(rig):
    pytest.importorskip("PIL")
    env = rig.env(image_size=(16, 16))
    obs, _ = env.reset()
    assert obs["frames"]["wrist"].shape == (16, 16, 3)
    assert obs["frames"]["wrist"].dtype == np.uint8
    assert env.observation_space.contains(obs)


def test_piper_style_state_fields(rig):
    env = rig.env(state_fields=("tcp_pose", "joint_pos", "joint_vel", "gripper_pos"))
    obs, _ = env.reset()
    assert list(obs["state"]) == ["tcp_pose", "joint_pos", "joint_vel", "gripper_pos"]


def test_close_disconnects_and_task_description_defaults_empty(rig):
    env = rig.env()
    assert env.task_description == ""
    assert env.render() is None
    env.reset()
    assert rig.robot.is_connected
    env.close()
    assert not rig.robot.is_connected
    env.close()
