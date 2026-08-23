"""Tests for VideoRecorderWrapper. No real ffmpeg binary needed --
_ffmpeg_executable is monkeypatched to a tiny shell stub that copies stdin
to a marker file, so these assert the wrapper's own frame-piping/lifecycle
logic without depending on ffmpeg being installed in the test environment.
"""
from __future__ import annotations

import os
import stat

import gymnasium as gym
import torch

from rl_garden_real_world.envs.wrappers import video_recorder as video_recorder_module
from rl_garden_real_world.envs.wrappers.video_recorder import VideoRecorderWrapper


class _FakeEnv(gym.Env):
    num_envs = 1

    def __init__(self):
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        return self._obs(), {}

    def _obs(self):
        frame = torch.full((1, 2, 2, 3), float(self._t), dtype=torch.uint8)
        return {"state": torch.zeros(1, 4), "wrist_cam": frame}

    def step(self, action):
        del action
        self._t += 1
        terminated = self._t >= 2
        return (
            self._obs(),
            torch.tensor([0.0]),
            torch.tensor([terminated]),
            torch.tensor([False]),
            {},
        )


def _install_stub_ffmpeg(tmp_path, monkeypatch):
    """A minimal cat-to-file stub standing in for real ffmpeg: reads all of
    stdin and writes it to the output path (the wrapper's last CLI arg)."""
    stub = tmp_path / "fake_ffmpeg.py"
    stub.write_text(
        "import sys\n"
        "out_path = sys.argv[-1]\n"
        "data = sys.stdin.buffer.read()\n"
        "with open(out_path, 'wb') as f:\n"
        "    f.write(data)\n"
    )
    wrapper_script = tmp_path / "fake_ffmpeg"
    wrapper_script.write_text(f"#!/bin/sh\nexec python3 {stub} \"$@\"\n")
    wrapper_script.chmod(wrapper_script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(
        video_recorder_module, "_ffmpeg_executable", lambda: str(wrapper_script)
    )


def test_reset_and_step_write_frames_to_output_file(tmp_path, monkeypatch):
    _install_stub_ffmpeg(tmp_path, monkeypatch)
    save_dir = tmp_path / "videos"
    wrapped = VideoRecorderWrapper(_FakeEnv(), save_dir=str(save_dir), image_key="wrist_cam")

    wrapped.reset()
    wrapped.step(torch.zeros(1))  # not terminal -- video stays open
    wrapped.step(torch.zeros(1))  # terminal at t=2 -- closes the video

    out_path = save_dir / "episode_00000000.mp4"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_second_episode_writes_a_new_file(tmp_path, monkeypatch):
    _install_stub_ffmpeg(tmp_path, monkeypatch)
    save_dir = tmp_path / "videos"
    wrapped = VideoRecorderWrapper(_FakeEnv(), save_dir=str(save_dir), image_key="wrist_cam")

    wrapped.reset()
    wrapped.step(torch.zeros(1))
    wrapped.step(torch.zeros(1))  # closes episode 0
    wrapped.reset()  # opens episode 1

    assert (save_dir / "episode_00000000.mp4").exists()
    assert wrapped._num_videos == 2


def test_image_key_auto_picked_when_not_given(tmp_path, monkeypatch):
    _install_stub_ffmpeg(tmp_path, monkeypatch)
    wrapped = VideoRecorderWrapper(_FakeEnv(), save_dir=str(tmp_path / "videos"))
    assert wrapped._pick_key({"state": None, "wrist_cam": None}) == "wrist_cam"


def test_getattr_forwards_to_wrapped_env(tmp_path, monkeypatch):
    _install_stub_ffmpeg(tmp_path, monkeypatch)
    wrapped = VideoRecorderWrapper(_FakeEnv(), save_dir=str(tmp_path / "videos"))
    assert wrapped.num_envs == 1
