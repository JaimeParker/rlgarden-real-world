"""Real-world rollout video recording (HIL-SERL's ``VideoRecorder``/
``VideoWrapper``, ``3rd_party/hil-serl/serl_launcher/serl_launcher/wrappers/
{video_recorder,video_wrapper}.py``): accumulates obs-stream image frames
(``VideoWrapper``'s approach -- reuses images already flowing through the
pipeline, no separate ``render()`` call, since real Franka envs don't
implement one meaningfully) and writes them to an mp4 on episode end
(``VideoRecorder``'s approach -- rl-garden has no separate offline
eval-video-saving script to pair a write-nothing ``VideoWrapper`` with).

Pipes raw frames to an ``ffmpeg`` subprocess rather than depending on
``imageio``/``tensorflow`` (upstream's choice, ``tf.io.gfile`` used only for
path handling) -- the exact convention already used by
``rl_garden/envs/robotwin/adapter.py``'s eval-video logging, duplicated here
(not imported) to avoid pulling RoboTwin's reward-registry module into the
real-world stack's otherwise-lazy import graph for an 8-line helper.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import gymnasium as gym
import torch


def _ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


class VideoRecorderWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        image_key: Optional[str] = None,
        fps: int = 10,
        save_prefix: str = "episode",
    ) -> None:
        super().__init__(env)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.image_key = image_key  # None -> auto-pick first non-"state" key
        self.fps = fps
        self.save_prefix = save_prefix
        self._num_videos = 0
        self._ffmpeg: Optional[subprocess.Popen] = None

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def _pick_key(self, obs: dict) -> str:
        if self.image_key is not None:
            return self.image_key
        return next(k for k in obs if k != "state")

    def _frame_bytes(self, obs: dict) -> bytes:
        key = self._pick_key(obs)
        frame = obs[key][0].detach().to("cpu", torch.uint8).numpy()
        return frame.tobytes()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._close_video()
        self._start_video(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._ffmpeg is not None:
            self._ffmpeg.stdin.write(self._frame_bytes(obs))
        if bool(terminated) or bool(truncated):
            self._close_video()
        return obs, reward, terminated, truncated, info

    def _start_video(self, obs: dict) -> None:
        key = self._pick_key(obs)
        h, w = obs[key].shape[1:3]
        out_path = self.save_dir / f"{self.save_prefix}_{self._num_videos:08d}.mp4"
        self._num_videos += 1
        self._ffmpeg = subprocess.Popen(
            [
                _ffmpeg_executable(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{w}x{h}",
                "-framerate",
                str(self.fps),
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                str(out_path),
            ],
            stdin=subprocess.PIPE,
        )
        self._ffmpeg.stdin.write(self._frame_bytes(obs))

    def _close_video(self) -> None:
        if self._ffmpeg is None:
            return
        self._ffmpeg.stdin.close()
        self._ffmpeg.wait(timeout=10.0)
        self._ffmpeg = None
