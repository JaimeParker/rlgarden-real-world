"""Eval-only deployment of a trained rl-garden policy on a ``real_robot`` backend robot.

No learner, no sync server, no data collection: restore the policy from a checkpoint, then
run ``episodes`` reset/step episodes and report steps, return, RTT and robot health.

Policy restore mirrors rl-garden's ``examples/eval_checkpoint.py`` (there is no library-level
eval loader): the online registry's ``<algo>`` Args are filled from the run's ``config.json``
(``<run>/config.json`` next to ``<run>/checkpoints/``) and the checkpoint's
``metadata.hyperparameters``, the agent is rebuilt with ``build_<algo>`` against the deploy
env's spaces, and ``BaseAlgorithm.load`` restores the weights (no replay, no optimizers).
Checkpoint/space mismatches therefore surface before the robot moves (nothing connects until
the first ``reset``). The env is the bare backend env: training-time observation wrappers
(HIL-SERL frame stacking / rotvec conversion) are not re-applied.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Optional

import torch

from rl_garden_real_world.envs.real_robot.config import RealRobotEnvConfig

_BACKEND = "real_robot"
# Inverse of rl_garden.common.effective_config.json_value's non-finite float encoding.
_JSON_FLOAT_SENTINELS = {"Infinity": float("inf"), "-Infinity": float("-inf"), "NaN": float("nan")}
DEVICE_ERROR_EXIT_CODE = 2


@dataclass
class DeployArgs:
    """Run a trained policy on a real robot (eval only)."""

    # Path to a checkpoint_<step>.pt written by a SERL/HIL-SERL learner (or any online run).
    checkpoint: str
    # Online-registry algorithm that produced the checkpoint ("rlpd" for serl,
    # "rlpd_hybrid" for hil_serl).
    algo: str = "rlpd"
    # Robot connection and env settings (exactly one of --real_robot.robot / .remote).
    real_robot: RealRobotEnvConfig = field(default_factory=RealRobotEnvConfig)
    # Number of reset/step episodes to run.
    episodes: int = 5
    # Use the policy's deterministic (mean) action.
    deterministic: bool = True
    # Per-episode step cap on top of real_robot.max_episode_steps; None means no extra cap.
    max_steps_per_episode: Optional[int] = None
    # Device the restored policy runs inference on.
    device: str = "cpu"
    # Seeds rl-garden RNGs and the first reset.
    seed: int = 0
    # Training run's config.json; defaults to <checkpoint dir>/../config.json when present.
    config_path: Optional[str] = None


@dataclass
class EpisodeResult:
    steps: int
    episode_return: float
    truncated: bool
    rtt_ms_mean: Optional[float]
    health_ok: Optional[bool]


# -- policy restore -------------------------------------------------------------------


def _set_existing_attr(target: Any, key: str, value: Any) -> None:
    if not hasattr(target, key):
        return
    current = getattr(target, key)
    if isinstance(current, float) and isinstance(value, str) and value in _JSON_FLOAT_SENTINELS:
        value = _JSON_FLOAT_SENTINELS[value]
    if is_dataclass(current) and value is None:
        # Collapsed sentinel dataclass (e.g. ObsGroups()) serialized as None: keep the default.
        return
    if is_dataclass(current) and isinstance(value, Mapping):
        if type(current).__dataclass_params__.frozen:
            names = {f.name for f in fields(current)}
            setattr(target, key, replace(current, **{k: v for k, v in value.items() if k in names}))
        else:
            _apply_mapping(current, value)
        return
    setattr(target, key, value)


def _apply_mapping(target: Any, values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        _set_existing_attr(target, key, value)


def _config_path(args: DeployArgs) -> Path:
    if args.config_path is not None:
        return Path(args.config_path)
    return Path(args.checkpoint).parent.parent / "config.json"


def restore_training_args(args: DeployArgs) -> Any:
    """The ``<algo>`` online Args as the checkpoint's run trained them, with inference-only overrides."""
    from rl_garden.common.checkpoint import load_checkpoint_file
    from rl_garden.training.online._registry import registry

    registry.discover()
    entries = registry.entries()
    if args.algo not in entries:
        raise ValueError(f"unknown algo {args.algo!r}; online registry has {sorted(entries)}")
    training_args = entries[args.algo].args_cls()

    config_path = _config_path(args)
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        _apply_mapping(training_args, config.get("inputs", config.get("args", config)))
    checkpoint = load_checkpoint_file(args.checkpoint, map_location="cpu")
    hyperparameters = checkpoint.get("metadata", {}).get("hyperparameters", {})
    if isinstance(hyperparameters, Mapping):
        _apply_mapping(training_args, hyperparameters)

    _apply_mapping(
        training_args,
        {
            "env_backend": _BACKEND,
            "num_envs": 1,
            "num_eval_envs": 1,
            "seed": args.seed,
            "buffer_device": "cpu",
            "buffer_size": 8,
            "offline_dataset": None,
            "load_checkpoint": None,
            "load_replay_buffer": False,
            "checkpoint_dir": None,
            "checkpoint_freq": 0,
            "save_replay_buffer": False,
            "save_final_checkpoint": False,
            "eval_freq": 0,
            "capture_video": False,
            "log_type": "none",
        },
    )
    return training_args


def build_deploy_env(args: DeployArgs, training_args: Any):
    """The real_robot env with the observation layout the policy was trained on."""
    import rl_garden_real_world.envs.backends.real_robot  # noqa: F401  (registers the backend)
    from rl_garden.envs.backend_registry import EnvRequest, make_training_envs
    from rl_garden.observations import ObservationConfig

    request = EnvRequest(
        env_id=getattr(training_args, "env_id", "real_robot"),
        num_envs=1,
        observation=getattr(training_args, "obs", None) or ObservationConfig(),
        control_mode="",
        render_mode="rgb_array",
        seed=args.seed,
        num_eval_envs=1,
        create_eval_env=False,
        backend_config=args.real_robot,
    )
    env, _ = make_training_envs(_BACKEND, request)
    return env


def restore_policy(args: DeployArgs, training_args: Any, env: Any):
    """Rebuild the ``<algo>`` agent on ``env``'s spaces and load the checkpoint weights."""
    import importlib

    from rl_garden.common import seed_everything

    seed_everything(args.seed)
    module = importlib.import_module(f"rl_garden.training.online.{args.algo}")
    build_agent = getattr(module, f"build_{args.algo}")
    agent = build_agent(training_args, env, None, logger=None, checkpoint_dir=None)
    agent.load(args.checkpoint, load_replay_buffer=False, load_optimizers=False)
    # Online builders construct on device "auto"; inference runs where --device says.
    return agent.policy.to(torch.device(args.device))


# -- episode loop ---------------------------------------------------------------------


def _find_robot(env: Any) -> Any:
    """``env.robot`` through any wrappers (gymnasium wrappers do not forward attributes)."""
    while env is not None:
        robot = getattr(env, "robot", None)
        if robot is not None:
            return robot
        env = getattr(env, "env", None)
    return None


def _health_ok(env: Any) -> Optional[bool]:
    robot = _find_robot(env)
    if robot is None:
        return None
    return bool(robot.health().ok)


def _obs_to_device(obs: Any, device: torch.device) -> Any:
    if isinstance(obs, dict):
        return {key: value.to(device) for key, value in obs.items()}
    return obs.to(device)


def _env_device(obs: Any) -> torch.device:
    sample = next(iter(obs.values())) if isinstance(obs, dict) else obs
    return sample.device


def _run_episode(args: DeployArgs, env: Any, policy: Any, device: torch.device, seed) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    steps, episode_return, truncated = 0, 0.0, False
    rtts: list[float] = []
    while True:
        # Same call as ActorLoop._predict.
        with torch.no_grad():
            action = policy.predict(_obs_to_device(obs, device), deterministic=args.deterministic)
        obs, reward, terminated, env_truncated, info = env.step(action.to(_env_device(obs)))
        steps += 1
        episode_return += float(torch.as_tensor(reward).sum())
        rtt = info.get("rtt_ms") if isinstance(info, Mapping) else None
        if rtt is not None:
            rtts.append(float(rtt))
        hit_cap = args.max_steps_per_episode is not None and steps >= args.max_steps_per_episode
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(env_truncated).any()) or hit_cap:
            truncated = bool(torch.as_tensor(env_truncated).any()) or hit_cap
            break
    return EpisodeResult(
        steps=steps,
        episode_return=episode_return,
        truncated=truncated,
        rtt_ms_mean=sum(rtts) / len(rtts) if rtts else None,
        health_ok=_health_ok(env),
    )


def _format_rtt(rtt_ms: Optional[float]) -> str:
    return "n/a" if rtt_ms is None else f"{rtt_ms:.1f}"


def run_deploy(args: DeployArgs, *, env: Any = None, policy: Any = None) -> list[EpisodeResult]:
    """Run ``args.episodes`` episodes; ``env``/``policy`` are injectable for tests.

    The env is always closed (which disconnects the robot), including on ``DeviceError``.
    """
    training_args = restore_training_args(args) if env is None or policy is None else None
    if env is None:
        env = build_deploy_env(args, training_args)
    try:
        if policy is None:
            policy = restore_policy(args, training_args, env)
        policy.eval()
        device = torch.device(args.device)
        results: list[EpisodeResult] = []
        for episode in range(args.episodes):
            result = _run_episode(args, env, policy, device, seed=args.seed if episode == 0 else None)
            results.append(result)
            print(
                f"[deploy] episode={episode} steps={result.steps} "
                f"return={result.episode_return:.3f} truncated={result.truncated} "
                f"rtt_ms={_format_rtt(result.rtt_ms_mean)} health_ok={result.health_ok}",
                flush=True,
            )
        _print_summary(results)
        return results
    finally:
        env.close()


def _print_summary(results: list[EpisodeResult]) -> None:
    count = len(results)
    mean_steps = sum(r.steps for r in results) / count if count else 0.0
    mean_return = sum(r.episode_return for r in results) / count if count else 0.0
    rtts = [r.rtt_ms_mean for r in results if r.rtt_ms_mean is not None]
    all_healthy = all(r.health_ok is not False for r in results)
    print(
        f"[deploy] summary episodes={count} mean_steps={mean_steps:.1f} "
        f"mean_return={mean_return:.3f} rtt_ms={_format_rtt(sum(rtts) / len(rtts) if rtts else None)} "
        f"all_healthy={all_healthy}",
        flush=True,
    )


def main(args: DeployArgs, *, env: Any = None, policy: Any = None) -> int:
    """CLI wrapper: exit code 0, or 2 after printing a ``DeviceError``'s message and hint."""
    from robot_infra.devices.errors import DeviceError

    try:
        run_deploy(args, env=env, policy=policy)
    except DeviceError as error:
        print(f"[deploy] device error: {error.message}", file=sys.stderr, flush=True)
        if error.hint:
            print(f"[deploy] hint: {error.hint}", file=sys.stderr, flush=True)
        return DEVICE_ERROR_EXIT_CODE
    return 0
