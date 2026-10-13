"""HIL-SERL run function.

Targets HIL-SERL's ``train_rlpd.py`` capability set (online RLPD + demo
mixing + human intervention + reward classifier), not BC or HG-DAgger's
growing-dataset problem -- see
docs/superpowers/specs/2026-07-09-hil-serl-migration-design.md.
"""
from __future__ import annotations

import dataclasses
import logging

import torch

_LOGGER = logging.getLogger(__name__)


def _state_starts_with_tcp_pose(args) -> bool:
    """FrankaRealEnv's state always starts with tcp_pose; real_robot's follows state_fields."""
    if args.env_backend != "real_robot":
        return True
    fields = args.real_robot.state_fields
    return bool(fields) and fields[0] == "tcp_pose"


def _hil_serl_env_request(args, run_name):
    from rl_garden.common.env_args import make_env_request

    return make_env_request(args, run_name)


def _build_env(args, env_request):
    """Composes the env wrappers built in the SERL v1 round but never wired
    up until now: temporal observation stacking (innermost, optional) ->
    rotvec observation conversion (optional) -> reward classifier -> human
    intervention -> optional reset-free forward/backward (outer, per
    ``FWBWActorLoop``/``FWBWResetFreeWrapper``'s documented requirement that
    it sit outermost)."""
    from rl_garden.envs.backend_registry import make_training_envs
    from rl_garden.envs.wrappers.frame_stack import ImageFrameStackWrapper
    from rl_garden.envs.wrappers.fwbw_reset_free import FWBWResetFreeWrapper
    from rl_garden.envs.wrappers.reward_classifier import RewardClassifierWrapper
    from rl_garden_real_world.envs.wrappers.rotvec_obs import RotvecObsWrapper
    from rl_garden_real_world.envs.wrappers.grasp_penalty import GraspPenaltyWrapper
    from rl_garden_real_world.envs.wrappers.teleop_intervention import TeleopInterventionWrapper

    if args.env_backend == "real_robot":
        cfg = args.real_robot
        if cfg.action_mode != "cartesian_delta" or not cfg.gripper_in_action:
            raise ValueError(
                "hil_serl teleop intervention requires real_robot.action_mode=cartesian_delta "
                "with gripper_in_action=true (7-D action); got "
                f"action_mode={cfg.action_mode!r}, gripper_in_action={cfg.gripper_in_action}"
            )

    env, _ = make_training_envs(args.env_backend, env_request)

    if args.obs.frame_stack > 1:
        # FrankaRealEnv has no _init_raw_obs/update_obs_space (ManiSkill-only
        # APIs) and its camera keys are user-configured, not necessarily
        # "rgb*"/"depth*"-prefixed -- so image_keys must be passed explicitly.
        # Reuses classifier_image_keys (same "which keys are cameras"
        # question the classifier wrapper below already answers) with the
        # same "every non-state key" fallback used there.
        stack_image_keys = args.classifier_image_keys or tuple(
            k for k in env.single_observation_space.spaces if k != "state"
        )
        env = ImageFrameStackWrapper(env, frame_stack=args.obs.frame_stack, image_keys=stack_image_keys)

    if args.convert_obs_rotation:
        if _state_starts_with_tcp_pose(args):
            env = RotvecObsWrapper(env)
        else:
            _LOGGER.warning(
                "convert_obs_rotation skipped: real_robot state_fields %s do not start with "
                "tcp_pose, so there is no quaternion at state[3:7]",
                list(args.real_robot.state_fields),
            )

    if args.classifier_checkpoint is not None:
        from rl_garden.common.utils import get_device
        from rl_garden.models.reward.success.model import load_classifier_fn

        # FrankaRealEnv's observation space is {"state": ..., <camera_key>: image, ...}
        # (see rl_garden_real_world/envs/franka_real/env.py) -- camera keys are whatever
        # the user configured (cfg.camera_keys), not necessarily "rgb*"/"depth*"
        # prefixed, so every non-"state" key is an image key here.
        image_keys = args.classifier_image_keys or tuple(
            k for k in env.single_observation_space.spaces if k != "state"
        )
        classifier_fn = load_classifier_fn(
            args.classifier_checkpoint,
            env.single_observation_space,
            image_keys,
            device=get_device("auto"),
        )
        env = RewardClassifierWrapper(env, classifier_fn, threshold=args.classifier_threshold)

    env = TeleopInterventionWrapper(env, device=args.teleop_device)

    if args.use_grasp_penalty:
        # Must wrap outside TeleopInterventionWrapper so
        # info["intervene_action"] (the actually-executed action under
        # human intervention) is already present when this runs.
        env = GraspPenaltyWrapper(env, penalty=args.grasp_penalty_value)

    if args.fwbw:
        env = FWBWResetFreeWrapper(env)

    return env


def _run_actor(args) -> None:
    """Non-FWBW: one scratch agent, one ``HilSerlActorLoop`` -- same shape as
    ``serl.py``'s ``_run_actor``. FWBW: two scratch agents (forward/backward)
    and two sync clients, driving the existing ``FWBWActorLoop`` (a single
    actor process still only ever drives the one physical robot; it's the
    *policy/sync-client pair* that's doubled, switched per step based on
    ``info["fwbw_direction"]``)."""
    from rl_garden_real_world import FWBWActorLoop
    from rl_garden_real_world.hil_serl import HilSerlActorLoop, HilSerlActorSyncClient

    env_request = _hil_serl_env_request(args, run_name="real_world_actor")
    env = _build_env(args, env_request)
    if args.record_video_dir is not None:
        # Applied here (not inside _build_env) since it's actor-only --
        # _build_env is shared by actor/learner/eval, and only the actor
        # process ever calls env.step() against the real robot.
        from rl_garden_real_world.envs.wrappers.video_recorder import VideoRecorderWrapper

        env = VideoRecorderWrapper(env, save_dir=args.record_video_dir)

    if not args.fwbw:
        agent = _rebuild_scratch_agent_for_env(args, env)
        sync_client = HilSerlActorSyncClient(f"http://{args.sync_host}:{args.sync_port}")
        loop = HilSerlActorLoop(
            env,
            agent.policy,
            sync_client,
            control_hz=args.control_hz,
            device=agent.device,
            deterministic=args.deterministic_actor,
            seed=args.seed,
        )
        loop.run()
        return

    backward_port = args.sync_port_backward or (args.sync_port + 1)
    agents = {
        "forward": _rebuild_scratch_agent_for_env(args, env),
        "backward": _rebuild_scratch_agent_for_env(args, env),
    }
    sync_clients = {
        "forward": HilSerlActorSyncClient(f"http://{args.sync_host}:{args.sync_port}"),
        "backward": HilSerlActorSyncClient(f"http://{args.sync_host}:{backward_port}"),
    }
    loop = FWBWActorLoop(
        env,
        {k: a.policy for k, a in agents.items()},
        sync_clients,
        control_hz=args.control_hz,
        device=agents["forward"].device,
        deterministic=args.deterministic_actor,
        seed=args.seed,
    )
    loop.run()


def _rebuild_scratch_agent_for_env(args, env):
    from rl_garden.training.online.rlpd_hybrid import build_rlpd_hybrid

    scratch_args = dataclasses.replace(
        args, buffer_size=8, load_checkpoint=None, offline_dataset=None
    )
    return build_rlpd_hybrid(scratch_args, env, None, logger=None, checkpoint_dir=None)


def _run_learner(args) -> None:
    """Builds a full ``RLPDHybrid`` instance and drives
    ``rl_garden_real_world.hil_serl.HilSerlLearnerLoop``. In FWBW mode, a
    learner process is launched once per direction (``--fwbw_direction
    forward|backward``), each bound to its own sync port -- matching the
    SERL v1 design's "two full actor/learner pairs" for reset-free
    training. Non-FWBW: identical shape to ``serl.py``'s ``_run_learner``."""
    import os
    import time

    from rl_garden.common import Logger, seed_everything
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden_real_world.hil_serl import HilSerlLearnerLoop
    from rl_garden.training.online.rlpd_hybrid import build_rlpd_hybrid

    seed_everything(args.seed)

    direction_suffix = f"_{args.fwbw_direction}" if args.fwbw else ""
    port = (
        (args.sync_port_backward or (args.sync_port + 1))
        if args.fwbw and args.fwbw_direction == "backward"
        else args.sync_port
    )

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = (
        args.exp_name or f"{args.env_id}__hil_serl_real_world{direction_suffix}__{args.seed}__{int(time.time())}"
    )
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args, training_phase="real_world", algorithm="hil_serl", run_name=run_name, log_dir=args.log_dir
    )
    logger = Logger.create(
        log_type=args.log_type,
        log_dir=args.log_dir,
        run_name=run_name,
        config=resolved_config,
        start_time=start_time,
        log_keywords=args.log_keywords,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        log_group=args.log_group or args.env_id,
    )

    env_request = _hil_serl_env_request(args, run_name=run_name)
    env = _build_env(args, env_request)

    agent = build_rlpd_hybrid(args, env, None, logger=logger, checkpoint_dir=checkpoint_dir)
    agent.init_demo_buffer(args.demo_buffer_size, args.demo_data_ratio)

    loop = HilSerlLearnerLoop(
        agent,
        host=args.sync_host,
        port=port,
        checkpoint_dir=checkpoint_dir,
        buffer_period=args.buffer_period,
        demo_dataset_paths=args.demo_dataset_paths,
        train_freq=args.train_freq,
        publish_freq=args.publish_freq,
    )
    try:
        loop.run()
    finally:
        logger.close()


def _run_eval(args) -> None:
    """Loads a checkpoint and runs deterministic evaluation episodes directly
    on the real robot -- no sync client, no learner process, no data
    collection (item 10 of docs/hil-serl-roadmap.md, HIL-SERL's own
    ``--eval_checkpoint_step``/``--eval_n_trajs``, ``train_rlpd.py:71-110``).
    Per-episode success is read off ``terminated``: ``RewardClassifierWrapper``
    sets it True exactly when the classifier judges success
    (``rl_garden/envs/wrappers/reward_classifier.py``); a ``truncated`` episode
    (timeout) counts as a failure. FWBW is out of scope this round -- same
    "single-arm only" boundary already documented elsewhere in the real-world
    RL stack."""
    from rl_garden_real_world.actor_loop import ActorLoop
    from rl_garden.training.online.rlpd_hybrid import build_rlpd_hybrid

    if args.fwbw:
        raise NotImplementedError("hil_serl eval mode does not support --fwbw yet.")
    if args.load_checkpoint is None:
        raise ValueError("hil_serl eval requires --load_checkpoint.")

    env_request = _hil_serl_env_request(args, run_name="real_world_eval")
    env = _build_env(args, env_request)

    eval_args = dataclasses.replace(args, buffer_size=8, offline_dataset=None)
    agent = build_rlpd_hybrid(eval_args, env, None, logger=None, checkpoint_dir=None)
    agent.policy.eval()

    successes = 0
    returns: list[float] = []
    obs, _ = env.reset(seed=args.seed)
    for episode in range(args.eval_n_trajs):
        episode_return = 0.0
        while True:
            with torch.no_grad():
                action = agent.policy.predict(agent._obs_to_policy_device(obs), deterministic=True)
            env_action = action.to(ActorLoop.env_device(obs))
            obs, reward, terminated, truncated, info = env.step(env_action)
            episode_return += float(reward.sum())
            if bool(terminated) or bool(truncated):
                success = bool(terminated)
                successes += int(success)
                returns.append(episode_return)
                print(f"[eval] episode={episode} success={success} return={episode_return:.3f}", flush=True)
                obs, _ = env.reset(seed=args.seed)
                break

    success_rate = successes / args.eval_n_trajs
    mean_return = sum(returns) / len(returns) if returns else 0.0
    print(
        f"[eval] success_rate={success_rate:.3f} mean_return={mean_return:.3f} "
        f"n_trajs={args.eval_n_trajs}",
        flush=True,
    )


def run_hil_serl(args: "HilSerlArgs") -> None:
    if args.role == "actor":
        _run_actor(args)
    elif args.role == "learner":
        _run_learner(args)
    else:
        _run_eval(args)


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402
from typing import Literal, Optional, Sequence  # noqa: E402

from rl_garden.encoders.config import EncoderConfig  # noqa: E402
from rl_garden.training.online.rlpd_hybrid import RLPDHybridArgs  # noqa: E402
from rl_garden_real_world.training._args import (  # noqa: E402
    FrankaRealConfig,
    RealWorldFrankaArgs,
)
from rl_garden_real_world.training._registry import registry  # noqa: E402


@dataclass
class HilSerlArgs(RealWorldFrankaArgs, RLPDHybridArgs):
    franka_real: FrankaRealConfig = field(default_factory=FrankaRealConfig)

    # Overrides RealWorldFrankaArgs.role's Literal["actor", "learner"] with a
    # third "eval" role (item 10 of docs/hil-serl-roadmap.md) -- kept local to
    # HilSerlArgs rather than the shared RealWorldFrankaArgs so this round's
    # scope stays limited to hil_serl; serl.py is unaffected.
    role: Literal["actor", "learner", "eval"] = "actor"
    # Only used when role="eval": number of deterministic episodes to run.
    eval_n_trajs: int = 10

    classifier_checkpoint: Optional[str] = None
    classifier_threshold: float = 0.5
    classifier_image_keys: Sequence[str] = field(default_factory=tuple)

    # Gripper-flip reward shaping added to RLPDHybrid's discrete critic
    # target on top of the raw env reward (HIL-SERL's grasp_penalty) --
    # see rl_garden_real_world/envs/wrappers/grasp_penalty.py.
    use_grasp_penalty: bool = False
    grasp_penalty_value: float = -0.05

    # Records each actor-side episode to <record_video_dir>/episode_*.mp4
    # via ffmpeg -- actor role only (see _run_actor). None disables it.
    record_video_dir: Optional[str] = None

    # Replay-buffer image-frame dedup (MemoryEfficientDictReplayBuffer) --
    # only meaningful when obs.frame_stack > 1; a memory-usage optimization,
    # not a correctness change, for long image-heavy real-world sessions.
    memory_efficient_buffer: bool = False

    teleop_device: Literal["pico", "spacemouse"] = "pico"
    # Converts FrankaRealEnv's raw quaternion state slice to a rotation
    # vector (matching the action side's own rotvec convention). Off by
    # default -- changes "state"'s observation shape (20 -> 19), so it's an
    # explicit opt-in, not a silent default change.
    convert_obs_rotation: bool = False

    # Growing human-intervention demo buffer (DemoInterventionMixin), mixed
    # into training via RLPD's existing offline_replay_buffer/
    # offline_data_ratio slot -- see rl_garden/buffers/demo_intervention.py.
    # Not supported together with --offline_dataset (RLPD's static prior
    # dataset) in this round; both use the same slot.
    demo_buffer_size: int = 100_000
    demo_data_ratio: float = 0.5
    # Crash-recovery pkl snapshot cadence (received transitions between
    # snapshots), mirroring HIL-SERL's own buffer_period.
    buffer_period: int = 1000
    # Pre-collected demo transitions (HIL-SERL's --demo_path equivalent),
    # glob patterns over pkl files in the same transition-dict format as the
    # buffer snapshots above. Loaded into the demo buffer unconditionally on
    # every start, before crash-recovery snapshot reload.
    demo_dataset_paths: Sequence[str] = field(default_factory=tuple)

    fwbw: bool = False
    # Only used when fwbw=True. Actor: builds both directions' sync clients
    # (forward on sync_port, backward on sync_port_backward). Learner: which
    # direction this particular learner process owns.
    sync_port_backward: Optional[int] = None
    fwbw_direction: Literal["forward", "backward"] = "forward"

    # Below: HilSerlArgs-local overrides of RLPDHybridArgs/EncoderConfig
    # defaults, matching what HIL-SERL's own real-robot recipes
    # (serl_launcher/serl_launcher/utils/launcher.py) actually run --
    # scoped here only, not touched on RLPDHybridArgs/EncoderConfig
    # themselves, so sim rlpd_hybrid training and serl.py/SerlArgs are
    # unaffected. See docs/hil-serl-roadmap.md's "Behavioral Divergences"
    # section.
    backup_entropy: bool = False
    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            image_augmentation="random_shift",
            pooling_method="spatial_learned_embeddings",
        )
    )


registry.register("hil_serl", HilSerlArgs, run_hil_serl)
