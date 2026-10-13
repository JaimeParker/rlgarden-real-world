"""Real-world BC: offline imitation over pre-collected demo pkls.

Structurally unlike ``serl``/``hil_serl``: BC (``rl_garden.algorithms.bc.BC``)
is an ``OfflineRLAlgorithm`` -- a static dataset, no online env stepping --
not an ``OffPolicyAlgorithm``, so there is no actor/learner HTTP split here
at all: no ``sync.py``, no ``LearnerLoop``/``ActorLoop``. This reuses
``rl_garden``'s existing offline BC algorithm construction
(``rl_garden.training.offline.bc.build_bc``) directly against a
demo-pkl-populated buffer, rather than ``rl_garden.training.offline._runner
.run_offline`` (which expects rl-garden's own ``--offline_dataset`` h5/minari
format, not the real-world pkl-list-of-transition-dicts convention demo data
already uses -- see ``rl_garden_real_world.training._demo_loading``).
"""
from __future__ import annotations

import os
import time

import torch


def run_real_world_bc(args: "RealWorldBCArgs") -> None:
    from rl_garden.common import Logger, seed_everything
    from rl_garden.common.resolved_config import persist_resolved_config
    from rl_garden.envs.backend_registry import make_training_envs
    from rl_garden.training.offline._runner import _eval_env_request
    from rl_garden.training.offline.bc import build_bc
    from rl_garden_real_world.action_contract import demo_contract_of
    from rl_garden_real_world.training._demo_loading import load_demo_pkls

    seed_everything(args.seed)

    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.exp_name or f"{args.env_id}__real_world_bc__{args.seed}__{int(time.time())}"
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.log_dir, run_name, "checkpoints")

    resolved_config = persist_resolved_config(
        args, training_phase="real_world", algorithm="real_world_bc", run_name=run_name, log_dir=args.log_dir
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

    # Env is used only for its observation/action-space shape -- BC never
    # calls reset()/step() on it, and FrankaRealEnv makes no network call
    # until one of those is (mirrors serl.py's/hil_serl.py's learner-side
    # env construction). Reuses the offline-args-shaped request builder
    # (BCArgs has no online-training num_envs; _eval_env_request resolves
    # the offline spec_num_envs/num_eval_steps itself).
    env_request = _eval_env_request(args)
    env, _ = make_training_envs(args.env_backend, env_request)

    agent = build_bc(args, env, logger, eval_env=None)
    loaded = load_demo_pkls(
        args.demo_dataset_paths,
        agent.replay_buffer.add,
        agent.buffer_device,
        expected_contract=demo_contract_of(env),
    )
    if loaded == 0:
        raise ValueError(
            f"No demo transitions loaded from demo_dataset_paths={args.demo_dataset_paths!r}."
        )

    try:
        for step in range(1, args.num_offline_steps + 1):
            metrics = agent.train(1, compute_info=(step % args.log_freq == 0))
            if step % args.log_freq == 0:
                logger.log(metrics, step)
            if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0:
                agent.save(os.path.join(checkpoint_dir, f"checkpoint_{step}.pt"))
        if args.save_final_checkpoint:
            agent.save(os.path.join(checkpoint_dir, "final.pt"))
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# Args + registration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402
from typing import Sequence  # noqa: E402

from rl_garden.training.offline.bc import BCArgs  # noqa: E402
from rl_garden_real_world.training._args import (  # noqa: E402
    FrankaRealConfig,
    RealWorldFrankaArgs,
)
from rl_garden_real_world.training._registry import registry  # noqa: E402


@dataclass
class RealWorldBCArgs(RealWorldFrankaArgs, BCArgs):
    """RealWorldFrankaArgs is composed for its env-backend fields
    (franka_real, env_backend) only -- role/sync_host/sync_port/control_hz/
    deterministic_actor/train_freq/publish_freq are inherited but unused:
    BC has no actor/learner split to apply them to."""

    franka_real: FrankaRealConfig = field(default_factory=FrankaRealConfig)
    demo_dataset_paths: Sequence[str] = field(default_factory=tuple)
    num_offline_steps: int = 100_000


registry.register("real_world_bc", RealWorldBCArgs, run_real_world_bc)
