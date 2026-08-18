# rlgarden-real-world

Real-world RL orchestration split out of
[rl-garden](https://github.com/JaimeParker/rl-garden): `ActorLoop`/
`LearnerLoop`/sync (SERL and HIL-SERL variants), the `franka_real` env
backend, teleop intervention wrapper, and the reward-classifier
data-collection tool.

This package imports `rl_garden` as a normal dependency (algorithms,
buffers, policies, the `EnvBackend`/`BasePolicy` contracts) and registers
`franka_real` through rl-garden's `rlgarden.env_backends` entry-point group
— no file inside the `rl_garden` package itself needs editing to add it.
Hardware/communication code lives in the sibling
[rlgarden-robot-infra](https://github.com/JaimeParker/rlgarden-robot-infra)
repo (extra: `robot-infra`); `franka_real`'s own coupling to it is HTTP/JSON
only (`FrankaBridgeClient`), no shared Python types.

**Critical invariant**: RLinf's real-world RL training goes through the
identical `EnvWorker`/`EmbodiedSACFSDPPolicy` path as simulation (verified
against RLinf source directly, see rl-garden's
`docs/design/rlinf-integration.md`). As long as `franka_real` keeps
satisfying rl-garden's `EnvBackend` api_version 2 contract exactly as it
does today, rl-garden's existing RLinf Pattern-A adapter
(`rl_garden/integrations/rlinf/{sac_actor,sac_rollout}.py`) can drive this
env with zero real-robot-specific RLinf code. Don't break that contract
when extending this package.

Run training with `python examples/train_real_world.py serl --role ...`
(see that file's module docstring for the two-process actor/learner launch
shape).
