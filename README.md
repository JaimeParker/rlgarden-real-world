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

## Running on hardware

Prerequisite: the robot daemon runs on the robot PC (see the robot_infra
README, [How to Start](../robot_infra/README.md#how-to-start)):

```bash
python -m robot_infra.server --config robot.yaml --bind 0.0.0.0:50051
```

Train (two processes, GPU machine and robot-side machine):

```bash
python examples/train_real_world.py serl --role learner ...
python examples/train_real_world.py serl --role actor --env-backend real_robot \
    --real-robot.remote ROBOT_PC:50051 --real-robot.num-joints 7 \
    --real-robot.cameras wrist --real-robot.image-size 128 128 ...
```

Deploy a trained checkpoint (eval only):

```bash
python examples/deploy_real_robot.py --checkpoint runs/.../checkpoints/checkpoint_<step>.pt \
    --algo rlpd --real-robot.remote ROBOT_PC:50051 --real-robot.num-joints 7 ...
```

RLinf: start from
[`configs/rlinf/realworld_rlgarden_franka.yaml`](configs/rlinf/realworld_rlgarden_franka.yaml).
The package must be importable in the Ray env-worker process, and RLinf kills
any roscore running on the env node.

**RLinf**: RLinf drives this package through Gymnasium registration, not
through rl-garden's adapter. In an RLinf real-world config set the env block's
`env_type: real` and `init_params.id:
"rl_garden_real_world.rlinf.register:RLGardenReal-v0"`; RLinf's
`RealWorldEnv` then `gym.make`s `RealRobotEnv` (one robot per env worker) with
`override_cfg` as its config. See
[`configs/rlinf/realworld_rlgarden_franka.yaml`](configs/rlinf/realworld_rlgarden_franka.yaml)
for the env block and its hazards, and `tests/test_rlinf_registration.py` for
the contract. rl-garden's Pattern-A RLinf adapter
(`rl_garden/integrations/rlinf/{sac_actor,sac_rollout}.py`) runs RLinf's own
env worker and has no real-robot or image path, so it cannot drive this
package.

Run training with `python examples/train_real_world.py serl --role ...`
(see that file's module docstring for the two-process actor/learner launch
shape). See [`docs/teleop.md`](docs/teleop.md) for the teleop demo/recording
workflow, and [`docs/hil-serl-roadmap.md`](docs/hil-serl-roadmap.md) for what
HIL-SERL capability was and wasn't migrated, and why.
