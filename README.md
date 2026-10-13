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

Everything below drives a `real_robot` env (any `robot_infra` robot). `franka_real` (the HTTP
bridge) is always cartesian 7-D and is not covered here. Install this package
(`pip install -e real_world`) so that the `real_robot` backend is registered through the
`rlgarden.env_backends` entry point; without it `--env-backend real_robot` fails with
`Unknown env backend 'real_robot'`.

### Robot daemon

The robot daemon runs on the robot PC (see the robot_infra README,
[How to Start](../robot_infra/README.md#how-to-start)):

```bash
python -m robot_infra.server --config robot.yaml --bind 0.0.0.0:50051 --watchdog-s 5
```

The daemon idles a servoing arm that gets no action for `--watchdog-s` (default 0.5 s), and the
env enters servo mode at the end of `reset()`. If the first policy step after a reset can take
longer than that (first inference, a busy learner), raise `--watchdog-s`; otherwise the first
`step()` fails with `DeviceFault: the daemon watchdog stopped the arm ...`.

### Configuring the env

Cameras and image size come from the observation group, not from `--real-robot.*`:
`--obs.rgb wrist --obs.image-size 128 128` (a remote robot only reports its camera sizes after
connecting, so `--obs.image-size` is required with cameras). `--real-robot.cameras` is filled
in from `--obs.rgb`; setting it to something else raises `ObservationContractError`.
`--real-robot.joint-limits` is a list of `[lo, hi]` pairs, which the command line cannot
express (it flattens them); give it in a YAML preset passed with `--config`, which any
`train_real_world.py` run accepts (`--print-config` validates it without touching the robot):

```yaml
# hil.yaml
env_backend: real_robot
env_id: my_task
obs:
  rgb: [wrist]
  image_size: [64, 64]
real_robot:
  remote: ROBOT_PC:50051
  token: T
  num_joints: 7
  action_mode: joint_abs
  joint_limits: [[-2.8, 2.8], [-1.7, 1.7], [-2.8, 2.8], [-3.0, -0.1], [-2.8, 2.8], [0.0, 3.7], [-2.8, 2.8]]
  state_fields: [joint_pos, gripper_pos]
  control_hz: 10
  max_episode_steps: 100
```

### Action contract

`action_mode` fixes what a normalized action in [-1, 1] means; a trailing gripper dimension is
added when `gripper_in_action` is true (the default).

| `action_mode` | arm action | `joint_limits` | needs |
|---|---|---|---|
| `cartesian_delta` | 6-D twist: xyz scaled by `xyz_scale_m`, rotation by `rot_scale_rad` | optional | `tcp_pose` in the state |
| `joint_delta` | `num_joints` deltas scaled by `joint_scale_rad` | optional (clips) | `joint_pos` |
| `joint_abs` | `num_joints` values mapped linearly onto `joint_limits` | **required** (the mapping is built at construction) | `joint_pos`, and the robot must report `max_delta_per_step.joint_rad` |

hil_serl (`--env-backend real_robot`) also checks the teleop source against this contract when
the env is built (`TeleopInterventionWrapper`):

| teleop `action_mode` | gripper | support |
|---|---|---|
| `cartesian_delta` | yes | supported (spacemouse / pico / the robot's teleop part) |
| `joint_delta` / `joint_abs` | yes | supported through `--teleop-device robot` (the robot yaml's `teleop:` section) |
| any | no (`gripper_in_action=false`) | not supported (`RLPDHybrid` needs the gripper dim; no non-hybrid hil_serl yet) |

### Safety limits

The limits live in the robot yaml on the robot PC (`safety_box`, `joint_limits`,
`max_delta_per_step`) and are enforced by the daemon; the env can only be **tighter**:

- After connecting (at the first `reset()`), an unset `safety_box` / `joint_limits` /
  `home_joints` is inherited from the robot. An env limit looser than the robot's raises
  `DeviceFault` on that first reset; tighten the env value or loosen the yaml.
- A `joint_abs` env additionally requires the robot to report `max_delta_per_step.joint_rad`,
  because consecutive `joint_abs` actions can jump across the whole joint range. This holds for
  every entry point (hil_serl, deploy). A robot without it fails the first reset with
  `DeviceFault: action_mode joint_abs needs the robot's per-step joint limit ...`.
- `info["safety_clips"]` of `step()` is the cumulative number of servo commands the device
  clipped. It is refreshed every 10 steps (it costs a daemon round trip) and is `None` when the
  robot cannot report it. A rising count means the policy keeps asking for targets outside the
  limits. `reset()` homes with `move_to_joints`, so a `home_joints` outside the robot's limits
  raises `SafetyViolation` there.

### Teleop (`--teleop-device robot`)

hil_serl can take interventions from the teleop part of the robot itself (the robot yaml's
`teleop:` section: SpaceMouse or the `fake` backend; no leader-arm backend exists yet), served by the daemon:

- `cartesian_delta` actions pass through as they are; `joint_abs` is normalized onto
  `joint_limits`; `joint_delta` becomes `(q_leader - q_follower) / joint_scale_rad` (needs
  `joint_pos` in `state_fields`).
- A teleop gripper `g` in [0, 1] becomes `2g - 1`. A teleop without a gripper holds the gripper
  action last sent to the env (open, +1, right after a reset), so taking over the arm does not
  drop a grasped object.
- A teleop `episode_end` event ends the episode as `truncated` and sets
  `info["human_episode_end"]`; events pressed while the robot homes during `reset()` are dropped.
  A teleop read failure stops the actor (`DeviceFault`) instead of silently dropping the
  intervention.
- A remote robot only reports its teleop part once connected, so declare it with
  `--robot-teleop-action-mode {cartesian_delta,joint_abs,joint_delta}` (and
  `--no-robot-teleop-has-gripper` for a device without a gripper). The declaration is checked
  against the real part on connect and a mismatch raises `DeviceFault`.
- A configured teleop must be present for the robot to connect. For deployment use a robot yaml
  without the `teleop:` section.

Hardware-free try-out with the fake teleop (robot yaml; `on_end: loop` repeats the script so
episodes keep ending):

```yaml
teleop:
  name: leader
  backend: fake
  options:
    action_space: joint_abs
    num_joints: 7
    has_gripper: true
    on_end: loop
    script:
      - {action: [0, 0, 0, 0, 0, 0, 0], active: false, hold: 5}
      - {action: [0.1, 0, 0, -1.5, 0, 1.5, 0.7], gripper: 0.0, active: true, hold: 5}
      - {events: {episode_end: true}}
```

### Deploy

```bash
python examples/deploy_real_robot.py --checkpoint runs/.../checkpoints/checkpoint_<step>.pt \
    --algo rlpd_hybrid --real-robot.remote ROBOT_PC:50051 --real-robot.token T \
    --episodes 5 --max-steps-per-episode 100
```

(`--algo rlpd` for a serl checkpoint, `rlpd_hybrid` for hil_serl.) The training run's
`config.json` (next to `checkpoints/`) is the source of truth for the action/observation
contract:

- The deploy env takes `action_mode`, `gripper_in_action`, `num_joints`, `joint_limits`,
  `xyz_scale_m`, `rot_scale_rad`, `joint_scale_rad`, `state_fields`, `image_size`, `control_hz`
  and `max_episode_steps` from the run. It re-applies the run's observation wrappers by itself:
  `obs.frame_stack` (stacking the run's `classifier_image_keys`) and `convert_obs_rotation`.
  `--real-robot.*` only supplies connection fields (`robot`, `remote`, `token`, `home_joints`,
  `safety_box`, `reset_settle_s`, ...).
- A contract field given on the CLI that differs from the run's is an error (exit with a
  `ValueError`: `--real_robot.* contradicts the training run's real_robot config ...`).
  **Limitation:** a CLI value equal to the dataclass default cannot be told apart from an unset
  one, so `--real-robot.action-mode cartesian_delta` against a `joint_abs` run silently uses
  the run's `joint_abs`.
- Runs without a `real_robot` config (older runs, `franka_real`) fall back to the CLI with a
  warning, and the contract is then not verified.
- A `DeviceError` (daemon unreachable, watchdog, safety) prints its message and hint and exits
  with code 2.
- The daemon's `--jpeg-quality` is not part of the run's config: start the deploy daemon with
  the same value (or none) the training daemon used.

### Demo and snapshot contract sidecar

The hil_serl learner's buffer and demo-buffer snapshots get a `<name>.contract.json` next to
each pkl (`transitions_N.pkl` -> `transitions_N.contract.json`); `--demo-dataset-paths` pkls
are read together with their sidecar. One line of sorted JSON:

```json
{"action_mode": "joint_abs", "arm_action_dim": 7, "gripper_in_action": true, "joint_limits": [[-2.8, 2.8], ...], "joint_scale_rad": 0.05, "num_joints": 7, "rot_scale_rad": 0.1, "state_fields": ["joint_pos", "gripper_pos"], "xyz_scale_m": 0.02}
```

On load the sidecar is compared with the env's contract (floats with `allclose`). A mismatch
raises `ValueError: ... recorded action contract does not match the env's (differs in
['joint_limits']) ...`. A sidecar written before the scales and limits existed is checked on the
keys it has, with a warning. A pkl without a sidecar is checked for the action width only, with
a warning (`... has no action-contract metadata (<name>.contract.json missing) ...`).

### End-to-end try-out with the fake robot

One machine, no hardware (verified in the project container; the training runs are short and
the policy is untrained). Robot PC side, with a fake-backend `robot.yaml` of type `franka`
(`arm_backend: fake`, `gripper: {backend: fake}`, a fake camera named `wrist`, joint limits and
`max_delta_per_step: {xyz_m: 0.05, joint_rad: 0.1}` as in the robot_infra example yaml, and the
fake teleop above):

```bash
python -m robot_infra.server --config robot.yaml --bind 127.0.0.1:50061 --token T \
    --jpeg-quality 90 --watchdog-s 5 --no-preflight
```

Learner and actor (two terminals; `hil.yaml` as above with `remote: 127.0.0.1:50061`):

```bash
COMMON="--config hil.yaml --teleop-device robot --robot-teleop-action-mode joint_abs \
    --log-dir runs --log-type none --exp-name smoke --batch-size 16 --learning-starts 32 \
    --buffer-size 2000 --checkpoint-freq 10 --sync-host 127.0.0.1 --sync-port 6055 --publish-freq 10"
python examples/train_real_world.py hil_serl --role learner $COMMON
python examples/train_real_world.py hil_serl --role actor $COMMON
```

Stop both with Ctrl-C. The learner writes `runs/smoke/config.json` and
`runs/smoke/checkpoints/checkpoint_<update>.pt`. Then deploy the last checkpoint:

```bash
python examples/deploy_real_robot.py --checkpoint runs/smoke/checkpoints/checkpoint_<update>.pt \
    --algo rlpd_hybrid --real-robot.remote 127.0.0.1:50061 --real-robot.token T \
    --episodes 2 --max-steps-per-episode 15
```

which prints one `[deploy] episode=... steps=15 ... health_ok=True` line per episode and a
`[deploy] summary ...` line. On real hardware the same commands apply with the daemon on the
robot PC, `--bind 0.0.0.0:50051`, the real `ROBOT_PC:50051`, `--sync-host` set to the learner's
reachable address, and a real `control_hz`.

Two-process real-robot training for SERL (state-only or image) uses the same flags:

```bash
python examples/train_real_world.py serl --role learner ...
python examples/train_real_world.py serl --role actor --env-backend real_robot \
    --real-robot.remote ROBOT_PC:50051 --real-robot.num-joints 7 \
    --obs.rgb wrist --obs.image-size 128 128 ...
```

(the `serl` real-robot flags were only checked with `--print-config`; no `serl` run was done.)

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
