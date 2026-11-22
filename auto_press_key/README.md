# auto_press_key

Isaac Sim / Isaac Lab keypad typing for the UBC Rover rover2026 arm. The arm types a full 6-digit code on a keypad
in simulation, using a temporary key-presser end effector.

The task environment is `Rover-Keypad-Type-v4`: a damped-least-squares IK controller does the reaching, and a small
PPO policy adds a residual on top of it.

The key layout and the reward ideas (gentle-press rule, progress shaping, digit bonus) come from Pranav's earlier
LearnFlake HRL work (`keyboard_stack_v2` layout and the keypad reward in the old UBC-Snowbots/LearnFlake repo, which
is kept there as is).

## Result

Policy `model_2499` (2500 iterations, 2048 envs, about 21 minutes on one RTX 5080), evaluated on 1000 episodes
(one random 6-digit code per episode, seed 123, `eval_v4.py`):

| | IK only | IK + RL residual |
|---|---|---|
| full 6-digit code typed | 22.0% | **99.2%** |
| digits per episode (mean) | 2.55 | 5.95 |
| episodes ended by wrong spot | 57.4% | 0.8% |
| episodes ended by overpress | 20.6% | 0.0% |
| median time per digit | 1.89 s | 0.71 s |
| press xy error, median / p90 | 0.1 / 2.7 mm | 0.9 / 1.8 mm |
| tip speed at press, median | 0.015 m/s | 0.027 m/s |

Peak joint acceleration is about at the limit (2.2 to 2.3 rad/s^2 against 2.0 on the big joints), which is PD
overshoot on top of the commanded limit. A short video of the 3/3 run can be recorded with `record_v4.py`.

## Read this first: joint limits are placeholders

The speed and acceleration limits the policy was trained with are **placeholders, not motor data**:

| joints | max speed | max acceleration |
|---|---|---|
| shoulder, link 1, elbow | 0.8 rad/s | 2.0 rad/s^2 |
| wrist (a4, a5, a6) | 1.5 rad/s | 4.0 rad/s^2 |

The rover2026 URDF has `effort=0 velocity=0` on every joint, so no torque or speed data exists in the CAD export.
Masses and inertias are CAD estimates. The 99.2% is a statement about this simulated arm with these limits, not
about the real arm.

**Next blocker:** real motor and gearbox datasheet limits (joint velocity, acceleration, torque). Put them into
`keypad_type_v4/env_cfg.py` (`MAX_JOINT_VEL`, `MAX_JOINT_ACC`, actuator effort and velocity) and the URDF, then
retrain. After that: actuator lag, backlash and encoder noise, domain randomisation (mass, friction, keypad pose,
key stiffness), real key contact (keys are visual only now; the table is a collider), and a keypad pose estimate
from vision, because the policy observes the keypad pose directly.

## How v4 works

(`keypad_type_v4/env_cfg.py` has the full design notes and all tunables.)

- One episode is one code. After each press the tip must lift 12 mm before the next digit counts. There is no reset
  between digits.
- Action = one IK step (Cartesian speed cap 0.25 m/s in free space, 0.04 m/s within 3 cm of the key) plus an RL
  residual (0.005 rad per step, scaled by the same cap). Joint targets are integrated from a speed- and
  acceleration-limited joint velocity, with an anti-windup band around the measured joint positions.
- A digit counts when the tip is within 6 mm in xy of the key centre, 2 to 5 mm below the key top, and moving
  slower than 0.10 m/s. Failures: wrong spot (over the keypad plate only), slam (moving down faster than 0.25 m/s
  near the key), overpress (more than 5 mm).
- Reward: potential-based progress shaping, digit +30, full code +50, failure -15, plus small time, action,
  acceleration and speed costs.
- Lessons from three failed runs: an unscaled residual swamped the braking IK and no press ever happened;
  all-negative shaping taught "press once, then overpress to end the episode"; all-positive shaping taught
  "hover forever"; the digit bonus must beat P(fail) times the failure penalty.

## Layout

```
auto_press_key/
  keypad_type_v4/     env_cfg.py (config only, safe to import before Kit starts), env.py, agents.py (PPO config)
  train_v4.py         wraps Isaac Lab's rsl_rl train.py and registers the task
  eval_v4.py          episodes -> JSON: code success, fail kinds, press quality, joint speed/accel vs limits
  play_v4.py          live GUI loop, one arm, target key highlighted
  record_v4.py        headless mp4 with overlay
  press_demo.py       kinematic IK typing demo in the Isaac Sim GUI (no RL, no physics)
  make_ee_urdf.py     builds assets/rover2026_ee.urdf (presser EE, unlocked a6) from assets/rover2026.urdf
  import_ee.py        URDF -> USD into assets/ee_usd/
  assets/             URDFs, meshes, presser-EE USD, keypad.usd
  docs/               arm spec sheet, evaluation methodology memo
  legacy/             v3 single-press environment (superseded, see below)
  models/             the shipped policy (model_2499.pt, 1.3 MB) and its training params
```

`assets/` holds what v4 needs at runtime (`ee_usd/rover2026_ee/`, `keypad.usd`) plus the sources to rebuild it
(`rover2026.urdf`, `rover2026_ee.urdf`, `meshes/`). The presser end effector is temporary: 4 mm plate, 6 mm by
60 mm rod and an 8 mm ball on the +x face of `a6_EE_holder`; the tip frame `ee_tip` is 72 mm out. `a6_rotation` is
unlocked (the CAD export wrote `lower=upper=0` and Isaac imported it locked). The keypad has 10 keys, 16 mm square,
4 mm travel, on a table whose top is at z = 0.288 m; key tops are at z = 0.300 m.

## Setup

Tested with:

- Isaac Sim 6.0.1 (pip install, Python 3.12, torch 2.11.0+cu130), NVIDIA driver 595.58.03, RTX 5080 16 GB
- Isaac Lab 3.0 (`v3.0.0-beta2.patch1`), installed editable from a checkout, with the `rsl_rl` extra
- `rsl-rl-lib` 5.0.1, `imageio` and `imageio-ffmpeg` (for `record_v4.py`)

Isaac Sim 5.x did not start on driver 595, which is why this project is on 6.0.1.

The trained policy is in git at `models/model_2499.pt` (1.3 MB). It is the only checkpoint kept in git; other
checkpoints and run logs stay out (see `CONTRIBUTING.md`).

All commands below run from `auto_press_key/` with the Isaac Python environment. Set
`OMNI_KIT_ACCEPT_EULA=YES` once per shell.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACLAB=~/IsaacLab                      # your Isaac Lab checkout (train_v4.py needs it)

# evaluate (default checkpoint: models/model_2499.pt)
python eval_v4.py --headless --num_envs 1000
python eval_v4.py --headless --num_envs 1000 --baseline          # IK only

# train (about 21 minutes for 2500 iterations on an RTX 5080); logs go to ./logs/rsl_rl/keypad_type_v4/
python train_v4.py --headless --num_envs 2048 --max_iterations 2500

# watch live (needs a display)
python play_v4.py [--checkpoint <model_*.pt>] [--baseline]

# record a video
python record_v4.py --headless --enable_cameras --codes 3 --out keypad_v4.mp4
```

Set `ROVER_ASSETS` to use a different asset directory. If the GPU is shared, run long jobs through the host's job
queue and stop any GUI playback first.

Rebuild the presser-EE USD after editing the end effector:

```bash
python make_ee_urdf.py       # assets/rover2026.urdf -> assets/rover2026_ee.urdf
python import_ee.py          # assets/rover2026_ee.urdf -> assets/ee_usd/rover2026_ee/
```

## Isaac Lab 3.0 gotchas

- `train.py` loads the task config before Kit starts, so the config module must not import runtime modules. That is
  why `env_cfg.py` is separate from `env.py`. Otherwise: `free(): invalid pointer`, or `pxr` imported too early.
- API renames: `.data.X.torch`, `*_index` setters, `body_link_jacobian_w`, `wp.to_torch(robot._ALL_INDICES)`.
- Eval, play and record need `handle_deprecated_rsl_rl_cfg(cfg, metadata.version("rsl-rl-lib"))` before building
  the `OnPolicyRunner`.
- The URDF importer writes links as nested prims, and visuals are same-named instanced children.

## legacy/

`legacy/` has the v3 single-press environment (`Rover-Keypad-Press-v3`, 93.0% single presses in the original run).
It is superseded: each episode was one press followed by a teleport back to the start pose, and the policy learned
0.18 s lunges. It is kept as a smaller reference for the IK-plus-residual setup. No v3 model is shipped.

## Docs

- `docs/rover2026_arm_specs.md`: kinematic chain, masses, reach, joint limits (derived from the URDF; `docs/arm_specs.py` regenerates it)
- `docs/RL_EVAL_METHODOLOGY_MEMO.md`: what the old success numbers measured, and the evaluation protocol proposed in response
