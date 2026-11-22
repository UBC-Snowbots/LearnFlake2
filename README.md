# LearnFlake2

Learning-based autonomy work for UBC Rover. Two parts:

| Folder | What | Status |
|---|---|---|
| [`auto_press_key/`](auto_press_key/README.md) | Isaac Sim / Isaac Lab keypad typing with the rover2026 arm: IK plus a small RL residual types full 6-digit codes. | 99.2% full codes in simulation (IK alone: 22.0%). Joint limits are placeholders until the motor datasheets are in. |
| [`orion/`](orion/README.md) | Vision for the URC Autonomy mission: recognise the astronaut's Follow! gesture from a camera on a Jetson Orin NX (pose model plus a temporal skeleton classifier), with a benchmark harness. | Pipeline built and smoke-tested on public clips. No team gesture recordings yet; Jetson unverified. |

This repository is the continuation of the Isaac Sim typing work. The earlier MuJoCo/RoboSuite typing work stays in
UBC-Snowbots/LearnFlake, which is kept as is and no longer developed. The key layout and reward ideas in
`auto_press_key/` come from Pranav's HRL work there.

## Quick start

**Keypad typing** (needs Isaac Sim 6.0.1 and Isaac Lab 3.0; see `auto_press_key/README.md` for setup):

```bash
cd auto_press_key
export OMNI_KIT_ACCEPT_EULA=YES
python eval_v4.py --headless --num_envs 1000     # uses the shipped models/model_2499.pt
python eval_v4.py --headless --num_envs 1000 --baseline   # IK only, for comparison
```

**Orion** (CPU smoke of the benchmark harness; needs PyTorch):

```bash
cd orion
python scripts/smoke_test.py
```

See `orion/README.md` for the gesture workflow, the live camera demo and the Follow! protocol.

## Repository rules

- No checkpoints or datasets in git, except the single shipped keypad model, and nothing over 5 MB. See
  [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Enable the commit hook once per clone: `git config core.hooksPath scripts/hooks`.
- `scripts/check_hygiene.sh` runs in CI (`.github/workflows/repo_hygiene.yaml`).
