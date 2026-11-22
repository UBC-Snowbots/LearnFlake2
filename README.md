# LearnFlake2

Home for the learning, simulation and autonomy sub-projects of our part of the UBC Rover team. Each sub-project lives
in its own top-level folder with its own README, setup and status. Projects get added, merged and retired as the team's
work changes, so the list below is the current state, not a fixed plan.

## Sub-projects

| Folder | What | Status (updated with each change) |
|---|---|---|
| [`auto_press_key/`](auto_press_key/README.md) | Isaac Sim / Isaac Lab keypad typing with the rover2026 arm: IK plus a small RL residual types full 6-digit codes. | 99.2% full codes in simulation over 1000 codes (IK alone: 22.0%); re-checked from this repo at 99.0% over 100 codes. Joint limits are placeholders until the motor datasheets are in. |
| [`orion/`](orion/README.md) | Vision for the URC Autonomy mission: recognise the astronaut's Follow! gesture with a pose model plus a temporal skeleton classifier, on a Jetson Orin NX. Includes the benchmark harness and the post-detection Follow! protocol. | Live on the Jetson at about 21 fps with a webcam stream. The classifier is a stand-in trained on public hand-waving clips; no team gesture recordings yet. |

Earlier work: the MuJoCo/RoboSuite typing project stays in
[UBC-Snowbots/LearnFlake](https://github.com/UBC-Snowbots/LearnFlake), kept as is. The key layout and reward ideas in
`auto_press_key/` come from Pranav's HRL work there.

## Quick start

Every sub-project is self-contained: `cd` into its folder and follow its README. Two examples:

```bash
# keypad typing (needs Isaac Sim 6.0.1 and Isaac Lab 3.0; setup in auto_press_key/README.md)
cd auto_press_key
export OMNI_KIT_ACCEPT_EULA=YES
python eval_v4.py --headless --num_envs 1000              # shipped model, models/model_2499.pt
python eval_v4.py --headless --num_envs 1000 --baseline   # IK only, for comparison

# Orion (CPU smoke of the harness; needs PyTorch)
cd orion
python scripts/smoke_test.py
```

## Adding a sub-project

1. Make a top-level folder with a short `snake_case` name (`auto_press_key`, `orion`).
2. Give it a README that starts with what it does, how to set it up and how to run it, and a status line with real
   numbers and the setup they came from.
3. Keep its dependencies and environment inside the folder. Sub-projects do not import each other; if two need the
   same code, discuss it in a PR first.
4. Add a row to the table above in the same PR. When a project is retired, move its row to a "Retired" note with the
   last commit that has it, then delete the folder.

## Repository rules

- Nothing over 5 MB. No datasets, recordings or run logs. No checkpoints, except at most one small shipped model per
  sub-project, named in that sub-project's README. Details in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Commit subjects start with the sub-project folder: `orion: ...`, `auto_press_key: ...`.
- Enable the commit hook once per clone: `git config core.hooksPath scripts/hooks`.
- `scripts/check_hygiene.sh` runs in CI on every push and pull request (`.github/workflows/repo_hygiene.yaml`).
