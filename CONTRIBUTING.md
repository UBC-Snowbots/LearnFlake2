# Contributing

## What goes in git

- Source, configs, docs and small assets only. **No file over 5 MB.**
- **No datasets, no recordings, no run logs.** Put them in a GitHub Release (or the team drive) and link them from
  the relevant README.
- **No checkpoints**, with one exception: the single shipped keypad policy, `auto_press_key/models/model_2499.pt`
  (1.3 MB). A new shipped model replaces it in the same change; do not add model files next to it.
- Generated and local output (`__pycache__/`, `logs/`, `runs/`, `dist/`, `*.mp4`) is ignored. Keep it that way.

`scripts/check_hygiene.sh` enforces the size limit and the commit-message rules below. CI runs it on every push and
pull request.

## Commits

- One short subject line in the form `area: what changed` (for example `auto_press_key: scale residual by IK speed cap`),
  then an optional body that says why.
- Do not add `Co-authored-by:` lines or any `Generated with ...` line. The commit-msg hook strips them and CI fails
  on them.
- Enable the hook once per clone:

  ```bash
  git config core.hooksPath scripts/hooks
  ```

## Pull requests

- Branch from `main`, keep the change focused, and describe how you tested it (command and result).
- PRs are **squash-merged**: the PR title becomes the commit subject, so write it like one.
- Numbers in docs need their setup next to them (episodes, seed, checkpoint). Mark anything you did not run as
  unverified.

## Code and docs style

- Python 3.12, standard library first, no new dependency without a reason in the PR.
- Docs: plain English, short sentences, no em dashes, times in PT.
- Isaac scripts must work from a fresh clone: paths come from the file's own location or an environment variable, not
  from a home directory.

## Layout

- `auto_press_key/`: Isaac Sim keypad typing. Start with its README.
- `orion/`: gesture recognition. Its history was imported from the Orion repository; start with its README.
