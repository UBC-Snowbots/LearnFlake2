# Orion rename — 2026-11-06

Historical decision record. The later workspace rename is complete; see `notes/WORKSPACE_RENAME.md` for the current `~/projects/ubc-rover/Orion` location and compatibility-link policy.

Verdict: PASS for the Orion rename; parent workspace relocation explicitly deferred by the user.

Current layout:
- `~/projects/ubc-rover-arm/Orion/`: independent benchmark Git repository, formerly `rover-cv-benchmark/`.
- `~/projects/ubc-rover-arm/RoverFlake2/`: existing RoverFlake checkout, already in the workspace.
- `~/projects/ubc-rover-arm/LearnFlake/`: existing LearnFlake checkout, already in the workspace.
- `~/LearnFlake`: pre-existing link to the workspace checkout; unchanged and resolves successfully.

Moved the entire benchmark repository, including `.git` and run artifacts. Python package is now `orion_bench`; distribution name is `orion-rover-benchmark`. CLI, queue launcher, smoke, install metadata, current plan and README use the new name. No old benchmark-directory or Python-module alias remains. Historical run configs, logs, archives and dated notes retain original paths/names as evidence; their hashes were not rewritten. Historical references to `rover_bench/` describe today's `orion_bench/` sources. Source archive is now `dist/Orion-0.1.0.tar.gz`.

## Executed verification

- `python3 scripts/smoke_test.py --tsp --output-dir runs/verification`: passed built-in and TorchScript inference, deliberate model-load failure, SIGTERM, manifests/archive bytes, and CSV/JSON aggregation. Evidence: `runs/verification/smoke-xuuszl94`.
- Installed the renamed distribution in a disposable system-site-packages virtual environment. Imported `orion_bench` from outside the repository and verified the distribution identity.
- Actual `scripts/submit_tsp.sh`, launched outside the repository with a spaced config path and that installed environment, completed on its own private Task Spooler socket. Its server was stopped; the shared queue was untouched.
- Confirmed all three repository `.git` directories remain present, LearnFlake's existing alias resolves, and `envs/gesture/bin/python` still imports CUDA PyTorch successfully.

## Deferred workspace rename

Observed live rover-stack, recorder and ROS bag processes referencing `ubc-rover-arm`; scripts, generated ROS installs and environments contain absolute paths. Offered a compatibility link versus deferral; the user selected **Defer workspace rename**. Therefore the real parent remains `ubc-rover-arm`, no `ubc-rover` directory or old-path compatibility link was created, and no rover processes were stopped.

Next relocation requires a stopped rover stack and a coordinated migration of absolute paths, generated ROS install/build state, environment entry points/editable-package paths, Git worktree pointers and external aliases. Do not simply rename the live parent and assume these remain usable.
