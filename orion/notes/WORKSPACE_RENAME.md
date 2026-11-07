# Workspace rename — 2026-11-06

Verdict: workspace renamed to `~/projects/ubc-rover` following the user's renewed instruction to proceed.

The real workspace contains `LearnFlake/`, `RoverFlake2/`, and `Orion/` alongside the existing rover projects and environments. Moved the whole parent directory; all repository `.git` directories remain present. No repositories were copied, committed or published.

`~/projects/ubc-rover-arm` is now a compatibility symlink to the real `ubc-rover` directory. Existing live rover processes, ROS install state, scripts and virtual environments contain absolute old-prefix references. The symlink preserves their access without stopping or rewriting active workloads. It is an intentional exception to a clean path cutover, not a second workspace. Do not remove it until these callers are migrated during a stopped-stack maintenance window.

Updated the existing `~/LearnFlake` symlink to the canonical `~/projects/ubc-rover/LearnFlake` target. Current README, plan and integration note use the new canonical location. Earlier run artifacts and dated decision records retain historical paths and checksums.

Executed verification from the new directory:
- Queued CPU smoke passed all four lifecycle cases, artifact hashes/archive bytes and CSV/JSON aggregation. Evidence: `runs/verification/smoke-fkf8rxqm`.
- Confirmed LearnFlake's canonical alias resolves and all three repository `.git` directories remain present.

No live rover process was stopped. No Jetson, gesture accuracy or new GPU performance result is claimed. Orion's scope and earlier backend verification remain documented in README.md and notes/INTEGRATION.md.

Additional verification: the old and new workspace paths identify the same directory, and the relocated `envs/gesture/bin/python` imports PyTorch 2.14.1+cu130 with CUDA available.
