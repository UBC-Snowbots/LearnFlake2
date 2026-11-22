# PoseBatch — 2026-11-07

Implemented `scripts/run_pose_candidate.py` and `examples/yolo11n_pose_gpu.json`.

- Default group: sequential FP32/FP16 × distinct seeds 0/1/2, 50 warmup and 200 steady calls, synthetic batch [1,3,640,640]. Uses the current interpreter, never submits jobs or changes queue settings.
- `load_config` resolves the original model/provenance paths before absolute-path per-run configs are written. Each invocation creates a unique group, console logs, effective configs, complete CSV/JSON aggregates including failures/not-started attempts, and `group_summary.json` with manifest link and explicit desktop-only/no-accuracy/no-Jetson limits.
- SIGTERM/SIGINT are forwarded to the active benchmarking child, which is waited for before aggregation; remaining planned attempts remain represented. Interrupted/incomplete groups exit nonzero.
- Collection verdict additionally requires requested completed counts, finite nonnegative steady GPU-event timings, model hash agreement with export manifest, valid artifact hashes, and matching archive contents. This is not an accuracy or deployment verdict.

No benchmarks, tests, builds, exports, inference, formatters, or queue operations run by its author. The integration owner should exercise six-run collection plus failure/interruption handling and inspect standard artifacts/aggregate coverage after provenance support lands.
