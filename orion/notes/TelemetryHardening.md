# TelemetryHardening — 2026-11-06

Verdict: implemented; integration checks not run by its author.

- Changed only `rover_bench/telemetry.py` plus these requested notes.
- Added GPU-free availability validation, verified CUDA runtime PCI-to-UUID mapping, and explicit ambiguity/MIG errors instead of ordinal guesses.
- Preserved CPU/shared-GPU samples and other-process visibility; added CPU/affinity/RAM/thread/runtime/toolkit/Jetson metadata, corrected guest CPU double counting, and surfaced sampler failures through `errors` and CSV error fields.
- Added nanosecond UTC/monotonic timestamps, independent collection durations, `phase`/`phase_end`, verified GPU UUID, and process RSS high watermark. Memory fields retain `_mb` names but use MiB. Procfs VmHWM is process-lifetime, refreshed at stop; rows crossing phases must not be treated as pure-phase measurements.
- No builds, tests, lint, formatting, smoke runs, GPU workload changes, package changes, or commits. No experiments performed.
- Integration checks: malformed availability rejection without GPU calls; CPU sampling/error propagation; CUDA-visible-device identity on the shared GPU; phase-boundary aggregation and CSV/JSON serialization.
