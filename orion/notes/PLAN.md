# Orion development plan

Date: 2026-11-06

## Phase 1: Benchmark infrastructure — current delivery

Integration owner: Pranav. No commits or publishing without Pranav's request.

| Slice | Owner | Files |
|---|---|---|
| Backend timing, metadata, reproducibility | BackendHardening | orion_bench/backends.py |
| Environment, telemetry, availability | TelemetryHardening | orion_bench/telemetry.py |
| Reproducible smoke and usage | SmokeDocs | scripts/smoke_test.py, README.md, examples/, pyproject.toml |
| Lifecycle, aggregation, submission, relocation | integration owner | orion_bench/runner.py, orion_bench/aggregate.py, orion_bench/__main__.py, scripts/submit_tsp.sh, .gitignore |

Each owner writes a dated note and its own CSV ledger row in notes/. Siblings never modify the same path. Checks run after integration; long work uses the GPU/job queue, with blocking waits rather than job polling.

Predeclared functional acceptance:
- CPU smoke exits successfully and includes distinct warmup and steady samples.
- CUDA smoke, when the runtime is available, records GPU event timings and allocator peaks without CPU fallback.
- Model-load failures exit nonzero and preserve complete diagnostic artifacts.
- SIGTERM after steady samples yields an interrupted run and archive with consistent CSV/statistics.
- Every finalized archive contains the requested ten files; hashes verify.
- Aggregation handles completed, failed and interrupted runs in CSV/JSON.
- Private TSP smoke executes without altering the existing queue.
- No accuracy, isolation, Jetson speed, or model-suitability claims follow from smoke results.

Final repository: ~/projects/ubc-rover/Orion, alongside the existing RoverFlake2 and LearnFlake repositories. Existing repositories and their histories remain untouched. The parent workspace is now ubc-rover; ubc-rover-arm remains a compatibility symlink for existing live paths.

## Later priorities (not part of this implementation)

1. Profile representative pose models with fixed artifacts and explicit inference/pipeline boundaries.
2. Compare resource-constrained and shared-load desktop runs; never label them Jetson emulation.
3. Evaluate temporal beckoning candidates on labeled recordings; keep accuracy separate from synthetic compute measurements.
4. Export and validate compatible ONNX/TensorRT models on the actual Jetson Orin NX.

Before scoring any model experiment, specify the pass rule (latency/throughput, memory and accuracy requirements as applicable), run at least three seeds, and keep held-out data untouched during tuning. Report verdict first, with one kind of number per column. Actual Jetson runs use the same artifact and aggregation schema.
