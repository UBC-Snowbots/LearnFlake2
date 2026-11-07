# Integration handoff — 2026-11-06

Verdict: PASS for Phase 1 functional acceptance. No model-suitability or Jetson performance verdict.

Canonical repository: `~/projects/ubc-rover/Orion` (renamed after the checks below). Separate local Git repository initialized on `main`; no commits or remote publishing. Temporary source copy removed. Existing `RoverFlake2/` and `LearnFlake/` are siblings and were not modified. Historical benchmark artifacts retain their original absolute paths and checksums; rename verification is recorded in `notes/ORION_RENAME.md` and `notes/WORKSPACE_RENAME.md`.

## Executed acceptance

| Workflow | Lifecycle cases passed | Evidence directory |
|---|---:|---|
| Direct CPU smoke | 4 | runs/verification/smoke-hf27hz9c |
| Private-TSP CPU smoke | 4 | runs/verification/smoke-typq5z8r |
| Private-TSP CUDA smoke | 4 | runs/verification/smoke-6fmx4p2t |

Each suite exercised built-in inference, valid TorchScript convolution, missing-model failure, and SIGTERM after observed steady progress. It independently checked warmup/steady distributions, sample counts, throughput arithmetic, all required files, every manifest digest, archived bytes, and CSV/JSON aggregation. Private smoke queues were stopped by their owners; the existing shared queue was not changed.

Additional executed checks, with outputs under `runs/integration/`:
- ONNX Runtime CPU inference using its installed `datasets/mul_1.onnx` fixture, with two warmup and five steady invocations.
- Impossible free-VRAM threshold with zero wait: failed before model initialization, retained a failure archive; five invalid availability configurations rejected with exit 2 and no run directory.
- CUDA UUID visibility mapping matched the recorded physical GPU; CUDA-event durations and allocator peaks were present.
- CUDA FP16 inference: recorded float16 input and GPU timing; environment includes harness version and matching source-file SHA-256.
- Actual `scripts/submit_tsp.sh` launched from `/tmp`, with a configuration filename containing spaces, absolute interpreter and optional `env` wrapper; private queue completed four steady invocations. The site's `gpu-run` implementation was inspected, not used for this CPU check.
- Valid CPU TorchScript convolution ran 2,000 steady invocations with numeric steady CPU/RSS telemetry and no warnings. Parameter count, model hash and analytic convolution FLOP estimate verified.
- Editable installation succeeded in a disposable system-site-packages virtual environment using `pip install --no-deps --no-build-isolation -e .`; installed module help ran successfully.

## Profiling correction

A real TorchScript convolution executed successfully but the event profiler supplied no supported FLOP count. A separate dispatch-counter probe produced the analytic convolution count. Added the dispatch fallback before warmup, clearly labeled as an estimate, and kept a convolution regression in the reusable smoke. Re-ran both CPU and CUDA queued suites after this correction; both passed. FLOPs may still undercount unsupported operators and are never inferred from file size.

## Runtime and limits

Observed host: shared RTX 5080. Default Python: PyTorch 2.13.0+cpu and ONNX Runtime 1.30.0. Existing CUDA environment: `../envs/gesture/bin/python`, PyTorch 2.14.1+cu130. No packages were changed in that environment. Recent CUDA PyTorch emitted TorchScript deprecation warnings during smoke fixture export; warnings were not suppressed.

Unverified: TensorRT engine execution, ONNX Runtime CUDA, multi-GPU/MIG behavior, real Jetson hardware, actual camera preprocessing/postprocessing, gesture accuracy. No labeled dataset or competition latency/accuracy acceptance thresholds were provided. Functional smokes use one fixed seed; they are not scored experiments. Future model experiments require a predeclared pass rule and at least three seeds, with no held-out tuning.

Source installation and example commands are in README.md. Run artifacts remain in ignored `runs/`; a new smoke can regenerate them without downloading weights. Individual run `.tar.gz` files contain their complete reports and manifests.
