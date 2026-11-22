# SmokeDocs — 2026-11-06

Verdict: implementation complete; integration checks not run (verification left to the integration owner).

Added reusable offline CLI smoke covering completed, missing-model failed, and steady-progress SIGTERM interrupted runs; warmup/steady distribution recomputation, complete manifest/archive byte validation, and JSON/CSV aggregation. Optional CUDA and private Task Spooler paths use the selected interpreter and never the default queue. Temporary output is default; explicit output directory preserves a unique run group.

Added dependency-light packaging with optional PyTorch/CPU ONNX Runtime extras, deterministic/cpu_threads example controls, and README measurement/schema/limits/install/queue documentation. No scored model comparison or performance experiment was conducted, and no commits were made. Parent owns final integration/relocation and verification from the final location.

Checks for the integration owner: `python scripts/smoke_test.py`, optional `--tsp`, and explicit compatible-runtime `--device cuda --tsp`. ONNX/TensorRT/Jetson require separate real models/runtimes/hardware; no results claimed.
