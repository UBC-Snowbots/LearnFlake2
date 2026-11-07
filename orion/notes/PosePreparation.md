# PosePreparation — 2026-11-07

Implemented `orion_bench.prepare_pose`: offline local checkpoint hashing before pickle load; exact pin `ultralytics_version`; validated pose task, single class and 17x3 keypoints; CPU-only unfrozen/unoptimized TorchScript tracing; deterministic numeric comparison after serialization; hash/settings-checked repeat; exclusive lock and atomic non-overwriting artifact/manifest publication.

Supported input is FP32 RGB NCHW `[1,3,640,640]`; expected raw decoded output `[1,56,8400]`. No NMS, tracking, gesture classification, accuracy assessment, training, or package/checkpoint downloads. Numeric comparison uses arange modulo 251 divided by 250 with rtol=atol=0.0001; actual error and parameter count are generated only on execution.

Assumptions: provenance pin is trusted (checkpoint is pickle); installed pinned Ultralytics supplies its local checkpoint dependencies; output filesystem supports hard links. Source pin schema is `source` plus top-level `ultralytics_version`; source license and extra provenance fields are retained verbatim. Shape is a supported contract, not a graph-level input guard. Runtime FP16 casting is left to the benchmark; no freeze/mobile optimization is applied.

No exports, inference, tests, formatters, or benchmarks executed by its author. The integration owner should execute the supplied CLI, repeat it to check idempotence, verify hash/version refusal, and exercise runtime FP16 casting in integration. No output-directory files were created by this worker.
