# First pose candidate — 2026-11-07

**Verdict: desktop measurement collection passed; Jetson suitability remains unknown.** YOLO11n-pose is a pose estimator, not a beckoning classifier. FP16 was slower than FP32 in this TorchScript path. No training, held-out tuning, gesture accuracy scoring, or Jetson budget scoring occurred.

## Reproducible identity

- Official source: [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11/blob/8b8ac7d1fae7468f85dbf89670dd66f41f485aab/yolo11n-pose.pt), revision `8b8ac7d1fae7468f85dbf89670dd66f41f485aab`.
- Checkpoint SHA-256: `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`; checkpoint size: 6,255,593 bytes.
- Export: `models/yolo11n_pose_640/yolo11n-pose.torchscript`, 12,374,471 bytes, SHA-256 `8e165341bdec67df22603d0b85126114bb311d5693f212d5b1ebc8fbc07edaaa`.
- Python 3.12.3, PyTorch 2.14.1+cu130, Ultralytics 8.4.174; CPU export, static FP32 batch-one RGB `[1,3,640,640]`, non-frozen graph. Raw decoded output `[1,56,8400]`, 17 keypoints with three values each. AGPL-3.0 / Enterprise licensing needs review before redistribution or closed-source deployment.
- Serialized versus eager export-mode FP32 output agreed with maximum absolute error **0**, at rtol/atol 0.0001 on the declared deterministic input. Repeated preparation verified the existing export without overwriting it. This is not FP16 accuracy validation.
- **2,874,462 parameters**, excluding buffers; **7,466,984,000 estimated supported-operator FLOPs per forward**, using dispatch counting and two FLOPs per multiply-add. This is an operator estimate, potentially incomplete, not measured hardware work.

The original event-profiler estimate was only 63,772,001 FLOPs: it missed scripted convolutions while counting some elementwise work. Dispatch counting now takes priority; the convolution-plus-elementwise smoke catches this regression. Earlier CPU artifacts under `runs/pose-cpu-reference/cpu-c3b4a034` are superseded for profiling. Corrected results below use dispatch counting. Own queued job 154 was withdrawn before execution and replaced by 155 after this correction; the experiment's pass rule was unchanged.

## RTX experiment

Protocol was declared in `notes/POSE_PROTOCOL.md` before scoring. Six sequential runs: seeds 0/1/2, FP32/FP16, one intra-op CPU thread, deterministic mode, 50 warmup and 200 steady forwards each. Seeds vary synthetic input; they do not provide independent hardware samples or confidence intervals. Internal YOLO pose-head decoding is timed; decode/resize/normalization, NMS, application-level extraction, tracking and temporal classification are not. Synthetic pipeline stage timings are not a camera pipeline measurement.

Host: NVIDIA GeForce RTX 5080, driver 595.58.03, CUDA 13.0, AMD Ryzen 9 9950X, Linux x86_64. Runtime settings record one intra-op thread but 30 inter-op threads; this is not a one-thread process limit. TF32 and cuDNN benchmarking were disabled. FP16 casts the FP32 TorchScript artifact; it is not TensorRT or proof of uniform internal half precision. The cause of its slower latency was not established.

| Precision | Input seed | Mean GPU event latency (ms) | GPU event P95 (ms) | Peak GPU allocator bytes |
|---|---:|---:|---:|---:|
| FP32 | 0 | 2.994 | 3.427 | 70976000 |
| FP32 | 1 | 2.932 | 3.355 | 70976000 |
| FP32 | 2 | 2.871 | 2.965 | 70976000 |
| FP16 | 0 | 3.975 | 4.036 | 54065664 |
| FP16 | 1 | 4.031 | 4.258 | 54065664 |
| FP16 | 2 | 4.143 | 4.779 | 54065664 |

Allocator peaks cover PyTorch allocations after warmup, not total GPU memory or startup. Highest whole-process peak host RSS was 1,635,581,952 bytes for FP32 and 1,934,962,688 bytes for FP16; RSS includes initialization/profiling. These differently scoped figures must not be summed or used as Jetson unified-memory requirements. Per-run reports contain warmup/steady distributions, wall timing, throughput and telemetry.

Submitted as shared TSP job **155**, label `OrionPose640`, through `gpu-run`. Thoth's owner safely paused/checkpointed its work and held its own queued jobs; Orion did not kill or reorder another project's processes. Thoth confirmed automatic resumption on jobs 156/157 after Orion completed. The cooperative lock does not establish uncontended hardware; retain the recorded telemetry and contention limitations.

Evidence: `runs/pose-first/20261107T090432Z-d8a19bd8/`. `group_summary.json` says `measurements_collected`. All six completed with 200 finite steady GPU timings and 50 warmup samples, matching model provenance. Integrator independently checked every run manifest digest and archived file byte. CSV/JSON aggregation, per-run logs, telemetry, provenance and six archives remain in that group.

## Separate desktop CPU reference

Same artifact, FP32, same input dimensions and counts, one intra-op thread, AMD desktop CPU. These are not Jetson CPU estimates.

| Input seed | Mean synchronized host inference latency (ms) | Host inference P95 (ms) |
|---:|---:|---:|
| 0 | 86.721 | 88.287 |
| 1 | 85.495 | 87.931 |
| 2 | 85.203 | 87.333 |

Evidence: `runs/pose-cpu-reference/cpu-dispatch-3bb7840d/aggregate.csv` and `aggregate.json`.

## Functional verification and handoff

- Core private-TSP CPU smoke passed five scenarios, including TorchScript convolution profiling, manifest mismatch rejection, missing model and SIGTERM; artifacts at `runs/verification/smoke-__dej5_c`.
- Separate throwaway checks preserved three failed attempts and, on SIGTERM after a steady row, forwarded the signal, reaped the child, retained its interrupted archive and kept two unstarted attempts in aggregation. Control fixtures used CPU only; they are not GPU measurements.
- Invalid checkpoint bytes were rejected with a trusted-pin SHA-256 error and no successful export artifacts.
- A real-image CPU smoke on Ultralytics' bundled `bus.jpg` returned four pose instances with 17 finite keypoints each. Saved output was visually inspected at `runs/pose/runs/pose-image-smoke/bus.jpg`. This is qualitative execution evidence, not accuracy.
- Reproduce download/export/queue commands from README. Export and batch ownership notes are `PosePreparation.md` and `PoseBatch.md`; integration owner owns cross-file changes. All owners have rows in `notes/ledger.csv`. Its six CUDA lifecycle cases are the six completed candidate runs; scored experiments remain zero because no gesture accuracy or deployment budget was scored. No commits or publishing performed.

## Deployment gate still open

These results establish an executable desktop screening candidate, not that it fits the rover. No real Jetson execution, loaded-stack deadline/health measurement, tegrastats integration, TensorRT/ONNX export of this candidate, or gesture evaluation was performed. The team's “ultra” setting has not been identified by exact board, JetPack version and `nvpmodel -q --verbose` output. CPU ~80% and RAM ~50% are observations, not safe incremental budgets.

Before declaring suitability: measure the actual Orin NX 16 GB with its existing stack under normal and busiest conditions, record power/clocks, measure additional CPU work against existing deadlines, retain a measured RAM safety reserve, measure the inference GPU budget without regressions, and specify required gesture detection-onset latency. Then profile the complete camera-to-temporal-decision pipeline on that board. RTX latency, allocator bytes and desktop thread limits cannot substitute for those gates.
