# First pose candidate protocol — 2026-11-07

Verdict before measurement: candidate unmeasured; Jetson viability unknown.

Candidate: pretrained Ultralytics YOLO11n-pose, official Hugging Face revision and checkpoint digest in examples/yolo11n_pose_pin.json. No training or threshold tuning. Review AGPL-3.0 / Enterprise licensing before redistributing or integrating into a closed-source deployment.

## Fixed experiment

- Synthetic input, one batch of shape [1,3,640,640], reused per run; raw pose forward pass only.
- CPU-exported TorchScript model; static shape, no NMS or external postprocessing; validate exported FP32 against eager export-mode raw output before benchmark.
- GPU PyTorch backend, FP32 and FP16, three distinct input seeds (0,1,2) per precision.
- One CPU intra-op thread, deterministic mode enabled, 50 warmup and 200 steady invocations per run.
- Submit GPU work through the existing Task Spooler queue and gpu-run cooperative lock; never reorder jobs, change queue concurrency, stop another job or bypass its lock.
- Preserve per-run environment, model/provenance hashes, latency samples, telemetry, failure logs and archives. Include all runs in aggregate; never silently drop a failed run.

## Predeclared pass rule

This experiment passes **measurement collection**, not deployment suitability, only if all six requested runs complete, each has 200 steady samples and finite GPU event measurements, and each model hash matches the validated export manifest. Export must pass numeric agreement and expected 17x3-keypoint output checks. Verify archive/manifests. Report incomplete if any run fails or is interrupted.

Do not score RTX timings against Jetson budgets or call allocation-only peaks total model/system memory. Report each seed and precision separately; do not combine precisions or substitute inference rate for camera FPS. Optional CPU runs are a separate desktop CPU reference, never a Jetson CPU approximation.

## Explicitly excluded

Camera decode/resize/normalization, NMS/application-level pose extraction, tracking, temporal gesture classification, accuracy, detection-onset latency, real robotics deadlines and real Jetson performance. A real-image pose smoke is qualitative execution evidence, not accuracy evaluation. No held-out data is used for tuning.

Post-run terminology correction, 2026-11-07: the original exclusion of “pose decode” was too broad. The exported network includes internal YOLO pose-head decoding; external selection/postprocessing is excluded. No pass rule, configuration, or score was changed.
