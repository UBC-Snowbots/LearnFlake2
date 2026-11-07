# Orion

Private repository: [Luxaris/Orion](https://github.com/Luxaris/Orion). Install the Python package from a GitHub release (needs repo access):

```sh
gh release download v0.1.0 -R Luxaris/Orion -p '*.whl' -p 'orion-live-kit.tar.gz' -p 'SHA256SUMS'
sha256sum -c SHA256SUMS
pip install ./orion_rover_benchmark-0.1.0-py3-none-any.whl          # extras: [live], [train], [pose]
# or from source: pip install "git+ssh://git@github.com/Luxaris/Orion.git@v0.1.0"
```

`orion-live-kit.tar.gz` is the portable camera/video kit with model weights (see `docs/PORTABLE.md`). Weights are not committed to Git.

Unattended **synthetic inference** benchmarking with durable per-run artifacts. This is not a camera pipeline, dataset evaluation, gesture/detection accuracy measurement, or Jetson emulator. Compare only matching model, shape, precision, backend boundaries, and recorded hardware/software conditions. No performance claim follows from the built-in smoke model.

## Install and run

Python 3.11+ is required. From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[pytorch]'
python -m orion_bench run --config examples/smoke_cpu.json
```

The base package has no runtime dependencies. `.[pytorch]` adds PyTorch; `.[onnxruntime]` adds NumPy and CPU ONNX Runtime. Install the appropriate vendor CUDA PyTorch wheel, ONNX Runtime GPU package, or TensorRT distribution yourself for the target platform; packaging does not force GPU dependencies. Optional `onnx` enables ONNX initializer-count metadata, not inference. Do not replace a working CUDA environment's packages just to run CPU examples.

```sh
# In an existing compatible CUDA PyTorch environment:
python -m orion_bench run --config examples/smoke_gpu.json
```

## Configuration

Configuration is a flat JSON object. Unknown keys are rejected. Model/output paths resolve relative to the configuration file, not the launch directory; absolute paths are safest for queued jobs. Example outputs are under `examples/benchmark_results`.

| Key | Default / meaning |
|---|---|
| `backend` | `pytorch`; also `onnxruntime`, `tensorrt` |
| `model` | `smoke` (PyTorch only), otherwise a model file path |
| `model_manifest` | `null`; optional JSON provenance path, resolved relative to config; export SHA-256 and input shape must match |
| `device`, `gpu_index` | `cpu`, `0`; `cuda` uses a logical index under `CUDA_VISIBLE_DEVICES` |
| `input_shape` | `[1,3,224,224]`; positive dimensions including batch first |
| `precision` | `fp32`; also `fp16`; CPU fp16 is unsupported |
| `warmup`, `iterations` | `10`, `100`; invocation counts, warmup may be zero |
| `seed` | `0`; nonnegative integer, synthetic input is generated once and reused |
| `deterministic` | `true`; boolean, requests runtime deterministic behavior |
| `cpu_threads` | `1`; positive integer, PyTorch intra-op / ONNX Runtime intra-op threads |
| `output_root` | `runs`; output parent directory |
| `platform_label` | `desktop`; descriptive label, not hardware detection |
| `telemetry_interval_s` | `1.0`; positive finite seconds |
| `availability` | `{}`; optional GPU availability thresholds below |

PyTorch deterministic mode enables deterministic algorithms (unsupported nondeterministic operations fail), controls cuDNN behavior, and disables TF32. CUDA uses `CUBLAS_WORKSPACE_CONFIG=:4096:8` unless already configured. ONNX Runtime requests deterministic compute and sequential execution. TensorRT uses a prebuilt engine: this flag does not change tactics or guarantee engine determinism. Seeds/settings do **not** guarantee bitwise equivalence across runtime versions, hardware, providers, or operating systems, and do not make measured durations repeatable. Actual settings and scope are in `model_info.json`.

`availability` accepts `min_free_vram_mb` (MiB), `max_gpu_utilization_percent`, `timeout_s` (default 60 seconds), and `poll_s` (default 1 second). Thresholds require CUDA; `{}` disables waiting. Missing required metrics or timeout fails the run. Polling does not reserve the GPU or ensure isolation. Never kill another user's workload to satisfy thresholds.

## Backends and measurement boundaries

* **PyTorch:** built-in `smoke` is elementwise scale/bias/ReLU, not a representative vision network. Otherwise provide a TorchScript model accepting one configured tensor.
* **ONNX Runtime:** provide a single-floating-input ONNX model matching shape and dtype. Selected provider must be available; CUDA does not silently fall back to CPU. Synchronous `session.run` includes input/output transfers and host outputs. CUDA-event inference timing is unavailable.
* **TensorRT:** provide a compatible serialized engine, CUDA and PyTorch. Requires named I/O and `execute_async_v3` (TensorRT 8.5+), one execution input, and resolvable outputs in the first optimization profile. Shape-tensor inputs and unresolved/data-dependent output dimensions are unsupported. Precision validates the engine input I/O dtype and supported output buffer dtypes; it does **not** rebuild the engine or measure/change its internal arithmetic precision. Output buffers stay on GPU.

Initialization, model loading, synthetic input allocation and optional FLOP profiling precede warmup and are excluded from latency samples. Preprocessing selects preallocated input; postprocessing retains outputs. There is no image decode, resize, normalization, detection decoding, or application decision. Inference wall time is runtime invocation through completion, including synchronization and backend transfers. CUDA-event time is reported for PyTorch CUDA and TensorRT only. End-to-end time covers these defined synthetic stages, including measurement overhead, not camera-to-decision latency; it need not equal the sum of stage measurements.

Serialized size is bytes on disk (built-in smoke uses an in-memory `torch.save` state dictionary, not an executable model). Model SHA-256 covers model-file bytes only, not ONNX external tensor data. PyTorch parameter count is parameter elements, excluding buffers; ONNX count, when available, is initializer elements, not trainable parameters. FLOPs are a supported-operator **estimate per forward invocation**, not measured hardware operations. Optional `torch.utils.flop_counter` dispatch counting is preferred because event profiling can omit scripted convolutions even while returning other nonzero counts; event profiling is the fallback. Multiply-adds count as two FLOPs; unsupported operators may be omitted. Method/scope is recorded. Unknown counts/estimates are null with reasons, never inferred from file size.

## Output schema and units

Each run gets a unique directory and adjacent `.tar.gz`. CLI completion prints `run_dir=...`, `archive=...`, `status=...`; completed exits zero, failed/interrupted exits nonzero. Invalid config or failures before directory creation may have no artifacts. SIGINT/SIGTERM normally finalize an interrupted archive; SIGKILL, storage failure, or machine loss can leave incomplete artifacts. Inspect status, errors and warnings before using partial data.

| Artifact | Contents |
|---|---|
| `config.json` | Effective configuration, resolved paths |
| `environment.json` | Host/OS/Python, installed runtime versions, CUDA visibility, GPU/driver discovery and availability errors, harness version and source-file SHA-256 fingerprints |
| `model_info.json` | Backend, model, shape/dtype, runtime settings, size/hash/count metadata, measurement and determinism scope |
| `summary.json` | `schema_version: "1.0"`, status (`completed`, `failed`, `interrupted`), `error`, `warnings`, `warmup_completed`, `completed_iterations`, `warmup_metrics`, steady `metrics`, `memory`, `telemetry` |
| `latency_samples.csv` | `phase,iteration,preprocess_ms,inference_wall_ms,inference_gpu_ms,postprocess_ms,end_to_end_ms` |
| `gpu_telemetry.csv` | Timestamp/phase, selected GPU, utilization, VRAM, power, temperature, clocks, other compute processes, errors |
| `system_telemetry.csv` | Timestamp/phase, host CPU utilization, total/available host memory, benchmark process RSS |
| `benchmark.log`, `report.md` | Lifecycle/errors and human-readable report |
| `manifest.sha256` | SHA-256 plus two spaces plus filename, covering all run files except itself |
| `model_provenance.json` | Optional verified model-manifest snapshot, also embedded under `model_info.provenance`; included in run hashes/archive |

Latency CSV phase is `warmup` or `steady`; iteration is zero-based within each phase. All five latency columns are **milliseconds**; unavailable GPU durations are empty CSV cells / JSON null, not zero. Each stage distribution in `metrics` and `warmup_metrics` contains `count,mean,median,p90,p95,p99`; percentiles linearly interpolate sorted observations. Warmup and steady distributions are separate. Empty distributions have count zero and null statistics.

Steady `metrics.measured_loop_seconds` is wall seconds including measurement, synchronization and sample-writing overhead. `inferences_per_second` is completed steady invocations / loop seconds. `effective_fps` multiplies by batch size: synthetic batch elements per second, not measured camera FPS.

Memory fields ending `_bytes` are bytes. Backend `gpu_allocated_bytes`, `gpu_reserved_bytes`, `gpu_peak_allocated_bytes`, `gpu_peak_reserved_bytes` describe the selected device's **process-wide PyTorch allocator only** where supported, not total GPU usage. Peaks reset after warmup; persistent model/input/live output remain included. TensorRT internal/driver allocations and other processes are excluded. `process_peak_rss_bytes` is Linux process peak RSS since process start, including initialization and warmup. Null values and scope/reason fields distinguish unavailable measurements.

Telemetry timestamps are UTC ISO 8601; utilization is percent, `_mb` memory fields are MiB despite the legacy column names, power is watts, temperature Celsius, clocks MHz. Host CPU utilization is aggregate host CPU activity between samples, not benchmark-only CPU use; the initial utilization may be blank. `other_compute_processes` is a JSON array (PID, name, VRAM MiB), or unavailable; blank cells/errors do not establish zero contention. Telemetry is sampled and can miss brief activity, especially tiny smoke runs; it is not a reservation or a source of exact inference power/energy measurements.

Both telemetry CSVs start with `timestamp,timestamp_ns,monotonic_ns,phase,phase_end,collection_duration_ms`. Timestamps are collected independently per stream: UTC text has nine fractional digits, epoch and monotonic values are integer nanoseconds, collection duration is milliseconds. Phase values are `availability`, `initialization`, `warmup`, `steady`, `finalization` (initial fallback `initializing`); `phase_end` reveals collection spanning a phase change.

GPU columns then are `gpu_index,gpu_uuid,gpu_utilization_percent,vram_used_mb,vram_free_mb,vram_total_mb,power_w,temperature_c,graphics_clock_mhz,memory_clock_mhz,other_compute_processes,error`. GPU index is the logical CUDA ordinal; UUID identifies the verified physical GPU. MIG telemetry mapping is unsupported rather than guessed. System columns then are `cpu_utilization_percent,memory_total_mb,memory_available_mb,process_rss_mb,process_rss_peak_mb,error`. RSS peak is process-lifetime peak, not just steady inference.

`summary.telemetry` groups GPU/system CSVs with `phase_sample_counts`, `steady_sampled_metrics` (`count,mean,min,max` per numeric field), `steady_other_compute_processes_seen`, and a scope explanation. These are sampled steady-phase observations, not exact allocation or power peaks; a tiny run may have no steady telemetry samples.

## Aggregation

```sh
python -m orion_bench.aggregate /absolute/path/to/RUN_ID /absolute/path/to/OTHER_RUN.tar.gz --output results.csv
python -m orion_bench.aggregate /absolute/path/to/RUN.tar.gz --output results.json
```

Reads directories and ZIP/TAR archives without extraction. JSON output is an array; CSV is a table. Scalar metadata is flattened under `summary.`, `config.`, `model_info.`, `environment.`; arrays are JSON-encoded strings. `source` and `run` identify each row. Missing/malformed artifacts appear in `artifact_errors`; failed/partial runs remain visible. Passing both a run directory and its archive intentionally produces duplicate rows.

## Task Spooler queues

Install `tsp`, activate the intended environment, then:

```sh
# Keep the site's existing TS_SOCKET (or default shared queue) for normal jobs.
scripts/submit_tsp.sh /absolute/path/to/config.json
# UBC Rover host: honor the existing cooperative gpu-run lock:
BENCH_GPU_WRAPPER=gpu-run scripts/submit_tsp.sh /absolute/path/to/gpu-config.json
# Use the printed job ID and the same TS_SOCKET for inspection:
tsp -s JOB_ID
tsp -c JOB_ID
# Gracefully interrupt your own job only:
tsp -k JOB_ID
```

The launcher resolves the config, runs from the project root, and preserves exported environment/active Python environment. It does not change queue concurrency. Keep `TS_SOCKET`/`TMPDIR` consistent; do not manipulate someone else's/default shared queue. Task Spooler coordinates only its own queue, not other GPU users. Shared GPU/CPU load, memory pressure, clocks, thermals and power affect comparisons. Desktop limits do not emulate Jetson; Jetson results require the actual hardware/software stack.

`BENCH_GPU_WRAPPER` is optional and names one executable, not a shell command. On this workspace `gpu-run` uses `flock /run/lock/gpu.lock`; it only serializes cooperating jobs and does not establish uncontended hardware. The launcher freezes the absolute Python executable at submission, not the model/config files; avoid changing those files while queued. Final effective config and model-file hash are recorded.

Orion lives at `~/projects/ubc-rover/Orion`, beside the existing `RoverFlake2` and `LearnFlake` repositories. The former parent path `~/projects/ubc-rover-arm` is a compatibility symlink so live rover processes, generated ROS installs and virtual environments retain access. No team repository contents are changed by running this harness.

## First pretrained pose candidate

The first candidate is **YOLO11n-pose**, not a gesture classifier. The official checkpoint is pinned by immutable Hugging Face revision and SHA-256 in `examples/yolo11n_pose_pin.json`. Code/model licensing is AGPL-3.0 or Ultralytics Enterprise; review before redistribution or closed-source deployment. No training or accuracy evaluation is performed.

Use a compatible PyTorch environment with `ultralytics==8.4.174` (`.[pose]` installs that optional dependency). Download with the Hugging Face `hf` CLI, then export entirely on CPU:

```sh
hf download Ultralytics/YOLO11 yolo11n-pose.pt --revision 8b8ac7d1fae7468f85dbf89670dd66f41f485aab --local-dir models/checkpoints/yolo11
python -m orion_bench.prepare_pose --checkpoint models/checkpoints/yolo11/yolo11n-pose.pt --output-dir models/yolo11n_pose_640 --pin examples/yolo11n_pose_pin.json --imgsz 640
python -m orion_bench run --config examples/yolo11n_pose_cpu.json
```

Preparation verifies the checkpoint hash **before pickle loading**, checks the pinned package version, exports a non-frozen raw TorchScript forward, and verifies serialized FP32 predictions against eager export-mode predictions. It supports fixed `[1,3,640,640]` input and 17 keypoints. A repeated preparation verifies the existing artifact/settings instead of overwriting it. `models/` contains generated/downloaded artifacts and is ignored by Git.

For the shared GPU host, submit the six-run worker to the existing queue and cooperative lock:

```sh
tsp -L OrionPose640 gpu-run "$(command -v python)" "$PWD/scripts/run_pose_candidate.py" --config "$PWD/examples/yolo11n_pose_gpu.json" --output-dir "$PWD/runs/pose-first"
```

Defaults are seeds **0, 1, 2**, FP32 and FP16, 50 warmup and 200 steady invocations per run, one CPU intra-op thread. The FP16 path casts the FP32 TorchScript model at runtime; it is not a TensorRT engine or a guarantee that every internal operation uses FP16. The worker never modifies queue settings; `--seeds` and `--precisions` permit explicit repeats. SIGTERM is forwarded to the active benchmark child and results are preserved.

Each unique group contains effective configs, console logs, standard per-run archives, `aggregate.csv`, `aggregate.json`, and `group_summary.json`. `measurements_collected` means all requested runs passed the collection checks—not that they meet Jetson limits. Failed/interrupted/unstarted attempts remain visible. Protocol and current evidence are in `notes/POSE_PROTOCOL.md` and `notes/POSE_CANDIDATE.md`.

These measurements include internal pose-head decoding, but exclude image decode, resize/normalization, NMS, application-level pose extraction, tracking and temporal classification. They do not establish camera FPS, gesture accuracy, reaction latency or loaded-Jetson resource headroom. ONNX/TensorRT exports of this candidate are not part of this first TorchScript benchmark.

## Temporal attention-gesture classifier

Public-data proxy for "someone is trying to get the rover's attention" (hand waving vs. confusers vs. everything else), on COCO-17 skeleton sequences. Protocol: `notes/ATTENTION_PROTOCOL.md`; results and limits: `notes/ATTENTION_RESULTS.md`. Datasets go in ignored `data/`; NTU RGB+D 120 is non-commercial research only.

```sh
python -m orion_bench.temporal_data --spec examples/attention_datasets.json --raw-dir data/raw/pyskl --output-dir data/attention_v1
python scripts/train_attention.py --data data/attention_v1 --output-dir runs/attention/w48 --window 48 --steps 20000 --eval-every 2500 --seed 0
python scripts/eval_attention.py --run runs/attention/w48/seed-0 --data data/attention_v1   # held-out test, once
python scripts/export_attention.py --run runs/attention/w48/seed-0 --output-dir models/attention_gcn_w48_s0
python -m orion_bench run --config examples/attention_gcn_cpu.json
```

Training selects checkpoint and threshold on validation only; `--score-test` or `eval_attention.py` scores test once for a frozen run.

## Portable camera and video test kit

Create a transferable archive containing the experimental pose and temporal models:

```sh
python scripts/package_live_kit.py --output "$PWD/dist/orion-live-kit.tar.gz"
```

Transfer and extract it on the target, verify `SHA256SUMS`, and follow the kit's README or [portable instructions](docs/PORTABLE.md). CPU and NVIDIA CUDA PyTorch are supported; Jetson dependencies must match its JetPack. This experimental pipeline has not been validated on Jetson or for beckoning, and must not control the rover.

## Reusable offline smoke

```sh
python scripts/smoke_test.py
python scripts/smoke_test.py --output-dir /tmp/rover-smoke-results
python scripts/smoke_test.py --tsp
# Explicit opt-in on compatible CUDA runtime/device:
python scripts/smoke_test.py --device cuda --gpu-index 0 --tsp
```

Requires only installed PyTorch (and Python standard library); ONNX is optional and not needed by the smoke. It launches actual `python -m orion_bench run` processes with tiny seeded inputs, verifies warmup/steady samples and independently recomputed distributions, artifact manifest hashes/archive bytes, deliberate missing-model failure and nonzero exit, and JSON/CSV aggregation. It sends SIGTERM only after observing a complete steady CSV row and checks the interrupted archive. No incidental wording or performance thresholds are tested. Outputs are temporary by default; `--output-dir` preserves a unique subdirectory. `--timeout` sets a per-command safety deadline, not a timing acceptance threshold. `--tsp` always creates its own unique private `TS_SOCKET`, uses `tsp -k` for interruption, and stops only its private server. It never operates on the default queue. Deterministic setup means seeded inputs/runtime policy, not identical latency values.

The smoke also exports and benchmarks a tiny TorchScript convolution, checking parameter count, model-file hash and the analytic convolution FLOP count when dispatch counting is installed. This catches models that execute correctly but lose profiling information through TorchScript.

## Evidence and known runtime limits

The original `v0_1` ZIP was not found in the supplied `/tmp` or `/home` searches. This is a new implementation, not a verified recreation of that archive.

Verified from this repository: editable installation; direct CPU smoke; private-TSP CPU and CUDA smoke (built-in model, TorchScript convolution, intentional model-load failure, SIGTERM, manifest/archive bytes and CSV/JSON aggregation); ONNX Runtime CPU fixture inference; CUDA FP16 and UUID visibility; availability timeout and invalid configurations; actual submission script with a spaced config path; steady host CPU/RSS telemetry. Evidence and exact locations are in `notes/INTEGRATION.md` and `notes/ledger.csv`.

TensorRT execution, ONNX Runtime CUDA, multi-GPU/MIG behavior and real Jetson performance remain unverified. No performance or model-suitability claim follows from these functional checks. Recent PyTorch versions warn that TorchScript tracing is deprecated; this harness currently accepts TorchScript files for the PyTorch backend, not `torch.export` programs.

Jetson systems without `nvidia-smi` retain the same artifact schema with unavailable GPU telemetry and explicit errors. Availability thresholds requiring those metrics cannot succeed there; use `{}` until a board-specific telemetry source is available. No desktop utilization, memory cap, power limit, or thread count is a Jetson equivalence claim.
