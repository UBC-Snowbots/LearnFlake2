# BackendHardening — 2026-11-06

Verdict: implementation complete; runtime acceptance checks not run (integration owner).

Owned implementation: `rover_bench/backends.py`.

- Validate/support `deterministic` (default true) and `cpu_threads` (default 1). Record actual runtime settings and version availability, with explicit TensorRT/ONNX determinism and internal-precision limitations.
- Record model-file SHA256 (ONNX external tensor data excluded), input dtype, parameter-count scope, and pre-warmup supported-operator FLOP-estimate scope. Smoke serialized size is a `torch.save` state dictionary, not an executable model. Optional `onnx` supplies initializer element counts; absence has an explicit reason.
- Reuse initialized CUDA timing events and execute on the selected device. CUDA/model failures during profiling propagate instead of becoming missing-profile metadata. ONNX runtime fallback is disabled; unsupported input contracts are rejected explicitly.
- Report current/peak selected-device PyTorch allocator memory with exclusions. Public addition: `reset_peak_memory()`; the integration owner should invoke after warmup. It resets observable PyTorch peaks only, not TensorRT/driver allocations or other processes.

Checks: not run; no experiments performed. The integration owner should run existing CPU/CUDA smoke and integration checks after all edits land. Nonzero CUDA-device timing requires a machine exposing that selected device; ONNX/TensorRT acceptance requires their real models/runtimes.
