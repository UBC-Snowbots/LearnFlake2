"""Inference backends for a single, preallocated synthetic input.

All reported durations are milliseconds. Initialization, model loading and synthetic
input allocation occur before sampling and are not included in the timings.
"""

import contextlib
import hashlib
import io
import math
import os
import time


def _ms(start):
    return (time.perf_counter_ns() - start) / 1_000_000.0


def _validate(config):
    shape = config.get("input_shape")
    if (not isinstance(shape, (list, tuple)) or not shape or
            any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)):
        raise ValueError("input_shape must be a nonempty list of positive integers")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config.get("precision") not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")
    index = config.get("gpu_index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("gpu_index must be a nonnegative integer")
    if config["device"] == "cpu" and config["precision"] == "fp16":
        raise ValueError("fp16 on CPU is unsupported")
    if not isinstance(config.get("deterministic", True), bool):
        raise ValueError("deterministic must be a boolean")
    threads = config.get("cpu_threads", 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    return tuple(shape)


def _file_size(path):
    return os.path.getsize(path) if path and os.path.isfile(path) else None


def _sha256(path):
    if path == "smoke":
        return None
    digest = hashlib.sha256()
    with open(os.fspath(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configure_torch(torch, config):
    deterministic = config.get("deterministic", True)
    if deterministic and config["device"] == "cuda":
        # Must precede the first cuBLAS handle creation; do not overwrite a user's
        # explicit (possibly incompatible) workspace setting.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.get("cpu_threads", 1))
    torch.manual_seed(config.get("seed", 0))
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _base_info(config, size):
    return {
        "backend": config["backend"], "model": str(config["model"]),
        "device": config["device"], "gpu_index": config.get("gpu_index", 0) if config["device"] == "cuda" else None,
        "input_shape": list(config["input_shape"]), "precision": config["precision"],
        "serialized_bytes": size,
        "serialized_bytes_reason": None if size is not None else "No serialized model file",
        "serialized_format": "model file" if size is not None else None,
        "model_sha256": _sha256(config["model"]),
        "model_sha256_scope": "Model file bytes only; excludes ONNX external tensor data" if config["model"] != "smoke" else "Unavailable: built-in smoke has no model file",
        "input_dtype": "float16" if config["precision"] == "fp16" else "float32",
        "precision_scope": "Requested input dtype; internal operator/engine precision is not measured or guaranteed",
        "deterministic_requested": config.get("deterministic", True),
        "cpu_threads_requested": config.get("cpu_threads", 1),
        "parameter_count": None, "parameter_count_reason": "Not available from this runtime",
        "flops": None, "flops_reason": "FLOPs are not measured",
        "pipeline_boundaries": {
            "initialization": "Model loading, runtime setup, and synthetic input allocation are excluded",
            "preprocess": "Select the already allocated synthetic input; no decoding, resizing, or normalization",
            "inference_wall": "Runtime invocation through completion, including synchronization and any runtime transfers",
            "inference_gpu": "CUDA event elapsed time when supported; otherwise unavailable",
            "postprocess": "Retain runtime outputs; no decoding or host transfer is explicitly performed",
            "end_to_end": "Preprocess, inference, and postprocess as defined above",
        },
    }


class _Backend:
    def model_info(self):
        return self._info

    def memory_info(self):
        return {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None,
                "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
                "scope": "Runtime allocations are not observable through this backend",
                "reason": "Runtime memory usage is unavailable"}

    def reset_peak_memory(self):
        """Reset observable allocator peaks after warmup, where supported."""
        pass

    def close(self):
        pass

    def sample(self):
        total = time.perf_counter_ns()
        start = time.perf_counter_ns()
        inputs = self._input
        preprocess = _ms(start)
        wall, gpu, outputs = self._infer(inputs)
        start = time.perf_counter_ns()
        self._output = outputs
        postprocess = _ms(start)
        return {"preprocess_ms": preprocess, "inference_wall_ms": wall,
                "inference_gpu_ms": gpu, "postprocess_ms": postprocess,
                "end_to_end_ms": _ms(total)}


class _TorchBackend(_Backend):
    def __init__(self, config, shape):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch backend requires torch") from exc
        self.torch = torch
        settings = _configure_torch(torch, config)
        self._peak_scope = "since process start, including initialization and profiling"
        self.device = torch.device("cpu" if config["device"] == "cpu" else f"cuda:{config.get('gpu_index', 0)}")
        if self.device.type == "cuda" and (not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count()):
            raise RuntimeError(f"Requested CUDA device {self.device} is unavailable")
        dtype = torch.float16 if config["precision"] == "fp16" else torch.float32
        model = config["model"]
        if model == "smoke":
            # Elementwise operations work for every valid shape and have parameters.
            class Smoke(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.scale = torch.nn.Parameter(torch.ones(()))
                    self.bias = torch.nn.Parameter(torch.zeros(()))

                def forward(self, x):
                    return torch.relu(x * self.scale + self.bias)
            self._model = Smoke()
        else:
            try:
                self._model = torch.jit.load(os.fspath(model), map_location=self.device)
            except Exception as exc:
                raise RuntimeError(f"Could not load TorchScript model {model!r}: {exc}") from exc
            arguments = list(self._model.forward.schema.arguments)
            if arguments and arguments[0].name == "self":
                arguments = arguments[1:]
            if len(arguments) != 1 or str(arguments[0].type) != "Tensor":
                raise ValueError("TorchScript model must accept exactly one Tensor input")
        self._model = self._model.to(device=self.device, dtype=dtype).eval()
        generator = torch.Generator(device=self.device).manual_seed(config.get("seed", 0))
        self._input = torch.randn(shape, device=self.device, dtype=dtype, generator=generator)
        self._info = _base_info(config, _file_size(model) if model != "smoke" else None)
        self._info["runtime_settings"] = settings
        self._info["determinism_scope"] = "Seeded synthetic input and PyTorch deterministic algorithms; unsupported nondeterministic operations raise; reproducibility across runtime versions/hardware is not guaranteed"
        self._info["precision_scope"] = "Input and floating-point model tensors cast to requested dtype; internal operator precision is not measured; TF32 disabled"
        if model == "smoke":
            serialized = io.BytesIO()
            torch.save(self._model.state_dict(), serialized)
            self._info["serialized_bytes"] = serialized.tell()
            self._info["serialized_bytes_reason"] = None
            self._info["serialized_format"] = "torch.save state_dict, not an executable model"
        self._info["flops_scope"] = "One forward pass before warmup; PyTorch supported-operator estimate, not measured hardware FLOPs"
        self._info["parameter_count"] = sum(p.numel() for p in self._model.parameters())
        self._info["parameter_count_reason"] = None
        self._info["parameter_count_scope"] = "Unique PyTorch parameter elements; excludes buffers"
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            with torch.cuda.device(self.device):
                self._start_event = torch.cuda.Event(enable_timing=True)
                self._end_event = torch.cuda.Event(enable_timing=True)
                stream = torch.cuda.current_stream(self.device)
                self._start_event.record(stream)
                self._end_event.record(stream)
                stream.synchronize()
        # Prefer dispatch counting: scripted convolution events may be invisible
        # to the event profiler even when it reports other (partial) FLOP counts.
        invocation_started = invocation_completed = False
        device_context = torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext()
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with device_context, torch.inference_mode(), FlopCounterMode(display=False) as counter:
                invocation_started = True
                result = self._model(self._input)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                invocation_completed = True
            del result
            flops = counter.get_total_flops()
            if flops:
                self._info["flops"] = flops
                self._info["flops_reason"] = "PyTorch dispatch-counter supported-operator estimate; two FLOPs per multiply-add; may undercount"
            else:
                self._info["flops_reason"] = "Dispatch counter found no supported FLOP-counted operators; not necessarily zero"
        except Exception as exc:
            if invocation_started and (not invocation_completed or self.device.type == "cuda"):
                raise
            self._info["flops_reason"] = f"Dispatch counter unavailable: {type(exc).__name__}: {exc}"
        if self._info["flops"] is None:
            invocation_started = invocation_completed = False
            try:
                with device_context, torch.inference_mode(), torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True
                ) as prof:
                    invocation_started = True
                    result = self._model(self._input)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    invocation_completed = True
                del result
                flops = sum(event.flops or 0 for event in prof.key_averages())
                if flops:
                    self._info["flops"] = flops
                    self._info["flops_reason"] = "Fallback PyTorch event-profiler supported-operator estimate; scripted convolution may be omitted"
                    self._info["flops_scope"] = "Additional pre-warmup forward because dispatch counting was unavailable; not measured hardware operations"
            except Exception as exc:
                if invocation_started and (not invocation_completed or self.device.type == "cuda"):
                    raise
                self._info["flops_reason"] += f"; event profiler unavailable: {type(exc).__name__}: {exc}"


    def _infer(self, inputs):
        torch = self.torch
        gpu = None
        device_context = torch.cuda.device(self.device) if self.device.type == "cuda" else contextlib.nullcontext()
        with device_context, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                stream = torch.cuda.current_stream(self.device)
            start = time.perf_counter_ns()
            if self.device.type == "cuda":
                self._start_event.record(stream)
            outputs = self._model(inputs)
            if self.device.type == "cuda":
                self._end_event.record(stream)
                torch.cuda.synchronize(self.device)
            wall = _ms(start)
            if self.device.type == "cuda":
                gpu = float(self._start_event.elapsed_time(self._end_event))
        return wall, gpu, outputs

    def memory_info(self):
        if self.device.type != "cuda":
            return super().memory_info()
        return {"gpu_allocated_bytes": self.torch.cuda.memory_allocated(self.device),
                "gpu_reserved_bytes": self.torch.cuda.memory_reserved(self.device),
                "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
                "reason": None, "scope": f"Process-wide PyTorch allocator on {self.device}; peaks {self._peak_scope}; excludes other processes and non-PyTorch allocations"}

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self._peak_scope = "since reset after warmup; persistent model/input and live warmup output remain included"

    def close(self):
        self._output = None
        self._input = None
        self._model = None


class _OnnxBackend(_Backend):
    def __init__(self, config, shape):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime backend requires numpy and onnxruntime") from exc
        model = config["model"]
        if model == "smoke":
            raise ValueError("ONNX Runtime requires an ONNX model file, not smoke")
        provider = "CUDAExecutionProvider" if config["device"] == "cuda" else "CPUExecutionProvider"
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"ONNX Runtime provider {provider} is unavailable")
        options = ort.SessionOptions()
        options.intra_op_num_threads = config.get("cpu_threads", 1)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.use_deterministic_compute = config.get("deterministic", True)
        ort.set_seed(config.get("seed", 0))
        if config["device"] == "cuda":
            # Reject graphs that would otherwise execute unsupported nodes on CPU.
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            providers = [(provider, {"device_id": config.get("gpu_index", 0)})]
        else:
            providers = [provider]
        try:
            self._session = ort.InferenceSession(os.fspath(model), sess_options=options, providers=providers)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize ONNX Runtime with {provider}: {exc}") from exc
        if provider not in self._session.get_providers():
            raise RuntimeError(f"ONNX Runtime did not activate requested provider {provider}")
        self._session.disable_fallback()
        if self._session.get_overridable_initializers():
            raise ValueError("ONNX overridable initializer inputs are unsupported")
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("ONNX model must have exactly one input")
        expected = "tensor(float16)" if config["precision"] == "fp16" else "tensor(float)"
        if inputs[0].type != expected:
            raise ValueError(f"ONNX input type {inputs[0].type} does not match {expected}")
        declared = inputs[0].shape
        if len(declared) != len(shape) or any(isinstance(d, int) and d > 0 and d != actual for d, actual in zip(declared, shape)):
            raise ValueError(f"ONNX input shape {declared} is incompatible with {shape}")
        self._name = inputs[0].name
        dtype = np.float16 if config["precision"] == "fp16" else np.float32
        self._input = np.random.default_rng(config.get("seed", 0)).standard_normal(shape).astype(dtype)
        self._info = _base_info(config, _file_size(model))
        self._info["provider"] = provider
        actual_options = self._session.get_session_options()
        self._info["runtime_settings"] = {
            "onnxruntime_version": ort.__version__, "numpy_version": np.__version__,
            "cpu_threads": actual_options.intra_op_num_threads,
            "cpu_interop_threads": actual_options.inter_op_num_threads,
            "execution_mode": str(actual_options.execution_mode),
            "deterministic_compute": actual_options.use_deterministic_compute,
            "providers": self._session.get_providers(),
            "provider_options": self._session.get_provider_options(),
            "cuda_version": None, "cudnn_version": None,
            "cuda_cudnn_version_reason": "Loaded CUDA/cuDNN versions are not exposed by the ONNX Runtime Python session API",
        }
        self._info["determinism_scope"] = "ONNX Runtime deterministic-compute request and seeded synthetic input; provider/kernel determinism is not independently verified"
        self._info["parameter_count_scope"] = "ONNX graph initializer elements (including sparse logical shapes); not a trainable-parameter count"
        try:
            import onnx
        except ImportError:
            self._info["parameter_count_reason"] = "Optional onnx package is not installed; ONNX Runtime does not expose initializer parameter counts"
        else:
            try:
                graph = onnx.load(os.fspath(model), load_external_data=False).graph
                self._info["parameter_count"] = (
                    sum(math.prod(tensor.dims) for tensor in graph.initializer)
                    + sum(math.prod(tensor.dims) for tensor in graph.sparse_initializer)
                )
                self._info["parameter_count_reason"] = None
            except Exception as exc:
                self._info["parameter_count_reason"] = f"ONNX metadata could not be read: {type(exc).__name__}: {exc}"
        self._info["pipeline_boundaries"]["inference_wall"] = "Synchronous session.run, including host/device transfers for CUDA and output copies to host"
        self._info["pipeline_boundaries"]["inference_gpu"] = "Unavailable: session.run does not expose a valid CUDA event interval"

    def _infer(self, inputs):
        start = time.perf_counter_ns()
        outputs = self._session.run(None, {self._name: inputs})
        return _ms(start), None, outputs

    def close(self):
        self._output = None
        self._input = None
        self._session = None


class _TensorRTBackend(_Backend):
    def __init__(self, config, shape):
        if config["device"] != "cuda":
            raise ValueError("TensorRT requires device=cuda")
        if config["model"] == "smoke":
            raise ValueError("TensorRT requires a serialized engine file, not smoke")
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise RuntimeError("TensorRT backend requires tensorrt and CUDA-enabled torch") from exc
        self.torch = torch
        settings = _configure_torch(torch, config)
        self._peak_scope = "since process start, including engine initialization"
        self.device = torch.device(f"cuda:{config.get('gpu_index', 0)}")
        if not torch.cuda.is_available() or self.device.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device {self.device} is unavailable")
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        with open(os.fspath(config["model"]), "rb") as f:
            engine_data = f.read()
        with torch.cuda.device(self.device):
            self._engine = self._runtime.deserialize_cuda_engine(engine_data)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the engine; check runtime, GPU and engine compatibility")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context")
            self._setup(trt, config, shape)
            torch.cuda.synchronize(self.device)
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            self._start_event.record(stream)
            self._end_event.record(stream)
            stream.synchronize()
        self._info = _base_info(config, len(engine_data))
        settings["tensorrt_version"] = trt.__version__
        settings["cuda_cudnn_version_scope"] = "PyTorch linked libraries; TensorRT may use different libraries, which are not exposed here"
        self._info["runtime_settings"] = settings
        self._info["determinism_scope"] = "Seeded synthetic input; PyTorch deterministic settings do not control prebuilt TensorRT engine tactics or guarantee engine determinism"
        self._info["precision_scope"] = "Engine input dtype checked against request; prebuilt engine internal precision and tactics are not changed or inferred"
        self._info["parameter_count_reason"] = "Serialized TensorRT engine does not expose a reliable parameter count"
        self._info["pipeline_boundaries"]["inference_wall"] = "TensorRT asynchronous execution followed by CUDA stream synchronization"
        self._info["pipeline_boundaries"]["postprocess"] = "Retain preallocated device output tensors; no decoding or host transfer"

    def _setup(self, trt, config, shape):
        torch = self.torch
        engine, context = self._engine, self._context
        if not hasattr(engine, "num_io_tensors") or not hasattr(context, "execute_async_v3"):
            raise RuntimeError("TensorRT backend requires the named I/O and execute_async_v3 APIs (TensorRT 8.5+)")
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        for tensor_name in names:
            if engine.get_tensor_location(tensor_name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"TensorRT host-located tensor {tensor_name!r} is unsupported")
            if engine.get_tensor_format(tensor_name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"TensorRT non-linear tensor format for {tensor_name!r} is unsupported")
        inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        if len(inputs) != 1:
            raise ValueError("TensorRT engine must have exactly one input")
        name = inputs[0]
        if hasattr(engine, "is_shape_inference_io") and engine.is_shape_inference_io(name):
            raise ValueError("TensorRT shape tensor inputs are unsupported")
        requested = trt.float16 if config["precision"] == "fp16" else trt.float32
        if engine.get_tensor_dtype(name) != requested:
            raise ValueError("TensorRT engine input dtype does not match requested precision")
        if not context.set_input_shape(name, shape):
            raise ValueError(f"TensorRT input shape {shape} is not supported by the engine's first optimization profile")
        generator = torch.Generator(device=self.device).manual_seed(config.get("seed", 0))
        dtype = torch.float16 if config["precision"] == "fp16" else torch.float32
        self._input = torch.randn(shape, device=self.device, dtype=dtype, generator=generator)
        self._buffers = {name: self._input}
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16,
                  trt.int32: torch.int32, trt.int8: torch.int8, trt.bool: torch.bool}
        for output in names:
            if output == name:
                continue
            out_shape = tuple(context.get_tensor_shape(output))
            if any(d < 0 for d in out_shape):
                raise ValueError(f"TensorRT output {output!r} has unresolved or data-dependent shape {out_shape}")
            out_dtype = engine.get_tensor_dtype(output)
            if out_dtype not in dtypes:
                raise ValueError(f"Unsupported TensorRT output dtype {out_dtype} for {output!r}")
            self._buffers[output] = torch.empty(out_shape, dtype=dtypes[out_dtype], device=self.device)
        for tensor_name, tensor in self._buffers.items():
            if not context.set_tensor_address(tensor_name, tensor.data_ptr()):
                raise RuntimeError(f"Could not bind TensorRT tensor {tensor_name!r}")

    def _infer(self, inputs):
        torch = self.torch
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            stream.synchronize()
            start_event = self._start_event
            end_event = self._end_event
            start = time.perf_counter_ns()
            start_event.record(stream)
            if not self._context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            end_event.record(stream)
            stream.synchronize()
            wall = _ms(start)
            gpu = float(start_event.elapsed_time(end_event))
        return wall, gpu, self._buffers

    def memory_info(self):
        return {"gpu_allocated_bytes": self.torch.cuda.memory_allocated(self.device),
                "gpu_reserved_bytes": self.torch.cuda.memory_reserved(self.device),
                "gpu_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(self.device),
                "gpu_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(self.device),
                "reason": None, "scope": f"Process-wide PyTorch allocator on {self.device}; peaks {self._peak_scope}; excludes TensorRT allocations, driver/context allocations, and other processes"}

    def reset_peak_memory(self):
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_scope = "since reset after warmup; persistent PyTorch I/O buffers remain included"

    def close(self):
        self._output = None
        self._input = None
        self._buffers = None
        self._context = None
        self._engine = None
        self._runtime = None


def create_backend(config):
    """Create and initialize the backend selected by a flat configuration dict."""
    shape = _validate(config)
    backend = config.get("backend")
    if backend == "pytorch":
        return _TorchBackend(config, shape)
    if backend == "onnxruntime":
        return _OnnxBackend(config, shape)
    if backend == "tensorrt":
        return _TensorRTBackend(config, shape)
    raise ValueError(f"Unsupported backend: {backend!r}")
