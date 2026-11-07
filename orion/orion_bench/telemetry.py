"""Host discovery, GPU availability checks, and lightweight telemetry sampling."""

import csv
import ctypes
import ctypes.util
import datetime
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


_COMMAND_TIMEOUT_S = 2.0
_GPU_FIELDS = (
    "utilization.gpu", "memory.used", "memory.free", "memory.total",
    "power.draw", "temperature.gpu", "clocks.current.graphics",
    "clocks.current.memory",
)
_GPU_COLUMNS = (
    "gpu_utilization_percent", "vram_used_mb", "vram_free_mb", "vram_total_mb",
    "power_w", "temperature_c", "graphics_clock_mhz", "memory_clock_mhz",
)
_CUDART = None
_CUDART_ERROR = None
_CUDART_LOCK = threading.Lock()
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
    "CUBLAS_WORKSPACE_CONFIG",
)


def _sample_time():
    timestamp_ns = time.time_ns()
    seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
    timestamp = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return (timestamp.strftime("%Y-%m-%dT%H:%M:%S") + f".{nanoseconds:09d}+00:00",
            timestamp_ns, time.monotonic_ns())


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=_COMMAND_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode:
        return None, (result.stderr.strip() or result.stdout.strip() or
                      f"command exited with status {result.returncode}")
    return result.stdout, None


def _smi_query(index, fields, kind="gpu"):
    args = ["nvidia-smi", f"--query-{kind}={','.join(fields)}", "--format=csv,noheader,nounits"]
    if index is not None:
        args.extend(["-i", str(index)])
    text, error = _command(args)
    if error:
        return None, error
    try:
        rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    except csv.Error as exc:
        return None, str(exc)
    return rows, None


def _value(value):
    value = value.strip()
    missing = {"", "n/a", "[n/a]", "not supported", "[not supported]",
               "unknown", "-", "nan", "inf", "-inf", "infinity"}
    return "" if value.lower() in missing else value


def _cuda_runtime():
    """Load the process's CUDA runtime, without importing an inference framework."""
    global _CUDART, _CUDART_ERROR
    with _CUDART_LOCK:
        if _CUDART is not None or _CUDART_ERROR is not None:
            return _CUDART, _CUDART_ERROR
        candidates = []
        try:
            with open("/proc/self/maps", encoding="utf-8") as source:
                for line in source:
                    path = line.split()[-1]
                    if Path(path).name.startswith("libcudart.so"):
                        candidates.append(path)
        except OSError:
            pass
        found = ctypes.util.find_library("cudart")
        if found:
            candidates.append(found)
        roots = [Path(sys.prefix), Path("/usr/local/cuda")]
        roots.extend(Path(value) for key in ("CUDA_HOME", "CUDA_PATH")
                     if (value := os.environ.get(key)))
        directories = [root / subdir for root in roots for subdir in ("lib", "lib64")]
        directories.extend(Path(entry) / subdir for entry in sys.path if entry
                           for subdir in ("nvidia/cuda_runtime/lib", "torch/lib"))
        for directory in directories:
            candidates.extend(str(path) for path in sorted(directory.glob("libcudart.so*")))
        for candidate in dict.fromkeys(candidates):
            try:
                library = ctypes.CDLL(candidate)
                library.cudaDeviceGetPCIBusId.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
                library.cudaDeviceGetPCIBusId.restype = ctypes.c_int
                library.cudaGetErrorString.argtypes = (ctypes.c_int,)
                library.cudaGetErrorString.restype = ctypes.c_char_p
                library.cudaRuntimeGetVersion.argtypes = (ctypes.POINTER(ctypes.c_int),)
                library.cudaRuntimeGetVersion.restype = ctypes.c_int
                _CUDART = library
                return library, None
            except (OSError, AttributeError):
                continue
        _CUDART_ERROR = "CUDA runtime library with PCI identity support is unavailable"
        return None, _CUDART_ERROR


def _runtime_pci_bus_id(index):
    library, error = _cuda_runtime()
    if error:
        return None, error
    bus_id = ctypes.create_string_buffer(32)
    status = library.cudaDeviceGetPCIBusId(bus_id, len(bus_id), index)
    if status:
        detail = library.cudaGetErrorString(status)
        detail = detail.decode("utf-8", errors="replace") if detail else f"status {status}"
        return None, f"CUDA runtime device {index}: {detail}"
    return bus_id.value.decode("ascii"), None


def _pci_key(value):
    try:
        domain, bus, device_function = value.split(":")
        device, function = device_function.split(".")
        return tuple(int(part, 16) for part in (domain, bus, device, function))
    except (ValueError, AttributeError):
        return None


def _smi_index(config):
    """Resolve a CUDA logical ordinal to a verified physical GPU UUID.

    CUDA's default FASTEST_FIRST order and numeric CUDA_VISIBLE_DEVICES entries
    are not NVML indices. A runtime PCI identity is authoritative unless MIG
    makes physical-GPU metrics ambiguous. Explicit unique GPU UUIDs are also
    usable when no CUDA runtime is installed.
    """
    index = config.get("gpu_index", 0)
    if type(index) is not int or index < 0:
        return None, f"invalid CUDA GPU index {index!r}"
    fields = ("uuid", "pci.bus_id", "mig.mode.current")
    rows, error = _smi_query(None, fields)
    if error:
        return None, f"cannot resolve CUDA device {index}: {error}"
    if not rows or any(len(row) != len(fields) for row in rows):
        return None, "cannot resolve CUDA identity: invalid nvidia-smi GPU inventory"
    inventory = [[_value(row[0]), _value(row[1]), row[2].strip()] for row in rows]
    bus_id, runtime_error = _runtime_pci_bus_id(index)
    if bus_id is not None:
        bus_key = _pci_key(bus_id)
        matches = [row for row in inventory if bus_key is not None and _pci_key(row[1]) == bus_key]
        if len(matches) != 1:
            return None, f"CUDA device {index} PCI identity {bus_id!r} does not identify exactly one NVIDIA GPU"
        selected = matches[0]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = [part.strip() for part in visible.split(",")] if visible is not None else None
        token = devices[index] if devices is not None and index < len(devices) else None
        if token and token.startswith("GPU-"):
            seen = set()
            for visible_token in devices[:index + 1]:
                matches = [row for row in inventory
                           if visible_token.startswith("GPU-") and row[0].startswith(visible_token)]
                if (len(matches) != 1 or matches[0][0] in seen
                        or matches[0][2].lower() not in {"disabled", "n/a", "[n/a]", "not supported", "[not supported]"}):
                    return None, ("CUDA_VISIBLE_DEVICES does not establish an unambiguous "
                                  "non-MIG UUID-only device order; a working CUDA runtime is required")
                selected = matches[0]
                seen.add(selected[0])
        elif len(inventory) == 1 and index == 0 and (devices is None or devices == ["0"]):
            selected = inventory[0]
        else:
            return None, (f"ambiguous CUDA device {index}: {runtime_error}; "
                          "numeric CUDA ordinals cannot safely be mapped to nvidia-smi indices "
                          f"(CUDA_VISIBLE_DEVICES={visible!r}, "
                          f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')!r})")
    if selected[2].lower() not in {"disabled", "n/a", "[n/a]", "not supported", "[not supported]"}:
        return None, ("MIG is enabled or its state is unrecognized; physical-GPU "
                      "telemetry cannot be attributed to the configured CUDA device")
    if not selected[0].startswith("GPU-"):
        return None, "nvidia-smi did not provide a usable physical GPU UUID"
    return selected[0], None


def _gpu_row(config, fields, selector=None):
    if selector is None:
        selector, error = _smi_index(config)
        if error:
            return None, error
    rows, error = _smi_query(selector, fields)
    if error:
        return None, f"GPU {selector}: {error}"
    if not rows or len(rows) != 1 or len(rows[0]) != len(fields):
        return None, f"GPU {selector}: nvidia-smi did not return exactly one usable GPU row"
    return [_value(value) for value in rows[0]], None


def _optional_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip().rstrip("\0")
    except (OSError, UnicodeError):
        return None


def _runtime_versions():
    versions = {}
    torch = sys.modules.get("torch")
    if torch is not None:
        versions["torch"] = getattr(torch, "__version__", None)
        versions["torch_cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        if hasattr(torch, "get_num_threads"):
            versions["torch_num_threads"] = torch.get_num_threads()
        if hasattr(torch, "get_num_interop_threads"):
            versions["torch_num_interop_threads"] = torch.get_num_interop_threads()
    for name in ("onnxruntime", "tensorrt"):
        module = sys.modules.get(name)
        if module is not None:
            versions[name] = getattr(module, "__version__", None)
    if _CUDART is not None:
        version = ctypes.c_int()
        if _CUDART.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
            versions["cuda_runtime"] = version.value
    return versions


def environment(config):
    """Return host, process, runtime, toolkit, Jetson, and NVIDIA GPU metadata."""
    packages = {}
    for name in ("torch", "onnxruntime", "onnxruntime-gpu", "tensorrt", "numpy", "psutil",
                 "nvidia-cuda-runtime-cu11", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    _, host_metrics, host_errors = _system_sample(None)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    status, status_error = _process_status()
    cpu_model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"model name", "Hardware", "Processor"}:
                    cpu_model = value.strip()
                    break
    except OSError:
        pass
    result = {
        "hostname": platform.node(), "platform": platform.platform(),
        "platform_label": config.get("platform_label"), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_model": cpu_model,
        "cpu_logical_count": os.cpu_count(), "cpu_affinity": affinity,
        "cpu_available_count": len(affinity) if affinity is not None else os.cpu_count(),
        "memory_total_mb": host_metrics[1], "memory_available_mb": host_metrics[2],
        "process_rss_mb": host_metrics[3], "process_rss_peak_mb": host_metrics[4],
        "process_num_threads": status.get("Threads"),
        "thread_settings": {
            "configured_cpu_threads": config.get("cpu_threads", 1),
            "configured_deterministic": config.get("deterministic", True),
            "environment": {key: os.environ.get(key) for key in _THREAD_ENV},
        },
        "python": sys.version, "python_executable": sys.executable, "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "device": config.get("device", "cpu"), "gpu_index": config.get("gpu_index", 0),
        "gpus": [], "selected_gpu": None, "gpu_error": None,
        "gpu_inventory_scope": "physical GPUs reported by nvidia-smi, not CUDA ordinal order",
        "jetson_model": _optional_text("/proc/device-tree/model"),
        "jetson_compatible": _optional_text("/proc/device-tree/compatible"),
        "jetson_l4t_release": _optional_text("/etc/nv_tegra_release"),
        "environment_errors": list(dict.fromkeys(host_errors + ([status_error] if status_error else []))),
    }
    fields = ("index", "uuid", "name", "driver_version", "memory.total", "pci.bus_id")
    rows, error = _smi_query(None, fields)
    if error:
        result["gpu_error"] = error
    else:
        for row in rows:
            if len(row) == len(fields):
                result["gpus"].append(dict(zip(
                    ("index", "uuid", "name", "driver_version", "memory_total_mb", "pci_bus_id"),
                    map(_value, row))))
            else:
                result["environment_errors"].append("nvidia-smi returned a malformed GPU inventory row")
        if config.get("device", "cpu") == "cuda":
            selector, error = _smi_index(config)
            if error:
                result["gpu_error"] = error
            else:
                result["selected_gpu"] = next(
                    (gpu for gpu in result["gpus"] if gpu["uuid"] == selector), None)
                if result["selected_gpu"] is None:
                    result["gpu_error"] = "resolved CUDA GPU is absent from the recorded inventory"
    toolkit_roots = [Path(value) for key in ("CUDA_HOME", "CUDA_PATH")
                     if (value := os.environ.get(key))]
    toolkit_roots.append(Path("/usr/local/cuda"))
    result["cuda_toolkit_version_file"] = None
    for root in toolkit_roots:
        for filename in ("version.json", "version.txt"):
            text = _optional_text(root / filename)
            if text:
                result["cuda_toolkit_version_file"] = {"path": str(root / filename), "contents": text}
                break
        if result["cuda_toolkit_version_file"] is not None:
            break
    result["nvcc_version"] = None
    if executable := shutil.which("nvcc"):
        text, error = _command([executable, "--version"])
        result["nvcc_version"] = text.strip() if text is not None else None
        if error:
            result["environment_errors"].append(f"nvcc version: {error}")
    result["runtime_versions"] = _runtime_versions()
    return result


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_availability(config):
    """Validate availability settings without loading CUDA or querying a GPU."""
    availability = config.get("availability", {})
    if not isinstance(availability, dict):
        raise ValueError("availability must be an object")
    allowed = {"min_free_vram_mb", "max_gpu_utilization_percent", "timeout_s", "poll_s"}
    unknown = set(availability) - allowed
    if unknown:
        raise ValueError(f"unknown availability keys: {sorted(unknown, key=str)}")
    for key, value in availability.items():
        if not _finite_number(value):
            raise ValueError(f"availability.{key} must be a finite number")
        if value < 0 or (key == "poll_s" and value == 0):
            raise ValueError(f"availability.{key} must be {'positive' if key == 'poll_s' else 'nonnegative'}")
        if key == "max_gpu_utilization_percent" and value > 100:
            raise ValueError("availability.max_gpu_utilization_percent must be between 0 and 100")
    if ({"min_free_vram_mb", "max_gpu_utilization_percent"} & availability.keys()
            and config.get("device", "cpu") != "cuda"):
        raise ValueError("GPU availability thresholds require device='cuda'")


def availability_check(config):
    """Wait for validated thresholds on the verified CUDA device; never modify it."""
    validate_availability(config)
    availability = config.get("availability", {})
    minimum = availability.get("min_free_vram_mb")
    maximum = availability.get("max_gpu_utilization_percent")
    if minimum is None and maximum is None:
        return
    timeout = float(availability.get("timeout_s", 60))
    poll = float(availability.get("poll_s", 1))
    deadline = time.monotonic() + timeout
    selector, error = _smi_index(config)
    if error:
        raise RuntimeError(f"GPU availability cannot identify the configured CUDA device: {error}")
    last_reason = "GPU metrics unavailable"
    while True:
        row, error = _gpu_row(config, ("memory.free", "utilization.gpu"), selector)
        if error:
            last_reason = error
        else:
            reasons = []
            for label, value, threshold, meets in (
                ("free VRAM (MB)", row[0], minimum, lambda v, t: v >= t),
                ("GPU utilization (%)", row[1], maximum, lambda v, t: v <= t),
            ):
                if threshold is None:
                    continue
                try:
                    number = float(value)
                    if not math.isfinite(number) or number < 0 or not meets(number, float(threshold)):
                        reasons.append(f"{label}={number:g} does not meet threshold {threshold}")
                except (ValueError, TypeError):
                    reasons.append(f"{label} is unavailable ({value or 'blank'})")
            if not reasons:
                return
            last_reason = "; ".join(reasons)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"GPU availability timed out after {timeout:g}s: {last_reason}")
        time.sleep(min(poll, remaining, threading.TIMEOUT_MAX))


def _process_status():
    wanted = {"VmRSS", "VmHWM", "Threads"}
    try:
        with open("/proc/self/status", encoding="utf-8") as source:
            return {parts[0].rstrip(":"): int(parts[1]) for line in source
                    if len(parts := line.split()) >= 2 and parts[0].rstrip(":") in wanted}, None
    except (OSError, ValueError) as exc:
        return {}, f"process status: {exc}"


def _system_sample(previous):
    cpu = ""
    current = None
    errors = []
    try:
        with open("/proc/stat", encoding="utf-8") as source:
            values = [int(value) for value in source.readline().split()[1:]]
        # guest and guest_nice are already included in user and nice.
        current = (sum(values[:8]), values[3] + (values[4] if len(values) > 4 else 0))
        if previous is not None and current[0] > previous[0]:
            total_delta, idle_delta = current[0] - previous[0], current[1] - previous[1]
            if 0 <= idle_delta <= total_delta:
                cpu = 100.0 * (1.0 - idle_delta / total_delta)
            else:
                errors.append("CPU counters changed inconsistently; utilization is unavailable")
    except (OSError, ValueError, IndexError) as exc:
        errors.append(f"CPU utilization: {exc}")
    memory_total = memory_available = ""
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            memory = {parts[0].rstrip(":"): int(parts[1]) for line in source if len(parts := line.split()) >= 2}
        memory_total = memory["MemTotal"] / 1024 if "MemTotal" in memory else ""
        memory_available = memory["MemAvailable"] / 1024 if "MemAvailable" in memory else ""
        if memory_total == "" or memory_available == "":
            errors.append("host RAM total or availability is unsupported")
    except (OSError, ValueError) as exc:
        errors.append(f"host RAM: {exc}")
    status, error = _process_status()
    if error:
        errors.append(error)
    rss = status["VmRSS"] / 1024 if "VmRSS" in status else ""
    peak = status["VmHWM"] / 1024 if "VmHWM" in status else ""
    if rss == "" or peak == "":
        errors.append("process RSS or RSS high watermark is unsupported")
    return current, (cpu, memory_total, memory_available, rss, peak), errors


class Telemetry:
    """Sample host and verified physical-GPU metrics; surface all sampler errors.

    Each row timestamps the beginning of its own collection. phase_end and
    collection_duration_ms expose samples that straddle a benchmark transition.
    Blank values mean unsupported/unavailable, never zero.
    """

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        interval = config.get("telemetry_interval_s", 1.0)
        if not _finite_number(interval) or interval <= 0:
            raise ValueError("telemetry_interval_s must be a positive finite number")
        self.interval = float(interval)
        self.errors = []
        self.process_rss_peak_mb = None
        self._phase = "initializing"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._gpu_selector = None
        self._gpu_identity_error = None

    def set_phase(self, phase):
        if not isinstance(phase, str) or not phase:
            raise ValueError("telemetry phase must be a nonempty string")
        with self._phase_lock:
            self._phase = phase

    def _get_phase(self):
        with self._phase_lock:
            return self._phase

    def _record_error(self, error):
        if error and error not in self.errors:
            self.errors.append(error)

    def start(self):
        if self._thread is not None:
            raise RuntimeError("telemetry has already been started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timing_columns = ("timestamp", "timestamp_ns", "monotonic_ns", "phase", "phase_end", "collection_duration_ms")
        # Headers are created synchronously, including on systems without GPUs.
        with (self.output_dir / "gpu_telemetry.csv").open("w", newline="", encoding="utf-8") as gpu:
            csv.writer(gpu).writerow((*timing_columns, "gpu_index", "gpu_uuid", *_GPU_COLUMNS,
                                     "other_compute_processes", "error"))
        with (self.output_dir / "system_telemetry.csv").open("w", newline="", encoding="utf-8") as system:
            csv.writer(system).writerow((*timing_columns, "cpu_utilization_percent", "memory_total_mb",
                                        "memory_available_mb", "process_rss_mb", "process_rss_peak_mb", "error"))
        if self.config.get("device", "cpu") == "cuda":
            self._gpu_selector, self._gpu_identity_error = _smi_index(self.config)
            self._record_error(self._gpu_identity_error)
        self._thread = threading.Thread(target=self._sample, name="rover-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        status, error = _process_status()
        self._record_error(error)
        if "VmHWM" in status:
            self.process_rss_peak_mb = max(self.process_rss_peak_mb or 0, status["VmHWM"] / 1024)
        else:
            self._record_error("process RSS high watermark is unsupported")

    def _sample(self):
        try:
            self._sample_loop()
        except Exception as exc:
            self._record_error(f"telemetry sampler stopped: {type(exc).__name__}: {exc}")

    def _sample_loop(self):
        previous = None
        with (self.output_dir / "gpu_telemetry.csv").open("a", newline="", encoding="utf-8") as gpu, (self.output_dir / "system_telemetry.csv").open("a", newline="", encoding="utf-8") as system:
            gpu_writer, system_writer = csv.writer(gpu), csv.writer(system)
            while not self._stop.is_set():
                phase = self._get_phase()
                timing = _sample_time()
                previous, metrics, errors = _system_sample(previous)
                if metrics[4] != "":
                    self.process_rss_peak_mb = max(self.process_rss_peak_mb or 0, metrics[4])
                for error in errors:
                    self._record_error(error)
                system_writer.writerow((*timing, phase, self._get_phase(),
                                        (time.monotonic_ns() - timing[2]) / 1_000_000,
                                        *metrics, "; ".join(errors)))
                system.flush()
                if self.config.get("device", "cpu") == "cuda" and not self._stop.is_set():
                    phase = self._get_phase()
                    timing = _sample_time()
                    errors = []
                    values = None
                    if self._gpu_identity_error:
                        errors.append(self._gpu_identity_error)
                    else:
                        values, error = _gpu_row(self.config, _GPU_FIELDS, self._gpu_selector)
                        if error:
                            errors.append(error)
                    processes = ""
                    if values is not None:
                        for offset, value in enumerate(values):
                            try:
                                number = float(value)
                                valid = math.isfinite(number) and number >= 0 and (offset != 0 or number <= 100)
                            except ValueError:
                                valid = False
                            if not valid:
                                values[offset] = ""
                                errors.append(f"GPU {_GPU_FIELDS[offset]} is unsupported or unavailable")
                    if values is not None and not self._stop.is_set():
                        rows, error = _smi_query(self._gpu_selector,
                                                 ("pid", "process_name", "used_gpu_memory"), "compute-apps")
                        if error:
                            errors.append(f"GPU compute processes: {error}")
                        elif any(len(row) != 3 for row in rows):
                            errors.append("GPU compute processes: malformed nvidia-smi rows")
                        else:
                            processes = json.dumps([
                                {"pid": _value(row[0]), "name": _value(row[1]), "vram_mb": _value(row[2])}
                                for row in rows if row[0].strip() != str(os.getpid())
                            ], separators=(",", ":"))
                    for error in errors:
                        self._record_error(error)
                    gpu_writer.writerow((*timing, phase, self._get_phase(),
                                         (time.monotonic_ns() - timing[2]) / 1_000_000,
                                         self.config.get("gpu_index", 0), self._gpu_selector or "",
                                         *(values or [""] * len(_GPU_FIELDS)), processes, "; ".join(errors)))
                    gpu.flush()
                self._stop.wait(min(self.interval, threading.TIMEOUT_MAX))
