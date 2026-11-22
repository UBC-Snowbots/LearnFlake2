"""Run lifecycle, timing summaries, and durable result packaging."""
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import tarfile
import time
import traceback
import uuid

from .telemetry import Telemetry, availability_check, environment, validate_availability
from . import __version__

SCHEMA_VERSION = "1.0"
STAGES = ("preprocess_ms", "inference_wall_ms", "inference_gpu_ms", "postprocess_ms", "end_to_end_ms")
DEFAULTS = {
    "backend": "pytorch", "model": "smoke", "device": "cpu", "gpu_index": 0,
    "input_shape": [1, 3, 224, 224], "precision": "fp32", "warmup": 10,
    "iterations": 100, "seed": 0, "output_root": "runs", "platform_label": "desktop",
    "telemetry_interval_s": 1.0, "availability": {}, "deterministic": True, "cpu_threads": 1,
    "model_manifest": None,
}


class Interrupted(BaseException):
    pass


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("configuration must be a JSON object")
    unknown = set(data) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **data}
    if cfg["backend"] not in ("pytorch", "onnxruntime", "tensorrt"):
        raise ValueError("backend must be pytorch, onnxruntime, or tensorrt")
    if cfg["device"] not in ("cpu", "cuda") or cfg["precision"] not in ("fp32", "fp16"):
        raise ValueError("device must be cpu/cuda and precision must be fp32/fp16")
    if not isinstance(cfg["model"], str) or not cfg["model"]:
        raise ValueError("model must be smoke or a model file path")
    if cfg["model"] != "smoke":
        model = Path(cfg["model"]).expanduser()
        cfg["model"] = str((path.parent / model).resolve()) if not model.is_absolute() else str(model.resolve())
    if cfg["model_manifest"] is not None:
        if not isinstance(cfg["model_manifest"], str) or not cfg["model_manifest"]:
            raise ValueError("model_manifest must be a path string or null")
        manifest = Path(cfg["model_manifest"]).expanduser()
        cfg["model_manifest"] = str((path.parent / manifest).resolve()) if not manifest.is_absolute() else str(manifest.resolve())
    if not isinstance(cfg["input_shape"], list) or not cfg["input_shape"] or any(type(n) is not int or n <= 0 for n in cfg["input_shape"]):
        raise ValueError("input_shape must be a nonempty list of positive integers including batch")
    for key, minimum in (("warmup", 0), ("iterations", 1), ("gpu_index", 0), ("seed", 0), ("cpu_threads", 1)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(cfg["deterministic"]) is not bool:
        raise ValueError("deterministic must be a boolean")
    if not isinstance(cfg["availability"], dict):
        raise ValueError("availability must be an object")
    validate_availability(cfg)
    if type(cfg["telemetry_interval_s"]) not in (int, float) or not math.isfinite(cfg["telemetry_interval_s"]) or cfg["telemetry_interval_s"] <= 0:
        raise ValueError("telemetry_interval_s must be a positive finite number")
    if not isinstance(cfg["output_root"], str) or not cfg["output_root"] or not isinstance(cfg["platform_label"], str):
        raise ValueError("output_root and platform_label must be strings")
    # Resolve relative output paths against the configuration location, not the queue's working directory.
    out = Path(cfg["output_root"]).expanduser()
    cfg["output_root"] = str((path.parent / out).resolve()) if not out.is_absolute() else str(out.resolve())
    return cfg


def load_model_manifest(cfg):
    """Bind optional provenance to the exact artifact and tensor shape used."""
    if cfg["model_manifest"] is None:
        return None
    with Path(cfg["model_manifest"]).open(encoding="utf-8") as f:
        document = json.load(f)
    if not isinstance(document, dict) or not isinstance(document.get("export"), dict):
        raise ValueError("model_manifest must contain an export object")
    exported = document["export"]
    expected = exported.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("model_manifest export.sha256 must be a lowercase SHA-256 digest")
    if exported.get("input_shape") != cfg["input_shape"]:
        raise ValueError("model_manifest input_shape does not match benchmark configuration")
    digest = hashlib.sha256()
    with Path(cfg["model"]).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("model_manifest SHA-256 does not match model artifact")
    return document


def dump(path, obj):
    # A replace prevents consumers from observing a half-written JSON file.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def distribution(values):
    vals = sorted(x for x in values if x is not None and math.isfinite(x))
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None}
    def percentile(p):
        pos = (len(vals) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    return {"count": len(vals), "mean": sum(vals) / len(vals), "median": percentile(.5),
            "p90": percentile(.90), "p95": percentile(.95), "p99": percentile(.99)}


def telemetry_summary(run):
    """Summarize sampled steady-state telemetry; never infer unsampled peaks."""
    result = {}
    for filename in ("gpu_telemetry.csv", "system_telemetry.csv"):
        fields = {}
        phase_counts = {}
        other_processes_seen = False
        process_observations = 0
        with (run / filename).open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                phase = row.get("phase", "unknown")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if phase != "steady" or row.get("phase_end", phase) != phase:
                    continue
                if row.get("other_compute_processes"):
                    try:
                        processes = json.loads(row["other_compute_processes"])
                        if isinstance(processes, list):
                            process_observations += 1
                            other_processes_seen |= bool(processes)
                    except ValueError:
                        pass
                for key, value in row.items():
                    if not key or not key.endswith(("_percent", "_mb", "_w", "_c", "_mhz")):
                        continue
                    try:
                        number = float(value)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(number):
                        fields.setdefault(key, []).append(number)
        result[filename.removesuffix(".csv")] = {
            "phase_sample_counts": phase_counts,
            "steady_other_compute_processes_seen": other_processes_seen if process_observations else None,
            "steady_sampled_metrics": {
                key: {"count": len(values), "mean": sum(values) / len(values), "min": min(values), "max": max(values)}
                for key, values in fields.items()
            },
            "scope": "Samples entirely within steady phase only; utilization fields may themselves cover driver sampling windows; maxima are not true allocation or power peaks",
        }
    return result


def report(run, cfg, summary, info):
    metrics = summary.get("metrics", {})
    rows = ["# Benchmark report", "", f"- Status: {summary['status']}",
            f"- Platform label: {cfg['platform_label']}",
            f"- Backend / model / device: {cfg['backend']} / {cfg['model']} / {cfg['device']}",
            f"- Input shape / precision: {cfg['input_shape']} / {cfg['precision']}",
            f"- Completed warmup / steady iterations: {summary['warmup_completed']} / {summary['completed_iterations']}",
            f"- Error: {summary.get('error') or 'none'}", "", "## Steady-state latencies (ms)", "",
            "| Stage | Mean | Median | P90 | P95 | P99 |", "|---|---:|---:|---:|---:|---:|"]
    for name in STAGES:
        d = metrics.get(name, {})
        fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
        rows.append(f"| {name} | " + " | ".join(fmt(d.get(k)) for k in ("mean", "median", "p90", "p95", "p99")) + " |")
    rows += ["", f"- Inferences/s (batches): {metrics.get('inferences_per_second')}",
             f"- Effective FPS (batch elements/s): {metrics.get('effective_fps')}",
             "- Throughput uses measured steady-loop wall time including measurement and sample-writing overhead.",
             "- Warmup measurements are in latency_samples.csv; steady-state statistics exclude them.",
             "- Synthetic input and backend-specific pipeline boundaries are described in model_info.json; this is not camera-to-decision latency.",
             "- GPU utilization and free memory do not establish exclusive GPU access; telemetry is sampled and may miss short contention.",
             "- Desktop resource limits are not Jetson emulation; only measurements taken on Jetson establish Jetson performance.",
             "- No gesture accuracy or detection latency was measured.", ""]
    rows += ["## Model and memory", "",
             f"- Parameters: {info.get('parameter_count')}",
             f"- Serialized bytes: {info.get('serialized_bytes')}",
             f"- Estimated FLOPs per configured batch: {info.get('flops')}",
             f"- FLOP scope: {info.get('flops_reason')}", "",
             "```json", json.dumps(summary["memory"], indent=2, sort_keys=True), "```", "",
             "## Telemetry", "",
             "Steady-phase samples only. Empty metrics mean no supported samples, not zero utilization.", "",
             "```json", json.dumps(summary.get("telemetry", {}), indent=2, sort_keys=True), "```", "",
             "## Warnings", ""]
    rows.extend(f"- {warning}" for warning in summary.get("warnings", []))
    if not summary.get("warnings"):
        rows.append("- None recorded; this does not establish uncontended execution.")
    (run / "report.md").write_text("\n".join(rows), encoding="utf-8")


def package(run):
    manifest = run / "manifest.sha256"
    with manifest.open("w", encoding="utf-8") as out:
        for path in sorted(run.iterdir()):
            if path.is_file() and path != manifest:
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
                h = digest.hexdigest()
                out.write(f"{h}  {path.name}\n")
    archive = run.parent / (run.name + ".tar.gz")
    temp = run.parent / (run.name + ".tar.gz.tmp")
    with tarfile.open(temp, "w:gz") as tar:
        tar.add(run, arcname=run.name)
    os.replace(temp, archive)
    return archive


def run_benchmark(cfg):
    from .backends import create_backend
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    run = root / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    run.mkdir()
    dump(run / "config.json", cfg)
    dump(run / "environment.json", {})
    dump(run / "model_info.json", {"status": "not_initialized"})
    summary = {"schema_version": SCHEMA_VERSION, "status": "running", "error": None,
               "warmup_completed": 0, "completed_iterations": 0, "metrics": {}, "memory": {},
               "warnings": [], "warmup_metrics": {}}
    dump(run / "summary.json", summary)
    logger = logging.getLogger("orion_bench.run")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(run / "benchmark.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    telemetry = None
    backend = None
    info = {}
    samples = []
    warmup_samples = []
    loop_seconds = None
    old_handlers = {}
    committing = False
    pending_signal = None
    def interrupt(signum, frame):
        nonlocal pending_signal
        if committing:
            pending_signal = signum
            return
        raise Interrupted(f"received signal {signum}")
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, interrupt)
    try:
        logger.info("Starting run %s", run.name)
        host = environment(cfg)
        host["harness"] = {
            "version": __version__,
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
        dump(run / "environment.json", host)
        provenance = load_model_manifest(cfg)
        if provenance is not None:
            dump(run / "model_provenance.json", provenance)
            dump(run / "model_info.json", {"status": "not_initialized", "provenance": provenance})
        telemetry = Telemetry(cfg, run)
        telemetry.set_phase("availability")
        telemetry.start()
        availability_check(cfg)
        telemetry.set_phase("initialization")
        backend = create_backend(cfg)
        info = backend.model_info()
        if provenance is not None:
            if info["model_sha256"] != provenance["export"]["sha256"]:
                raise ValueError("model artifact changed after provenance validation")
            info["provenance"] = provenance
        dump(run / "model_info.json", info)
        with (run / "latency_samples.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["phase", "iteration", *STAGES])
            writer.writeheader()
            f.flush()
            for phase, count in (("warmup", cfg["warmup"]), ("steady", cfg["iterations"])):
                telemetry.set_phase(phase)
                if phase == "steady" and hasattr(backend, "reset_peak_memory"):
                    backend.reset_peak_memory()
                start = time.perf_counter() if phase == "steady" else None
                try:
                    for index in range(count):
                        sample = backend.sample()
                        row = {"phase": phase, "iteration": index, **{key: sample.get(key) for key in STAGES}}
                        # Commit a completed row and its statistics together.
                        committing = True
                        try:
                            writer.writerow(row)
                            f.flush()
                            summary["warmup_completed" if phase == "warmup" else "completed_iterations"] += 1
                            if phase == "steady":
                                samples.append(row)
                            else:
                                warmup_samples.append(row)
                        finally:
                            committing = False
                            if pending_signal is not None:
                                raise Interrupted(f"received signal {pending_signal}")
                finally:
                    if start is not None:
                        loop_seconds = time.perf_counter() - start
        summary["status"] = "completed"
        logger.info("Completed %s iterations", summary["completed_iterations"])
    except Interrupted as exc:
        summary["status"] = "interrupted"
        summary["error"] = str(exc)
        logger.warning("%s", exc)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Benchmark failed\n%s", traceback.format_exc())
    finally:
        # During finalization, subsequent termination signals do not interrupt artifact writing.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if telemetry is not None:
            telemetry.set_phase("finalization")
        if backend is not None:
            try:
                summary["memory"] = backend.memory_info()
            except Exception:
                logger.exception("Failed to obtain backend memory metrics")
            try:
                backend.close()
            except Exception:
                logger.exception("Backend cleanup failed")
        if telemetry is not None:
            try:
                telemetry.stop()
            except Exception:
                logger.exception("Telemetry cleanup failed")
            summary["warnings"].extend(telemetry.errors)
        try:
            import resource
            summary["memory"]["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            summary["memory"]["process_peak_rss_scope"] = "Linux process high watermark since start, including initialization and profiler"
        except (ImportError, AttributeError, OSError):
            summary["memory"]["process_peak_rss_bytes"] = None
        # A failure before sampling still yields a machine-readable empty sample file.
        if not (run / "latency_samples.csv").exists():
            with (run / "latency_samples.csv").open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=["phase", "iteration", *STAGES]).writeheader()
        for name in ("gpu_telemetry.csv", "system_telemetry.csv"):
            if not (run / name).exists():
                (run / name).write_text("timestamp\n", encoding="utf-8")
        metrics = {name: distribution([s[name] for s in samples]) for name in STAGES}
        rate = len(samples) / loop_seconds if loop_seconds and samples else None
        metrics.update({"measured_loop_seconds": loop_seconds, "inferences_per_second": rate,
                        "effective_fps": rate * cfg["input_shape"][0] if rate is not None else None})
        summary["metrics"] = metrics
        summary["warmup_metrics"] = {name: distribution([s[name] for s in warmup_samples]) for name in STAGES}
        summary["telemetry"] = telemetry_summary(run)
        dump(run / "summary.json", summary)
        report(run, cfg, summary, info)
        logger.info("Final status: %s", summary["status"])
        logger.removeHandler(handler)
        handler.close()
        archive = package(run)
        for sig, old in old_handlers.items():
            signal.signal(sig, old)
    print(f"run_dir={run}\narchive={archive}\nstatus={summary['status']}", flush=True)
    return 0 if summary["status"] == "completed" else (130 if summary["status"] == "interrupted" else 1)
