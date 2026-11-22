#!/usr/bin/env python3
"""Offline CLI/artifact smoke; requires torch, optionally CUDA and Task Spooler."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("preprocess_ms", "inference_wall_ms", "inference_gpu_ms", "postprocess_ms", "end_to_end_ms")
REQUIRED = {
    "config.json", "environment.json", "model_info.json", "summary.json",
    "latency_samples.csv", "gpu_telemetry.csv", "system_telemetry.csv",
    "benchmark.log", "report.md", "manifest.sha256",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["phase", "iteration", *STAGES], "latency CSV columns changed")
        return list(reader)


def validate_run(run, status, config):
    require(run.is_dir(), f"missing run directory: {run}")
    files = {p.name for p in run.iterdir() if p.is_file()}
    require(REQUIRED <= files, f"missing artifacts: {REQUIRED - files}")
    manifest = {}
    for line in (run / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in manifest and name in files, "invalid or duplicate manifest entry")
        manifest[name] = digest
        require(hashlib.sha256((run / name).read_bytes()).hexdigest() == digest, f"hash mismatch: {name}")
    require(set(manifest) == files - {"manifest.sha256"}, "manifest coverage mismatch")
    archive_path = run.with_name(run.name + ".tar.gz")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        require(set(members) == {f"{run.name}/{name}" for name in files}, "archive file coverage mismatch")
        for name in files:
            with archive.extractfile(members[f"{run.name}/{name}"]) as stream:
                require(stream.read() == (run / name).read_bytes(), f"archive differs: {name}")
    effective = load_json(run / "config.json")
    for key, value in config.items():
        require(effective[key] == value, f"effective config mismatch: {key}")
    summary = load_json(run / "summary.json")
    require(summary["schema_version"] == "1.0", "unexpected summary schema")
    require(summary["status"] == status, f"expected {status}, got {summary['status']}")
    require(bool(summary["error"]) == (status != "completed"), "status/error mismatch")
    require(isinstance(load_json(run / "environment.json"), dict), "invalid environment")
    info = load_json(run / "model_info.json")
    rows = sample_rows(run / "latency_samples.csv")
    require(all(row["phase"] in ("warmup", "steady") for row in rows), "unknown sample phase")
    warmup = [row for row in rows if row["phase"] == "warmup"]
    steady = [row for row in rows if row["phase"] == "steady"]
    require(len(warmup) == summary["warmup_completed"], "warmup count mismatch")
    require(len(steady) == summary["completed_iterations"], "steady count mismatch")
    for phase in (warmup, steady):
        require([int(row["iteration"]) for row in phase] == list(range(len(phase))), "sample index discontinuity")
    if status == "completed":
        require(len(warmup) == config["warmup"] and len(steady) == config["iterations"], "incomplete successful run")
        require(info["backend"] == "pytorch" and info["input_shape"] == config["input_shape"], "incorrect model metadata")
    elif status == "failed":
        require(not rows, "model-load failure unexpectedly sampled")
    else:
        require(0 < len(steady) < config["iterations"], "interruption did not preserve partial steady progress")
    for row in rows:
        for stage in STAGES:
            if stage == "inference_gpu_ms" and config["device"] == "cpu":
                require(row[stage] == "", "CPU GPU timing must be unavailable")
            else:
                require(row[stage] != "" and math.isfinite(float(row[stage])) and float(row[stage]) >= 0,
                        f"invalid measured duration: {stage}")
    metrics = summary["metrics"]
    for phase_name, phase_rows, distributions in (
        ("warmup", warmup, summary["warmup_metrics"]), ("steady", steady, metrics),
    ):
        for stage in STAGES:
            values = sorted(float(row[stage]) for row in phase_rows if row[stage] != "")
            distribution = distributions[stage]
            require(distribution["count"] == len(values), f"{phase_name} distribution count mismatch: {stage}")
            expected = {"mean": sum(values) / len(values)} if values else {"mean": None}
            for name, fraction in (("median", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
                if values:
                    position = (len(values) - 1) * fraction
                    lo = int(position)
                    hi = min(lo + 1, len(values) - 1)
                    expected[name] = values[lo] + (values[hi] - values[lo]) * (position - lo)
                else:
                    expected[name] = None
            for name, value in expected.items():
                actual = distribution[name]
                require(actual is None if value is None else actual is not None and math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12),
                        f"{phase_name} distribution mismatch: {stage}.{name}")
    if steady:
        seconds = metrics["measured_loop_seconds"]
        require(math.isfinite(seconds) and seconds > 0, "invalid loop duration")
        require(math.isclose(metrics["inferences_per_second"], len(steady) / seconds), "throughput calculation mismatch")
        require(math.isclose(metrics["effective_fps"], metrics["inferences_per_second"] * config["input_shape"][0]), "batch throughput mismatch")
    return summary, archive_path


class Launcher:
    def __init__(self, environment, queued, timeout):
        self.environment = environment
        self.queued = queued
        self.timeout = timeout

    def run(self, config_path, config, interrupt=False, expect_failure=False):
        command = [sys.executable, "-m", "orion_bench", "run", "--config", str(config_path)]
        if self.queued:
            command = ["tsp", "-f", *command]
        log_path = config_path.with_suffix(".console.log")
        with log_path.open("w+", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.timeout
            sent = False
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"CLI timed out; see {log_path}")
                    if interrupt and not sent:
                        for samples in Path(config["output_root"]).glob("*/latency_samples.csv"):
                            # Readiness is observed data, not an assumed startup delay.
                            with samples.open("rb") as stream:
                                progressed = any(line.startswith(b"steady,") and line.endswith(b"\n") for line in stream)
                            if progressed:
                                if self.queued:
                                    subprocess.run(["tsp", "-k"], env=self.environment, check=True, capture_output=True, timeout=10)
                                else:
                                    os.kill(process.pid, signal.SIGTERM)
                                sent = True
                                break
                    time.sleep(.01)
                code = process.wait()
            finally:
                if process.poll() is None:
                    # Only this invocation's private child/queue is ever terminated.
                    if self.queued:
                        subprocess.run(["tsp", "-k"], env=self.environment, capture_output=True, timeout=10, check=False)
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            log.seek(0)
            console = log.read()
        require(sent if interrupt else True, "CLI exited before steady progress could be interrupted")
        require((code != 0) if interrupt or expect_failure else (code == 0), f"unexpected CLI exit {code}:\n{console}")
        runs = [p for p in Path(config["output_root"]).iterdir() if p.is_dir()]
        require(len(runs) == 1, f"expected exactly one run: {runs}\n{console}")
        return runs[0]


def smoke(output, args, environment):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    class ConvWithScale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3)

        def forward(self, x):
            # Event profiling can count this elementwise work but omit the
            # scripted convolution; the estimate must not stop at that subtotal.
            return self.conv(x) * 2.0 + 1.0

    fixture = output / "conv.pt"
    torch.jit.trace(ConvWithScale().eval(), torch.zeros(2, 3, 8, 8)).save(str(fixture))
    provenance = {
        "source": {"kind": "local-smoke-fixture"},
        "export": {"sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(), "input_shape": [2, 3, 8, 8]},
    }
    manifest_path = output / "model_manifest.json"
    manifest_path.write_text(json.dumps(provenance), encoding="utf-8")
    mismatched_path = output / "mismatched_manifest.json"
    mismatched_path.write_text(json.dumps({**provenance, "export": {**provenance["export"], "sha256": "0" * 64}}), encoding="utf-8")
    try:
        from torch.utils.flop_counter import FlopCounterMode
        dispatch_counting_available = True
    except ImportError:
        dispatch_counting_available = False
    launcher = Launcher(environment, args.tsp, args.timeout)
    summaries = []
    sources = []
    for name, status in (("success", "completed"), ("torchscript", "completed"), ("manifest_mismatch", "failed"), ("failure", "failed"), ("interrupt", "interrupted")):
        config = {
            "backend": "pytorch", "model": "smoke", "device": args.device,
            "gpu_index": args.gpu_index, "input_shape": [2, 3, 8, 8], "precision": "fp32",
            "warmup": 3, "iterations": 7, "seed": 42, "deterministic": True,
            "cpu_threads": 1, "output_root": str(output / name),
            "platform_label": "cli-smoke", "telemetry_interval_s": .1, "availability": {},
        }
        if name in ("torchscript", "manifest_mismatch"):
            config["model"] = str(fixture)
            config["model_manifest"] = str(manifest_path if name == "torchscript" else mismatched_path)
        if name == "failure":
            config["model"] = str(output / "deliberately-missing-model.pt")
        if name == "interrupt":
            config["iterations"] = 10_000_000
        config_path = output / f"{name}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        run = launcher.run(config_path, config, interrupt=name == "interrupt", expect_failure=status == "failed")
        summary, archive = validate_run(run, status, config)
        if name == "torchscript":
            info = load_json(run / "model_info.json")
            require(info["parameter_count"] == 112, "TorchScript parameter count is incorrect")
            require(info["model_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest(), "model fingerprint mismatch")
            require(info["provenance"]["export"]["sha256"] == info["model_sha256"], "provenance is not bound to loaded model")
            require((run / "model_provenance.json").is_file(), "successful provenance run lacks self-contained model provenance")
            if dispatch_counting_available:
                require(info["flops"] == 2 * 2 * 4 * 6 * 6 * 3 * 3 * 3, "TorchScript convolution FLOP estimate is missing or incorrect")
        summaries.append(summary)
        sources.append(archive if name != "failure" else run)
    for extension in ("json", "csv"):
        destination = output / f"aggregate.{extension}"
        result = subprocess.run([sys.executable, "-m", "orion_bench.aggregate", *map(str, sources), "--output", str(destination)],
                                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=args.timeout)
        require(result.returncode == 0, f"aggregation failed: {result.stderr}")
        if extension == "json":
            rows = load_json(destination)
        else:
            with destination.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        require(len(rows) == len(summaries), "aggregation dropped or duplicated runs")
        for row, summary in zip(rows, summaries):
            require(not row.get("artifact_errors"), f"aggregation errors: {row.get('artifact_errors')}")
            require(row["summary.status"] == summary["status"], "aggregation status mismatch")
            require(int(row["summary.completed_iterations"]) == summary["completed_iterations"], "aggregation sample count mismatch")
            require(json.loads(row["config.input_shape"]) == [2, 3, 8, 8], "aggregation shape mismatch")
            require(row["config.device"] == args.device, "aggregation device mismatch")
    print(f"PASS: built-in/TorchScript inference, provenance mismatch rejection, model-load failure, SIGTERM artifacts, and JSON/CSV aggregation ({args.device}{', private tsp' if args.tsp else ''})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--tsp", action="store_true", help="use a unique private Task Spooler socket")
    parser.add_argument("--output-dir", type=Path, help="preserve artifacts in a new unique subdirectory here")
    parser.add_argument("--timeout", type=float, default=120, help="per-command safety deadline in seconds")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.gpu_index < 0:
        parser.error("timeout must be positive and finite; gpu-index must be nonnegative")
    if args.tsp and shutil.which("tsp") is None:
        parser.error("--tsp requires Task Spooler (tsp)")
    with tempfile.TemporaryDirectory(prefix="rover-smoke-") as temporary:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if args.tsp:
            environment["TS_SOCKET"] = str(Path(temporary) / "tsp.socket")
        if args.output_dir:
            args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
            output = Path(tempfile.mkdtemp(prefix="smoke-", dir=args.output_dir.expanduser().resolve()))
            print(f"Artifacts: {output}", flush=True)
        else:
            output = Path(temporary) / "results"
            output.mkdir()
        try:
            smoke(output, args, environment)
        finally:
            if args.tsp:
                # TS_SOCKET always identifies this smoke's private server, never a shared queue.
                subprocess.run(["tsp", "-K"], env=environment, capture_output=True, timeout=10, check=False)


if __name__ == "__main__":
    main()
