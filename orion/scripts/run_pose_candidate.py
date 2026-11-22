#!/usr/bin/env python3
"""Run a repeatable desktop pose group inside an already queued GPU worker.

Submit externally with tsp gpu-run <python> scripts/run_pose_candidate.py ... .
This script never submits jobs or changes Task Spooler configuration.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import uuid
import tarfile


ROOT = Path(__file__).resolve().parents[1]
# Direct script execution must work without an editable package installation.
sys.path.insert(0, str(ROOT))

from orion_bench.aggregate import aggregate
from orion_bench.runner import dump, load_config


LIMITATIONS = [
    "Desktop-only synthetic-input measurements; not camera-to-decision latency.",
    "Raw model forward only; no NMS, person tracking, or gesture classification.",
    "No pose or gesture accuracy was measured; no training was performed.",
    "No Jetson performance, fit, or budget verdict can be inferred from desktop/RTX results.",
    "FP32 export may be cast to FP16 by the benchmark backend at runtime.",
]


class Worker:
    """Forward cancellation, then reap the child before packaging group results."""

    def __init__(self):
        self.child = None
        self.interrupted_signal = None

    def interrupt(self, signum, frame):
        if self.interrupted_signal is None:
            self.interrupted_signal = signum
            self.forward_interrupt()

    def forward_interrupt(self):
        if self.child is not None and self.interrupted_signal is not None:
            try:
                self.child.send_signal(self.interrupted_signal)
            except ProcessLookupError:
                pass

    def run(self, config_path, log_path):
        command = [sys.executable, "-m", "orion_bench", "run", "--config", str(config_path)]
        with log_path.open("w", encoding="utf-8") as console:
            if self.interrupted_signal is not None:
                return None
            try:
                # Isolate terminal signals so only our explicit forwarding reaches the child.
                self.child = subprocess.Popen(
                    command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                # Covers cancellation between the pre-launch check and Popen returning.
                if self.interrupted_signal is not None:
                    self.forward_interrupt()
                return self.child.wait()
            finally:
                if self.child is not None:
                    # Never abandon the benchmarking process, including on wrapper errors.
                    if self.child.poll() is None:
                        if self.interrupted_signal is None:
                            self.child.terminate()
                        self.child.wait()
                    self.child = None


def read_result(record):
    """Use the CLI's declared run path, never directory discovery or queue polling."""
    log_path = Path(record["console_log"])
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if line.startswith("run_dir="):
            record["run_dir"] = line.removeprefix("run_dir=")
        elif line.startswith("archive="):
            record["archive"] = line.removeprefix("archive=")
        elif line.startswith("status="):
            record["cli_status"] = line.removeprefix("status=")
    if record["run_dir"] is None:
        record["status"] = "failed" if record["exit_code"] is not None else "not_started"
        record["error"] = record.get("error") or "CLI did not report a run directory"
        return
    try:
        summary = json.loads((Path(record["run_dir"]) / "summary.json").read_text(encoding="utf-8"))
        record["status"] = summary["status"]
        record["completed_iterations"] = summary.get("completed_iterations", 0)
        record["warmup_completed"] = summary.get("warmup_completed", 0)
        record["error"] = summary.get("error")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["status"] = "failed"
        record["error"] = f"Cannot read run summary: {exc}"


def collection_errors(record, base):
    """Check durable evidence before calling a completed run collected."""
    errors = []
    run = Path(record["run_dir"])
    try:
        with (run / "latency_samples.csv").open(newline="", encoding="utf-8") as stream:
            steady = [row for row in csv.DictReader(stream) if row["phase"] == "steady"]
        if len(steady) != base["iterations"]:
            errors.append("steady sample count does not match requested iterations")
        if any(not math.isfinite(float(row["inference_gpu_ms"])) or float(row["inference_gpu_ms"]) < 0 for row in steady):
            errors.append("GPU event measurements are missing, negative, or nonfinite")
        manifest = json.loads(Path(base["model_manifest"]).read_text(encoding="utf-8"))
        expected = manifest["export"]["sha256"]
        info = json.loads((run / "model_info.json").read_text(encoding="utf-8"))
        if not expected or info.get("model_sha256") != expected:
            errors.append("model hash does not match export manifest")
        hashes = {}
        for line in (run / "manifest.sha256").read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if name in hashes or Path(name).name != name:
                raise ValueError("invalid artifact hash manifest entry")
            hashes[name] = digest
        files = {path.name for path in run.iterdir() if path.is_file()}
        if set(hashes) != files - {"manifest.sha256"}:
            errors.append("artifact hash manifest coverage mismatch")
        for name, expected_hash in hashes.items():
            with (run / name).open("rb") as stream:
                if stream_sha256(stream) != expected_hash:
                    errors.append(f"artifact hash mismatch: {name}")
        archive_path = record.get("archive")
        if not archive_path:
            raise ValueError("CLI did not report an archive")
        with tarfile.open(archive_path, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if {member.name for member in members} != {f"{run.name}/{name}" for name in files} or len(members) != len(files):
                errors.append("archive coverage mismatch")
            for member in members:
                name = Path(member.name).name
                expected_hash = hashes.get(name)
                if name == "manifest.sha256":
                    with (run / name).open("rb") as stream:
                        expected_hash = stream_sha256(stream)
                with archive.extractfile(member) as stream:
                    if stream_sha256(stream) != expected_hash:
                        errors.append(f"archive hash mismatch: {member.name}")
    except (OSError, ValueError, KeyError, TypeError, EOFError, tarfile.TarError) as exc:
        errors.append(f"Cannot validate collection evidence: {exc}")
    return errors


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def write_results(group, records, base, worker):
    # Missing/unstarted run artifacts also become rows via the existing aggregator.
    sources = [record["run_dir"] or record["attempt_dir"] for record in records]
    rows = aggregate(sources)
    for row, record in zip(rows, records):
        row.update({f"group.{key}": record.get(key) for key in (
            "precision", "seed", "status", "exit_code", "console_log", "error",
        )})
        if record["status"] == "completed" and (
            record["exit_code"] != 0 or row.get("artifact_errors")
            or record.get("completed_iterations") != base["iterations"]
            or record.get("warmup_completed") != base["warmup"]
            or record.get("collection_errors")
        ):
            record["status"] = "incomplete"
            row["group.status"] = "incomplete"
        row["group.collection_errors"] = json.dumps(record.get("collection_errors", []))
    dump(group / "aggregate.json", rows)
    with (group / "aggregate.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    complete = worker.interrupted_signal is None and all(r["status"] == "completed" for r in records)
    summary = {
        "schema_version": "1.0",
        "status": "interrupted" if worker.interrupted_signal is not None else ("completed" if complete else "incomplete"),
        "verdict": "measurements_collected" if complete else "incomplete",
        "interrupted_signal": worker.interrupted_signal,
        "planned_runs": len(records),
        "completed_runs": sum(r["status"] == "completed" for r in records),
        "model_manifest": base.get("model_manifest"),
        "limitations": LIMITATIONS,
        "runs": records,
        "aggregate_csv": str(group / "aggregate.csv"),
        "aggregate_json": str(group / "aggregate.json"),
    }
    dump(group / "group_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/yolo11n_pose_gpu.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/pose-first")
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "fp16"), default=["fp32", "fp16"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    if len(set(args.precisions)) != len(args.precisions):
        parser.error("--precisions must be distinct")
    try:
        base = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = args.output_dir.expanduser().resolve()
    group = output / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    group.mkdir(parents=True)
    worker = Worker()
    old_handlers = {sig: signal.signal(sig, worker.interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    records = []
    try:
        # Materialize every planned effective config before running, with absolute paths
        # resolved by load_config relative to the original base configuration.
        for precision in args.precisions:
            for seed in args.seeds:
                attempt = group / f"{precision}-seed-{seed}"
                attempt.mkdir()
                cfg = {**base, "precision": precision, "seed": seed, "output_root": str(attempt / "runs")}
                dump(attempt / "config.json", cfg)
                records.append({
                    "precision": precision, "seed": seed, "status": "not_started",
                    "attempt_dir": str(attempt), "config": str(attempt / "config.json"),
                    "console_log": str(attempt / "console.log"),
                    "exit_code": None, "run_dir": None, "archive": None,
                })
        dump(group / "base_config.json", base)
        print(f"group_dir={group}", flush=True)
        for record in records:
            if worker.interrupted_signal is not None:
                break
            try:
                record["exit_code"] = worker.run(Path(record["config"]), Path(record["console_log"]))
                read_result(record)
                if record["status"] == "completed":
                    record["collection_errors"] = collection_errors(record, base)
            except OSError as exc:
                record["status"] = "failed"
                record["error"] = f"Cannot launch/read benchmark: {exc}"
            print(f"precision={record['precision']} seed={record['seed']} status={record['status']} run_dir={record['run_dir']}", flush=True)
    finally:
        try:
            summary = write_results(group, records, base, worker)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    print(f"group_summary={group / 'group_summary.json'}\nverdict={summary['verdict']}", flush=True)
    if worker.interrupted_signal is not None:
        return 128 + worker.interrupted_signal
    return 0 if summary["verdict"] == "measurements_collected" else 1


if __name__ == "__main__":
    sys.exit(main())
