"""Offline CPU preparation of pinned YOLO11n-pose raw TorchScript forwards.

Only batch 1, 3-channel 640x640 FP32 input is supported. Outputs are decoded
raw pose predictions, without NMS, tracking, gesture classification or accuracy
validation. Checkpoint loading uses pickle: only use a trusted provenance pin.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile


MODEL_NAME = "yolo11n-pose.torchscript"
RTOL = 1e-4
ATOL = 1e-4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pin(path):
    pin = json.loads(Path(path).read_text())
    source = pin["source"]
    for key in ("repo_id", "revision", "filename", "sha256", "license"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Pin source.{key} must be a nonempty string")
    if not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("Pin SHA256 must be 64 lowercase hex characters")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("Pin revision must be a full immutable commit SHA")
    if source["filename"] != "yolo11n-pose.pt":
        raise ValueError("Only the yolo11n-pose.pt candidate is supported")
    required = pin.get("ultralytics_version")
    if not isinstance(required, str) or not required:
        raise ValueError("Pin must specify ultralytics_version")
    return source, required


def prepare(checkpoint, output_dir, pin, imgsz):
    if imgsz != 640:
        raise ValueError("Only static batch=1, input=[1,3,640,640] is supported")
    source, required_version = read_pin(pin)
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Complete integrity verification before importing/loading the pickle reader.
    if sha256(checkpoint) != source["sha256"]:
        raise ValueError("Checkpoint SHA256 does not match the trusted pin")
    installed = importlib.metadata.version("ultralytics")
    if installed != required_version:
        raise ValueError(f"Require ultralytics=={required_version}; installed {installed}; install nothing automatically")

    # Hide accelerators before importing either ML library; every tensor and model
    # is also explicitly placed on CPU. No exporter/download helper is invoked.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": installed,
    }
    settings = {
        "filename": MODEL_NAME, "input_shape": [1, 3, 640, 640],
        "format": "torchscript", "precision": "fp32", "device": "cpu",
        "batch": 1, "dynamic": False, "nms": False,
        "optimize": False, "frozen": False, "cpu_threads": 1,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".prepare_pose.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Preparation lock exists: {lock}; inspect interrupted/concurrent preparation before removing it") from exc
    os.close(descriptor)
    artifact = output_dir / MODEL_NAME
    manifest_path = output_dir / "model_manifest.json"
    published = False
    manifest_published = False
    try:
        if artifact.exists() or manifest_path.exists():
            if not artifact.is_file() or not manifest_path.is_file():
                raise ValueError("Incomplete existing output; refusing to overwrite")
            existing = json.loads(manifest_path.read_text())
            expected_export = {**settings, "sha256": sha256(artifact)}
            validation = existing.get("validation", {})
            if (existing.get("source") != source or existing.get("export") != expected_export
                    or existing.get("software") != software
                    or validation.get("passed") is not True
                    or validation.get("rtol") != RTOL or validation.get("atol") != ATOL
                    or validation.get("input") != "arange_mod_251_div_250"
                    or validation.get("output_shape") != [1, 56, 8400]):
                raise ValueError("Existing manifest/hash/settings differ; choose a new output directory")
            print(f"Verified existing preparation: {manifest_path}")
            return manifest_path

        # Load from an already-present local file only; YOLO never receives a
        # model alias or URL that could trigger checkpoint acquisition.
        candidate = YOLO(str(checkpoint), task="pose")
        model = candidate.model.cpu().float().eval()
        head = model.model[-1]
        if (candidate.task != "pose" or getattr(model, "task", None) != "pose"
                or list(getattr(head, "kpt_shape", [])) != [17, 3]
                or getattr(head, "nc", None) != 1):
            raise ValueError("Expected single-person-class pose architecture with 17x3 keypoints")
        parameters = sum(p.numel() for p in model.parameters())
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Match the raw TorchScript export head convention without the Ultralytics
        # exporter (which writes beside the checkpoint and may freeze a graph).
        for module in model.modules():
            if isinstance(module, Detect):
                module.dynamic = False
                module.export = True
                module.format = "torchscript"
                module.xyxy = False
                module.shape = None
        numeric = (torch.arange(3 * imgsz * imgsz, dtype=torch.float32)
                   .remainder(251).div(250).reshape(1, 3, imgsz, imgsz))
        with tempfile.TemporaryDirectory(prefix=".pose-", dir=output_dir) as temporary:
            staged = Path(temporary) / MODEL_NAME
            with torch.inference_mode():
                # Warm caches before tracing so shape-dependent anchor creation
                # is not recorded differently by the trace's checking pass.
                model(numeric)
                eager = model(numeric)
                if not isinstance(eager, torch.Tensor) or list(eager.shape) != [1, 56, 8400]:
                    raise ValueError("Unexpected raw output: expected [1,56,8400]")
                traced = torch.jit.trace(model, numeric, strict=True, check_trace=True)
                torch.jit.save(traced, str(staged))
                loaded = torch.jit.load(str(staged), map_location="cpu").eval()
                actual = loaded(numeric)
                if not torch.isfinite(eager).all() or not torch.isfinite(actual).all():
                    raise ValueError("Nonfinite raw predictions")
                torch.testing.assert_close(actual, eager, rtol=RTOL, atol=ATOL)
                max_error = (actual - eager).abs().max().item()
                if eager.numel() <= 1 or eager.max().item() == eager.min().item():
                    raise ValueError("Degenerate raw predictions cannot establish numeric agreement")
            manifest = {
                "source": source,
                "export": {**settings, "sha256": sha256(staged)},
                "software": software,
                "validation": {
                    "passed": True, "device": "cpu", "precision": "fp32",
                    "input": "arange_mod_251_div_250", "rtol": RTOL, "atol": ATOL,
                    "max_absolute_error": max_error, "output_shape": list(actual.shape),
                    "comparison": "serialized TorchScript versus eager export-mode model",
                    "estimated_parameters": parameters,
                    "keypoint_shape": [17, 3],
                    "limitations": "Raw decoded forward only; no NMS, tracking, gesture classification or accuracy assessment. Fixed batch 1 and 640x640 RGB NCHW input; caller supplies normalized pixels.",
                },
            }
            staged_manifest = Path(temporary) / "model_manifest.json"
            staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
            # Hard links provide atomic, exclusive publication on the same FS.
            # The success manifest is published last, only after numeric validation.
            os.link(staged, artifact)
            published = True
            os.link(staged_manifest, manifest_path)
            manifest_published = True
        print(f"Prepared {artifact}; max absolute error={max_error:.9g}")
        return manifest_path
    except BaseException:
        if manifest_published:
            manifest_path.unlink(missing_ok=True)
        if published:
            artifact.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640, choices=[640])
    args = parser.parse_args()
    try:
        prepare(args.checkpoint, args.output_dir, args.pin, args.imgsz)
    except Exception as exc:
        print(f"Pose preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
