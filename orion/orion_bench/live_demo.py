"""Headless offline pose and temporal-classifier demonstration.

Inputs are local TorchScript artifacts, a camera index or a local video file.
The output is diagnostic, not a validated gesture detector.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


POSE_CONFIDENCE = 0.25
NMS_IOU = 0.45
TRACK_IOU = 0.3
TRACK_GAP = 0.5
TARGET_FPS = 15.0
WINDOW = 48


def artifact(path, filename):
    path = Path(path)
    if path.is_dir():
        path = path / filename
    if not path.is_file():
        raise ValueError(f"Local model file not found: {path}")
    return path


def classifier_contract(model_path, manifest_path):
    """(threshold, positive label, window length) from the classifier manifest."""
    path = Path(manifest_path) if manifest_path else model_path.parent / "model_manifest.json"
    if not path.exists():
        if manifest_path:
            raise ValueError(f"Classifier manifest not found: {path}")
        return None, "target", WINDOW
    manifest = json.loads(path.read_text())
    expected = manifest.get("export", {}).get("sha256")
    if expected:
        digest = hashlib.sha256()
        with model_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError("Classifier artifact does not match its manifest SHA-256")
    source = manifest.get("source", {})
    classes = source.get("classes")
    if classes is None or len(classes) < 2:
        raise ValueError("Classifier manifest must list classes with the target at index 1")
    value = source.get("validation_threshold")
    if value is None:
        raise ValueError(f"Classifier manifest has no source.validation_threshold: {path}")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Classifier threshold must be finite and between 0 and 1")
    shape = manifest.get("export", {}).get("input_shape", [1, WINDOW, 17, 3])
    if len(shape) != 4 or shape[2:] != [17, 3]:
        raise ValueError("Classifier input_shape must be [1, T, 17, 3]")
    return value, classes[1], int(shape[1])


def iou(box, boxes):
    left = np.maximum(box[:2], boxes[:, :2])
    right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(right - left, 0).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def pose_detections(model, frame, device):
    height, width = frame.shape[:2]
    scale = min(640 / width, 640 / height)
    resized_w, resized_h = round(width * scale), round(height * scale)
    left, top = (640 - resized_w) // 2, (640 - resized_h) // 2
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32).div_(255).unsqueeze(0)
    raw = model(tensor)
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (1, 56, 8400):
        raise ValueError("Pose model must return a tensor of shape [1,56,8400]")
    raw = raw[0].transpose(0, 1)
    raw = raw[raw[:, 4] >= POSE_CONFIDENCE].cpu().numpy()
    if not len(raw):
        return []
    if not np.isfinite(raw).all():
        raise ValueError("Pose model produced nonfinite predictions")
    boxes = np.empty((len(raw), 4), dtype=np.float32)
    boxes[:, :2] = raw[:, :2] - raw[:, 2:4] / 2
    boxes[:, 2:] = raw[:, :2] + raw[:, 2:4] / 2
    order = np.argsort(-raw[:, 4])
    keep = []
    while len(order):
        index = order[0]
        keep.append(index)
        remaining = order[1:]
        order = remaining[iou(boxes[index], boxes[remaining]) <= NMS_IOU]
    detections = []
    for index in keep:
        box = boxes[index].copy()
        box[[0, 2]] = np.clip((box[[0, 2]] - left) / scale, 0, width)
        box[[1, 3]] = np.clip((box[[1, 3]] - top) / scale, 0, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        points = raw[index, 5:].reshape(17, 3).copy()
        points[:, 0] = np.clip((points[:, 0] - left) / scale / width, 0, 1)
        points[:, 1] = np.clip((points[:, 1] - top) / scale / height, 0, 1)
        detections.append({"box": box, "confidence": float(raw[index, 4]), "points": points})
    return detections


def associate(tracks, detections, timestamp, next_id, window=WINDOW):
    # Remove stale identities before matching; never bridge a long occlusion.
    tracks = {key: value for key, value in tracks.items() if timestamp - value["last"] <= TRACK_GAP}
    pairs = []
    for key, track in tracks.items():
        if detections:
            overlaps = iou(track["box"], np.stack([d["box"] for d in detections]))
            pairs.extend((float(score), key, index) for index, score in enumerate(overlaps) if score >= TRACK_IOU)
    assigned_tracks, assigned_detections = set(), set()
    for _, key, index in sorted(pairs, reverse=True):
        if key not in assigned_tracks and index not in assigned_detections:
            detections[index]["id"] = key
            assigned_tracks.add(key)
            assigned_detections.add(index)
    for detection in detections:
        if "id" not in detection:
            detection["id"] = next_id
            tracks[next_id] = {"samples": deque(maxlen=window), "next_sample": timestamp,
                               "last": timestamp, "box": detection["box"]}
            next_id += 1
        track = tracks[detection["id"]]
        # Sample at 15 Hz using the latest prior observation (causal hold).
        # A missing observation resets the window instead of fabricating motion.
        if timestamp - track["last"] > 2.5 / TARGET_FPS:
            track["samples"].clear()
            track["next_sample"] = timestamp
        while track["next_sample"] <= timestamp + 1e-9:
            points = detection["points"] if "previous" not in track else track["previous"]
            if abs(track["next_sample"] - timestamp) < 1e-9:
                points = detection["points"]
            track["samples"].append(points.copy())
            track["next_sample"] += 1 / TARGET_FPS
        track.update(box=detection["box"], last=timestamp, previous=detection["points"])
    return tracks, next_id


def run(args):
    pose_path = artifact(args.pose, "yolo11n-pose.torchscript")
    classifier_path = artifact(args.classifier, "attention_gcn.torchscript")
    threshold, target, window = classifier_contract(classifier_path, args.classifier_manifest)
    if not 1 <= args.trigger_k <= args.trigger_m:
        raise ValueError("--trigger-k must be between 1 and --trigger-m")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    camera = args.source.isdecimal()
    source = int(args.source) if camera else args.source
    if not camera and not Path(source).is_file():
        raise ValueError(f"Local video file not found: {source}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "starting", "source": args.source, "device": args.device,
               "threshold": threshold, "target": target, "window": window,
               "trigger_rule": f"{args.trigger_k} of last {args.trigger_m} scored frames above threshold",
               "frames": 0, "processed_fps": None, "triggers": [],
               "note": "Diagnostic only; not validated for rover control."}
    capture = writer = None
    start = time.perf_counter()
    try:
        pose = torch.jit.load(str(pose_path), map_location=args.device).eval()
        classifier = torch.jit.load(str(classifier_path), map_location=args.device).eval()
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not camera and (not math.isfinite(source_fps) or source_fps <= 0):
            raise RuntimeError("Video has no usable FPS metadata for timestamps")
        output_fps = source_fps if not camera else (source_fps if math.isfinite(source_fps) and 1 <= source_fps <= 120 else TARGET_FPS)
        summary.update(source_fps=source_fps if math.isfinite(source_fps) else None, overlay_fps=output_fps,
                       timestamp_basis="monotonic capture time" if camera else "frame index / source FPS")
        tracks, next_id = {}, 1
        elapsed_pose = elapsed_classifier = 0.0
        with (output / "frames.jsonl").open("w") as records, torch.inference_mode():
            while args.max_frames is None or summary["frames"] < args.max_frames:
                frame_start = time.perf_counter()
                ok, frame = capture.read()
                captured = time.perf_counter()
                if not ok or frame is None:
                    if camera:
                        raise RuntimeError("Camera frame capture failed")
                    if summary["frames"] == 0:
                        raise RuntimeError("Video contains no readable frames")
                    break
                timestamp = captured - start if camera else summary["frames"] / source_fps
                height, width = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(output / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open output video writer (mp4v)")
                    summary["frame_size"] = [width, height]
                elif [width, height] != summary["frame_size"]:
                    raise RuntimeError("Source frame dimensions changed")
                pose_start = time.perf_counter()
                detections = pose_detections(pose, frame, args.device)
                pose_ms = (time.perf_counter() - pose_start) * 1000
                tracks, next_id = associate(tracks, detections, timestamp, next_id, window)
                classifier_start = time.perf_counter()
                people = []
                for detection in detections:
                    track = tracks[detection["id"]]
                    score = None
                    triggered = False
                    if len(track["samples"]) == window:
                        inputs = torch.from_numpy(np.stack(track["samples"])).unsqueeze(0).to(args.device)
                        logits = classifier(inputs)
                        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 3):
                            raise ValueError("Classifier must return logits of shape [1,3]")
                        score = float(torch.softmax(logits, dim=1)[0, 1].item())
                        if not math.isfinite(score):
                            raise ValueError("Classifier produced nonfinite scores")
                        if threshold is not None:
                            fires = track.setdefault("fires", deque(maxlen=args.trigger_m))
                            fires.append(score >= threshold)
                            triggered = sum(fires) >= args.trigger_k
                            if triggered and not track.get("triggered"):
                                summary["triggers"].append({"frame": summary["frames"], "timestamp_seconds": timestamp,
                                                            "track_id": detection["id"]})
                            track["triggered"] = triggered
                    box = detection["box"]
                    status = "warming up" if score is None else f"{target} {score:.2f}" + (" TRIGGER" if triggered else "")
                    colour = (0, 0, 255) if triggered else (0, 255, 0)
                    cv2.rectangle(frame, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), colour, 2)
                    cv2.putText(frame, f"ID {detection['id']} {status}", (int(box[0]), max(15, int(box[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
                    for x, y, confidence in detection["points"]:
                        if confidence >= 0.25:
                            cv2.circle(frame, (int(x * width), int(y * height)), 2, (0, 255, 255), -1)
                    people.append({"track_id": detection["id"], "box_xyxy": box.tolist(),
                                   "pose_confidence": detection["confidence"], "keypoints_xy_normalized_confidence": detection["points"].tolist(),
                                   "window_samples": len(track["samples"]), "target_score": score,
                                   "threshold_exceeded": None if score is None or threshold is None else score >= threshold,
                                   "triggered": triggered})
                classifier_ms = (time.perf_counter() - classifier_start) * 1000
                writer.write(frame)
                total_ms = (time.perf_counter() - frame_start) * 1000
                record = {"frame": summary["frames"], "timestamp_seconds": timestamp, "people": people,
                          "latency_ms": {"capture": (captured - frame_start) * 1000, "pose": pose_ms,
                                         "classification_and_overlay": classifier_ms, "total": total_ms}}
                records.write(json.dumps(record, allow_nan=False) + "\n")
                records.flush()
                summary["frames"] += 1
                elapsed_pose += pose_ms
                elapsed_classifier += classifier_ms
        summary.update(status="complete", mean_pose_ms=elapsed_pose / summary["frames"],
                       mean_classification_and_overlay_ms=elapsed_classifier / summary["frames"])
    except KeyboardInterrupt:
        summary.update(status="interrupted")
    except Exception as exc:
        summary.update(status="error", error=str(exc))
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        elapsed = time.perf_counter() - start
        summary["elapsed_seconds"] = elapsed
        summary["processed_fps"] = summary["frames"] / elapsed if elapsed > 0 else None
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['status']}: {summary['frames']} frames, {summary['processed_fps']:.2f} processed FPS; diagnostic output in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", required=True, help="Local pose TorchScript file or artifact directory")
    parser.add_argument("--classifier", required=True, help="Local classifier TorchScript file or artifact directory")
    parser.add_argument("--classifier-manifest", help="Optional classifier manifest (default: adjacent model_manifest.json)")
    parser.add_argument("--source", required=True, help="Nonnegative camera index or local video path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--trigger-k", type=int, default=6, help="frames above threshold needed to trigger")
    parser.add_argument("--trigger-m", type=int, default=8, help="recent scored frames considered")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"Live demo failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
