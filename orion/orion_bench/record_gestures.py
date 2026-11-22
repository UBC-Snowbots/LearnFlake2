"""Record and pose-label team gesture sessions for training.

Two subcommands, both producing a session directory with:
  keypoints.jsonl  per-frame tracked COCO-17 keypoints (image-normalized x, y, confidence)
  labels.csv       start_s,end_s,label segments (label: gesture | other | confuser)
  session.json     who/what/where metadata
  video.mp4        the raw frames (record only; extract leaves your original video in place)

record   Camera capture with a timed, printed prompt protocol, so segments are labelled
         as they are performed (an operator reads prompts to the astronaut).
extract  Run pose on an existing video plus a hand-written labels CSV (e.g. phone footage).
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import torch

from .live_demo import artifact, associate, pose_detections

LABELS = ("gesture", "other", "confuser")


def parse_protocol(text, repeats):
    """'gesture:4,other:6' → [(label, seconds), ...] repeated."""
    steps = []
    for item in text.split(","):
        label, seconds = item.split(":")
        if label not in LABELS:
            raise ValueError(f"Protocol label must be one of {LABELS}: {label}")
        steps.append((label, float(seconds)))
    return steps * repeats


def read_labels(path):
    segments = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            start, end, label = float(row["start_s"]), float(row["end_s"]), row["label"].strip()
            if label not in LABELS or not end > start >= 0:
                raise ValueError(f"Bad label row: {row}")
            segments.append((start, end, label))
    return segments


def write_labels(path, segments):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_s", "end_s", "label"])
        writer.writerows([[f"{s:.3f}", f"{e:.3f}", l] for s, e, l in segments])


def frame_record(index, timestamp, detections):
    return {"frame": index, "timestamp_seconds": timestamp, "people": [
        {"track_id": d["id"], "box_xyxy": [float(v) for v in d["box"]], "pose_confidence": d["confidence"],
         "keypoints": d["points"].round(5).tolist()} for d in detections]}


def prepare_output(path, meta):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Session directory is not empty: {path}")
    (path / "session.json").write_text(json.dumps(meta, indent=2) + "\n")


def record(args):
    steps = parse_protocol(args.protocol, args.repeats)
    out = Path(args.output_dir)
    pose = torch.jit.load(str(artifact(args.pose, "yolo11n-pose.torchscript")), map_location=args.device).eval()
    camera = args.source.isdecimal()
    capture = cv2.VideoCapture(int(args.source) if camera else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    meta = {"mode": "record", "person": args.person, "gesture_name": args.gesture_name,
            "suit": args.suit, "distance_m": args.distance, "notes": args.notes,
            "source": args.source, "protocol": steps}
    prepare_output(out, meta)
    writer, tracks, next_id, index, segments = None, {}, 1, 0, []
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    fps = fps if math.isfinite(fps) and 1 <= fps <= 120 else 15.0
    print(f"Lead-in {args.lead_in:.0f}s: astronaut stand in view, full body visible.", flush=True)
    start = time.perf_counter()
    plan, at = [], args.lead_in
    for label, seconds in steps:
        plan.append((at, at + seconds, label))
        at += seconds
    end_time, announced = at, -1
    with (out / "keypoints.jsonl").open("w") as records, torch.inference_mode():
        while True:
            ok, frame = capture.read()
            now = time.perf_counter() - start if camera else index / fps
            if not ok or frame is None or now >= end_time:
                break
            current = next((i for i, (s, e, _) in enumerate(plan) if s <= now < e), None)
            if current is not None and current != announced:
                announced = current
                s, e, label = plan[current]
                prompt = {"gesture": f"PERFORM '{args.gesture_name}' now", "other": "move normally (walk, stand, look around)",
                          "confuser": "do a lookalike: wave, stretch, point, swing arms"}[label]
                print(f"\a[{now:6.1f}s] {label.upper()} for {e - s:.0f}s: {prompt}", flush=True)
                segments.append((s, e, label))
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            detections = pose_detections(pose, frame, args.device)
            tracks, next_id = associate(tracks, detections, now, next_id)
            records.write(json.dumps(frame_record(index, now, detections)) + "\n")
            index += 1
    capture.release()
    if writer is not None:
        writer.release()
    if not segments or index == 0:
        raise RuntimeError("No labelled frames were recorded")
    # Trim a partially performed last segment to the actual recording end.
    segments[-1] = (segments[-1][0], min(segments[-1][1], now), segments[-1][2])
    write_labels(out / "labels.csv", segments)
    print(f"Saved {index} frames, {len(segments)} labelled segments to {out}")


def extract(args):
    segments = read_labels(args.labels)
    out = Path(args.output_dir)
    pose = torch.jit.load(str(artifact(args.pose, "yolo11n-pose.torchscript")), map_location=args.device).eval()
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Cannot open video with usable FPS: {args.video}")
    prepare_output(out, {"mode": "extract", "person": args.person, "gesture_name": args.gesture_name,
                         "suit": args.suit, "distance_m": args.distance, "notes": args.notes,
                         "source": str(Path(args.video).resolve())})
    tracks, next_id, index = {}, 1, 0
    with (out / "keypoints.jsonl").open("w") as records, torch.inference_mode():
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            timestamp = index / fps
            detections = pose_detections(pose, frame, args.device)
            tracks, next_id = associate(tracks, detections, timestamp, next_id)
            records.write(json.dumps(frame_record(index, timestamp, detections)) + "\n")
            index += 1
    capture.release()
    if index == 0:
        raise RuntimeError("Video contains no readable frames")
    write_labels(out / "labels.csv", segments)
    print(f"Extracted {index} frames ({index / fps:.1f}s) with {len(segments)} labelled segments to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "extract"):
        sub = commands.add_parser(name)
        sub.add_argument("--pose", required=True, help="pose TorchScript file or artifact directory")
        sub.add_argument("--output-dir", required=True, help="new, empty session directory")
        sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        sub.add_argument("--person", required=True, help="performer ID; splits are made by person")
        sub.add_argument("--gesture-name", default="two-arm overhead beckon")
        sub.add_argument("--suit", default="", help="clothing/suit description")
        sub.add_argument("--distance", type=float, help="approximate camera distance in metres")
        sub.add_argument("--notes", default="")
    rec = commands.choices["record"]
    rec.add_argument("--source", required=True, help="camera index or video path")
    rec.add_argument("--protocol", default="gesture:4,other:6,confuser:4,other:4",
                     help="label:seconds steps, labels gesture|other|confuser")
    rec.add_argument("--repeats", type=int, default=5)
    rec.add_argument("--lead-in", type=float, default=5.0, help="unlabelled seconds before the first prompt")
    ext = commands.choices["extract"]
    ext.add_argument("--video", required=True)
    ext.add_argument("--labels", required=True, help="CSV with start_s,end_s,label")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available", file=sys.stderr)
        return 2
    try:
        record(args) if args.command == "record" else extract(args)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Gesture session failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
