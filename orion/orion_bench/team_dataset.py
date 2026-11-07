"""Build a gesture training set from recorded team sessions (+ optional public negatives).

Classes: other / gesture / confuser (gesture is index 1, as the trainer expects).
Splits are by performer: every session of a test person is held out entirely.
Public PYSKL clips (data/attention_v1) can be added as negatives; public hand
waving becomes a *confuser* because it is not our chosen gesture.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CLASSES = ["other", "gesture", "confuser"]
TARGET_FPS = 15.0
PUBLIC_REMAP = {0: 0, 1: 2, 2: 2}  # public other/attention(wave)/confuser → ours


def load_session(path):
    frames = [json.loads(line) for line in (path / "keypoints.jsonl").read_text().splitlines() if line]
    with (path / "labels.csv").open(newline="") as stream:
        segments = [(float(r["start_s"]), float(r["end_s"]), r["label"].strip()) for r in csv.DictReader(stream)]
    meta = json.loads((path / "session.json").read_text())
    return frames, segments, meta


def performer_track(frames, segments):
    """Track ID present in the most labelled frames (the astronaut facing the camera)."""
    counts = {}
    for frame in frames:
        t = frame["timestamp_seconds"]
        if any(s <= t < e for s, e, _ in segments):
            for person in frame["people"]:
                counts[person["track_id"]] = counts.get(person["track_id"], 0) + 1
    if not counts:
        raise ValueError("No tracked person inside labelled segments")
    return max(counts, key=counts.get)


def segment_sequence(frames, track, start, end):
    """Resample the performer's keypoints to 15 Hz; missing detections → zero confidence."""
    times = np.array([f["timestamp_seconds"] for f in frames])
    end = min(end, times[-1] + 1 / TARGET_FPS)  # labels may overrun the recording
    samples = []
    for t in np.arange(start, end, 1 / TARGET_FPS):
        index = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(frames) - 1))
        person = next((p for p in frames[index]["people"] if p["track_id"] == track), None)
        samples.append(np.zeros((17, 3), np.float32) if person is None else np.asarray(person["keypoints"], np.float32))
    return np.stack(samples).astype(np.float16) if samples else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", required=True, type=Path, help="directory of session directories")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-people", default="", help="comma-separated performer IDs for validation")
    parser.add_argument("--test-people", default="", help="comma-separated performer IDs held out for test")
    parser.add_argument("--public", type=Path, help="optional data/attention_v1 directory for public negatives")
    parser.add_argument("--public-other-fraction", type=float, default=0.2,
                        help="fraction of public 'other' clips kept (confusers always kept)")
    parser.add_argument("--min-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"Refusing to overwrite non-empty {args.output_dir}", file=sys.stderr)
        return 2
    val_people = {p for p in args.val_people.split(",") if p}
    test_people = {p for p in args.test_people.split(",") if p}
    if val_people & test_people:
        print("A person cannot be in both validation and test", file=sys.stderr)
        return 2
    buffers = {s: {"seq": [], "label": [], "group": [], "name": []} for s in ("train", "val", "test")}
    people, sessions = {}, sorted(p for p in args.sessions.iterdir() if (p / "labels.csv").is_file())
    if not sessions:
        print(f"No session directories with labels.csv under {args.sessions}", file=sys.stderr)
        return 2
    for path in sessions:
        frames, segments, meta = load_session(path)
        person = str(meta["person"])
        split = "test" if person in test_people else "val" if person in val_people else "train"
        group = int(hashlib.sha256(person.encode()).hexdigest()[:8], 16)
        track = performer_track(frames, segments)
        for i, (start, end, label) in enumerate(segments):
            sequence = segment_sequence(frames, track, start, end)
            if sequence is None or len(sequence) < args.min_frames:
                continue
            buffer = buffers[split]
            buffer["seq"].append(sequence)
            buffer["label"].append(CLASSES.index(label))
            buffer["group"].append(group)
            buffer["name"].append(f"team/{person}/{path.name}/{i}")
        people.setdefault(person, {"split": split, "sessions": 0})["sessions"] += 1
    if args.public:
        rng = np.random.default_rng(args.seed)
        for split in buffers:
            data = np.load(args.public / f"{split}.npz")
            poses, offsets, labels, names = data["poses"], data["offsets"], data["labels"], data["names"]
            for i in range(len(labels)):
                ours = PUBLIC_REMAP[int(labels[i])]
                if ours == 0 and rng.random() >= args.public_other_fraction:
                    continue
                buffers[split]["seq"].append(poses[offsets[i]:offsets[i + 1]])
                buffers[split]["label"].append(ours)
                buffers[split]["group"].append(-1)
                buffers[split]["name"].append(f"public-{names[i]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"spec": {"classes": CLASSES, "target_fps": TARGET_FPS, "sessions": str(args.sessions.resolve()),
                       "val_people": sorted(val_people), "test_people": sorted(test_people),
                       "public": str(args.public.resolve()) if args.public else None,
                       "public_other_fraction": args.public_other_fraction, "seed": args.seed},
              "people": people, "splits": {}}
    for split, buffer in buffers.items():
        labels = np.array(buffer["label"], dtype=np.int64)
        lengths = np.array([len(s) for s in buffer["seq"]], dtype=np.int64)
        np.savez_compressed(args.output_dir / f"{split}.npz",
                            poses=np.concatenate(buffer["seq"]) if buffer["seq"] else np.zeros((0, 17, 3), np.float16),
                            offsets=np.concatenate([[0], np.cumsum(lengths)]), labels=labels,
                            groups=np.array(buffer["group"], dtype=np.int64), names=np.array(buffer["name"]))
        report["splits"][split] = {"clips": int(len(labels)),
                                   "per_class": {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)},
                                   "team_clips": int(sum(n.startswith("team/") for n in buffer["name"]))}
    spec_bytes = json.dumps(report["spec"], sort_keys=True).encode()
    report["spec_sha256"] = hashlib.sha256(spec_bytes).hexdigest()
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["splits"], indent=2))
    for split, info in report["splits"].items():
        if info["per_class"]["gesture"] == 0:
            print(f"WARNING: {split} has no gesture clips; record more people/sessions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
