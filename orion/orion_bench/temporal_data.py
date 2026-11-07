"""Download public 2D-skeleton action datasets and build attention-gesture sequences.

Sources and label mapping live in a JSON spec (see examples/attention_datasets.json).
Output is one compressed NPZ per split holding single-person COCO-17 sequences,
resampled to a fixed frame rate and stored as ragged arrays with offsets.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(source, raw_dir):
    """Download (if absent) and verify one source; never trust unverified bytes."""
    target = raw_dir / source["filename"]
    if not target.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as partial:
            with urllib.request.urlopen(source["url"]) as response:
                while block := response.read(1 << 20):
                    partial.write(block)
        os.replace(partial.name, target)
    if target.stat().st_size != source["bytes"] or sha256(target) != source["sha256"]:
        raise ValueError(f"{target} does not match pinned size/SHA-256; delete it and retry")
    return target


def select_person(keypoint, score):
    """Pick the person with the most confident, most moving skeleton.

    keypoint: [M,T,17,2], score: [M,T,17]. Deployment classifies one tracked
    person at a time, so training sequences are single-person as well.
    """
    if keypoint.shape[0] == 1:
        return keypoint[0], score[0]
    keypoint, score = keypoint.astype(np.float32), score.astype(np.float32)
    motion = np.nan_to_num(np.abs(np.diff(keypoint, axis=1)).sum(axis=(1, 2, 3)))
    confidence = np.nan_to_num(score.sum(axis=(1, 2)))
    best = int(np.argmax(confidence * (1.0 + motion / (motion.max() + 1e-6))))
    return keypoint[best], score[best]


def normalize(keypoint, score, img_shape):
    """Image-normalized x,y in [0,1] plus confidence → float16 [T,17,3]."""
    height, width = img_shape
    out = np.empty(keypoint.shape[:2] + (3,), dtype=np.float32)
    out[..., 0] = keypoint[..., 0] / width
    out[..., 1] = keypoint[..., 1] / height
    out[..., 2] = score
    return out.astype(np.float16)


def resample(sequence, source_fps, target_fps):
    count = max(1, int(round(len(sequence) * target_fps / source_fps)))
    index = np.minimum((np.arange(count) * source_fps / target_fps).round().astype(int),
                       len(sequence) - 1)
    return sequence[index]


def ntu_subject(frame_dir):
    return int(frame_dir[frame_dir.index("P") + 1:frame_dir.index("P") + 4])


def assign_split(dataset, frame_dir, official, val_groups):
    """Official train/test split; validation carved from train by subject/group only."""
    if frame_dir in official["test"]:
        return "test", None
    if frame_dir not in official["train"]:
        return None, None
    group = ntu_subject(frame_dir) if dataset == "ntu120" else None
    if group is not None and group in val_groups:
        return "val", group
    return "train", group


def build(spec, raw_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = spec["classes"]
    buffers = {split: {"seq": [], "label": [], "source": [], "group": [], "name": []} for split in SPLITS}
    report = {"spec_sha256": None, "sources": {}, "splits": {}}
    for source in spec["sources"]:
        path = fetch(source, raw_dir)
        with open(path, "rb") as stream:
            data = pickle.load(stream)  # Pinned SHA-256 verified above.
        official = {"train": set(data["split"][source["train_split"]]),
                    "test": set(data["split"][source["test_split"]])}
        val_groups = set(source.get("val_subjects", []))
        mapping = {int(k): classes.index(v) for k, v in source["label_map"].items()}
        default = classes.index(spec["default_class"])
        hmdb_val = source.get("val_fraction")
        rng = np.random.default_rng(source.get("seed", 0))
        kept = dropped = 0
        for item in data["annotations"]:
            split, group = assign_split(source["name"], item["frame_dir"], official, val_groups)
            if split is None or item["keypoint"].shape[0] == 0:
                dropped += 1
                continue
            if hmdb_val and split == "train":
                # HMDB has no subject IDs; group by source video stem to keep clips
                # cut from one video together.
                stem = item["frame_dir"].rsplit("_", 1)[0]
                group = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)
                if (group % 10_000) / 10_000 < hmdb_val:
                    split = "val"
            keypoint, score = select_person(item["keypoint"], item["keypoint_score"])
            sequence = resample(normalize(keypoint, score, item["img_shape"]),
                                source["fps"], spec["target_fps"])
            if len(sequence) < spec["min_frames"]:
                dropped += 1
                continue
            buffer = buffers[split]
            buffer["seq"].append(sequence)
            buffer["label"].append(mapping.get(int(item["label"]), default))
            buffer["source"].append(spec["sources"].index(source))
            buffer["group"].append(-1 if group is None else group)
            buffer["name"].append(f"{source['name']}/{item['frame_dir']}")
            kept += 1
        del data
        report["sources"][source["name"]] = {"kept": kept, "dropped": dropped, "sha256": source["sha256"]}
    for split, buffer in buffers.items():
        lengths = np.array([len(s) for s in buffer["seq"]], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        labels = np.array(buffer["label"], dtype=np.int64)
        np.savez_compressed(
            out_dir / f"{split}.npz",
            poses=np.concatenate(buffer["seq"]) if buffer["seq"] else np.zeros((0, 17, 3), np.float16),
            offsets=offsets, labels=labels,
            sources=np.array(buffer["source"], dtype=np.int64),
            groups=np.array(buffer["group"], dtype=np.int64),
            names=np.array(buffer["name"]))
        report["splits"][split] = {
            "sequences": int(len(labels)),
            "frames": int(offsets[-1]),
            "per_class": {c: int((labels == i).sum()) for i, c in enumerate(classes)},
            "per_source": {s["name"]: int((np.array(buffer["source"]) == i).sum())
                           for i, s in enumerate(spec["sources"])},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"Refusing to overwrite non-empty {args.output_dir}", file=sys.stderr)
        return 2
    spec = json.loads(args.spec.read_text())
    try:
        report = build(spec, args.raw_dir, args.output_dir)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 2
    report["spec_sha256"] = sha256(args.spec)
    report["spec"] = spec
    (args.output_dir / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["splits"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
