#!/usr/bin/env python3
"""Train/evaluate the causal attention-gesture classifier on prepared sequences.

Model selection and threshold choice use only the validation split. The test
split is scored once, with the frozen threshold, after training finishes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_bench.attention_model import FLIP, AttentionGCN  # noqa: E402


def load(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


class Split:
    def __init__(self, data, window):
        self.poses = torch.from_numpy(data["poses"].astype(np.float32))
        self.offsets = data["offsets"]
        self.labels = data["labels"]
        self.names = data["names"]
        self.window = window

    def __len__(self):
        return len(self.labels)

    def crop(self, i, end):
        """Window of `window` frames ending at `end` (exclusive), left-padded by repetition."""
        start, stop = self.offsets[i], self.offsets[i + 1]
        index = np.arange(end - self.window, end).clip(0, None) + start
        return self.poses[np.minimum(index, stop - 1)]


def augment(batch, generator):
    n = batch.shape[0]
    xy = batch[..., :2] - 0.5
    angle = (torch.rand(n, generator=generator) - 0.5) * 0.5
    scale = 0.8 + 0.4 * torch.rand(n, generator=generator)
    cos, sin = torch.cos(angle) * scale, torch.sin(angle) * scale
    rot = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    xy = torch.einsum("ntvc,ndc->ntvd", xy, rot) + 0.5
    xy = xy + 0.1 * (torch.rand(n, 1, 1, 2, generator=generator) - 0.5)
    out = torch.cat([xy, batch[..., 2:]], -1)
    flip = torch.rand(n, generator=generator) < 0.5
    flipped = out[:, :, FLIP].clone()
    flipped[..., 0] = 1 - flipped[..., 0]
    out = torch.where(flip[:, None, None, None], flipped, out)
    drop = torch.rand(out.shape[:3], generator=generator) < 0.05  # missed keypoints
    out[..., 2] = torch.where(drop, torch.zeros_like(out[..., 2]), out[..., 2])
    out[..., :2] = torch.where(drop[..., None], torch.zeros_like(out[..., :2]), out[..., :2])
    return out


def sample_batch(split, indices, generator):
    windows = []
    for i in indices:
        length = split.offsets[i + 1] - split.offsets[i]
        end = int(torch.randint(min(length, split.window // 2), length + 1, (1,), generator=generator))
        windows.append(split.crop(i, end))
    return torch.stack(windows)


@torch.inference_mode()
def score_clips(model, split, device, stride, batch_size=512):
    """Per-clip positive-class (index 1) probability = max over causal sliding windows."""
    model.eval()
    jobs = []
    for i in range(len(split)):
        length = split.offsets[i + 1] - split.offsets[i]
        ends = range(min(length, split.window), length + 1, stride)
        jobs.extend((i, e) for e in (ends if len(ends) else [length]))
    best = np.zeros(len(split))
    for begin in range(0, len(jobs), batch_size):
        chunk = jobs[begin:begin + batch_size]
        batch = torch.stack([split.crop(i, e) for i, e in chunk]).to(device)
        prob = model(batch).softmax(-1)[:, 1].float().cpu().numpy()
        np.maximum.at(best, [i for i, _ in chunk], prob)
    return best


def average_precision(score, positive):
    order = np.argsort(-score)
    hits = positive[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / max(hits.sum(), 1))


def metrics(score, labels, threshold, classes):
    positive = labels == 1  # class index 1 is the target gesture in every dataset
    fire = score >= threshold
    result = {
        "average_precision": average_precision(score, positive),
        "threshold": float(threshold),
        "recall": float(fire[positive].mean()) if positive.any() else None,
        "precision": float(fire[positive].sum() / max(fire.sum(), 1)),
        "positives": int(positive.sum()),
        "clips": int(len(labels)),
    }
    for name in classes:
        if name != classes[1]:
            mask = labels == classes.index(name)
            result[f"false_positive_rate_{name}"] = float(fire[mask].mean()) if mask.any() else None
    return result


def threshold_for(score, labels, classes, max_fpr):
    """Lowest threshold whose negative-class false-positive rate stays ≤ max_fpr."""
    negative = score[labels != 1]
    return float(np.quantile(negative, 1 - max_fpr)) if len(negative) else 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=1500)
    parser.add_argument("--eval-stride", type=int, default=4)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--score-test", action="store_true",
                        help="score held-out test once; only for frozen final configurations")
    parser.add_argument("--sample-ratio", default="1,1,1",
                        help="relative per-class sampling share, in dataset class order")
    args = parser.parse_args()

    report = json.loads((args.data / "dataset_report.json").read_text())
    classes = report["spec"]["classes"]
    run = args.output_dir / f"seed-{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    train = Split(load(args.data / "train.npz"), args.window)
    val = Split(load(args.data / "val.npz"), args.window)

    # Balance classes by sampling, rather than weighting the loss.
    counts = np.bincount(train.labels, minlength=len(classes)).astype(float)
    ratio = np.array([float(x) for x in args.sample_ratio.split(",")])
    weights = torch.from_numpy(ratio[train.labels] / counts[train.labels])
    model = AttentionGCN(len(classes), width=args.width).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps,
                                                   pct_start=0.05)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    history, best = [], None
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        indices = torch.multinomial(weights, args.batch, replacement=True, generator=generator).numpy()
        batch = augment(sample_batch(train, indices, generator), generator).to(device)
        target = torch.from_numpy(train.labels[indices]).to(device)
        loss = loss_fn(model(batch), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            score = score_clips(model, val, device, args.eval_stride)
            threshold = threshold_for(score, val.labels, classes, args.max_fpr)
            entry = {"step": step, "loss": loss.item(), "seconds": time.monotonic() - started,
                     "val": metrics(score, val.labels, threshold, classes)}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if best is None or entry["val"]["average_precision"] > best["val"]["average_precision"]:
                best = entry
                torch.save(model.state_dict(), run / "best.pt")

    # Frozen choice: best validation AP checkpoint and its validation threshold.
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    summary = {
        "seed": args.seed, "args": {k: str(v) for k, v in vars(args).items()},
        "classes": classes, "dataset_spec_sha256": report["spec_sha256"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
        "selected_step": best["step"], "validation": best["val"], "history": history,
    }
    if args.score_test:
        test = Split(load(args.data / "test.npz"), args.window)
        test_score = score_clips(model, test, device, args.eval_stride)
        summary["test"] = metrics(test_score, test.labels, best["val"]["threshold"], classes)
        sources = np.array([n.split("/", 1)[0] for n in test.names])
        summary["test_per_source"] = {
            str(name): metrics(test_score[sources == name], test.labels[sources == name],
                               best["val"]["threshold"], classes)
            for name in np.unique(sources)}
        np.savez_compressed(run / "test_scores.npz", score=test_score, labels=test.labels,
                            names=test.names)
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("seed", "selected_step", "validation", "test")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
