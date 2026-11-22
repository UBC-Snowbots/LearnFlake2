#!/usr/bin/env python3
"""Score a trained run's frozen checkpoint/threshold on the held-out test split once.

Adds `test` and `test_per_source` to the run's summary.json; refuses if present.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_bench.attention_model import AttentionGCN  # noqa: E402
from train_attention import Split, load, metrics, score_clips  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="seed-N directory")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    if "test" in summary:
        print("Test already scored for this run", file=sys.stderr)
        return 2
    options = summary["args"]
    classes = summary["classes"]
    model = AttentionGCN(len(classes), width=int(options.get("width", 48))).to(args.device)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location=args.device))
    test = Split(load(args.data / "test.npz"), int(options["window"]))
    score = score_clips(model, test, args.device, int(options["eval_stride"]))
    threshold = summary["validation"]["threshold"]
    summary["test"] = metrics(score, test.labels, threshold, classes)
    sources = np.array([n.split("/", 1)[0] for n in test.names])
    summary["test_per_source"] = {str(s): metrics(score[sources == s], test.labels[sources == s],
                                                  threshold, classes) for s in np.unique(sources)}
    np.savez_compressed(args.run / "test_scores.npz", score=score, labels=test.labels, names=test.names)
    (args.run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["test"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
