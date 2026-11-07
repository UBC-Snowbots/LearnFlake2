#!/usr/bin/env python3
"""Export a trained attention classifier run to TorchScript with an Orion manifest.

The exported module maps [1, T, 17, 3] keypoints (image-normalized x, y, confidence)
to 3 class logits; it is checked against the eager model before publication.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orion_bench.attention_model import AttentionGCN  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="seed-N directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    window = int(summary["args"]["window"])
    model = AttentionGCN(len(summary["classes"]), width=int(summary["args"].get("width", 48)))
    model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu"))
    model.eval()
    example = torch.rand(1, window, 17, 3, generator=torch.Generator().manual_seed(0))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    artifact = args.output_dir / "attention_gcn.torchscript"
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=True)
        torch.jit.save(traced, str(artifact))
        error = (torch.jit.load(str(artifact))(example) - model(example)).abs().max().item()
    if error > 1e-4:
        raise SystemExit(f"Exported model disagrees with eager model: {error}")
    manifest = {
        "source": {"run": str(args.run.resolve()), "checkpoint_sha256": summary["checkpoint_sha256"],
                   "dataset_spec_sha256": summary["dataset_spec_sha256"], "classes": summary["classes"],
                   "validation_threshold": summary["validation"]["threshold"]},
        "export": {"format": "torchscript", "input_shape": [1, window, 17, 3], "precision": "fp32",
                   "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()},
        "validation": {"max_absolute_error": error, "input": "uniform_rand_seed0"},
    }
    (args.output_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
