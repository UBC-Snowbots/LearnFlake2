# KitPackaging — 2026-11-07

Build from the repository root after `orion_bench/live_demo.py` and `docs/PORTABLE.md` are present:

```sh
python scripts/package_live_kit.py --output /absolute/path/orion-live-kit.tar.gz
```

The destination directory must already exist. The builder fails if a required input is missing, and atomically replaces the destination when successful. It includes only Python sources under `orion_bench/`, the two selected TorchScript models and their manifests, and `docs/PORTABLE.md` as the kit's `README.md`. It does not package datasets, runs, checkpoints, Python, or downloaded dependencies. GPU/runtime binaries are not treated as portable; the target needs compatible dependencies installed separately as described in the kit README.

Archive tree (`*.py` denotes the package's Python sources, including the live entrypoint):

```text
orion-live-kit/
  README.md
  SHA256SUMS
  orion_bench/
    *.py
  models/
    yolo11n_pose_640/
      model_manifest.json
      yolo11n-pose.torchscript
    attention_gcn_w48_s0/
      model_manifest.json
      attention_gcn.torchscript
```

To inspect and verify after transfer (requires `tar` and `sha256sum`):

```sh
tar -tzf /absolute/path/orion-live-kit.tar.gz
tar -xzf /absolute/path/orion-live-kit.tar.gz
(cd orion-live-kit && sha256sum -c SHA256SUMS)
```

`SHA256SUMS` covers all payload files, not itself, and detects accidental payload corruption but is not an authenticity signature. Archive member paths, ordering and metadata are normalized for reproducible archives when inputs are unchanged. No build, test, inference, or Jetson execution was run by its author; the integration owner should verify the archive and target setup.
