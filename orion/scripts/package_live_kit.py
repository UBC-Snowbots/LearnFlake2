#!/usr/bin/env python3
"""Build the self-contained source/model portion of the portable live kit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path
import tarfile
import tempfile


REPO = Path(__file__).resolve().parent.parent
KIT_ROOT = "orion-live-kit"
REQUIRED = {
    "README.md": "docs/PORTABLE.md",
    "GESTURE_WORKFLOW.md": "docs/GESTURE_WORKFLOW.md",
    "orion_bench/__init__.py": "orion_bench/__init__.py",
    "orion_bench/live_demo.py": "orion_bench/live_demo.py",
    "models/yolo11n_pose_640/model_manifest.json": "models/yolo11n_pose_640/model_manifest.json",
    "models/yolo11n_pose_640/yolo11n-pose.torchscript": "models/yolo11n_pose_640/yolo11n-pose.torchscript",
    "models/attention_gcn_w48_s0/model_manifest.json": "models/attention_gcn_w48_s0/model_manifest.json",
    "models/attention_gcn_w48_s0/attention_gcn.torchscript": "models/attention_gcn_w48_s0/attention_gcn.torchscript",
}


def collect_files() -> dict[str, Path]:
    files = {destination: REPO / source for destination, source in REQUIRED.items()}
    for source in (REPO / "orion_bench").rglob("*.py"):
        files[source.relative_to(REPO).as_posix()] = source
    for destination, source in files.items():
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing or non-regular required file for {destination}: {source}")
    return dict(sorted(files.items()))


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: tarfile.TarFile, name: str, size: int, source: object) -> None:
    info = tarfile.TarInfo(f"{KIT_ROOT}/{name}")
    info.size = size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, source)


def build(output: Path) -> None:
    files = collect_files()
    if not output.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {output.parent}")
    sums = "".join(f"{checksum(path)}  {name}\n" for name, path in files.items()).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".orion-kit-", delete=False) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, path in files.items():
                        with path.open("rb") as source:
                            add_file(archive, name, os.fstat(source.fileno()).st_size, source)
                    add_file(archive, "SHA256SUMS", len(sums), io.BytesIO(sums))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="destination .tar.gz file")
    args = parser.parse_args()
    try:
        build(args.output)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Cannot create live kit: {exc}\n")
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
