"""Combine benchmark artifacts from run directories and archives."""

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


ARTIFACTS = ("summary.json", "config.json", "model_info.json", "environment.json")


def _flatten(value, prefix, row):
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(child, f"{prefix}.{key}" if prefix else str(key), row)
    elif value is None or isinstance(value, (str, int, float, bool)):
        row[prefix] = value
    elif isinstance(value, list):
        # Shapes and similar small arrays are useful as a single CSV column.
        row[prefix] = json.dumps(value, separators=(",", ":"))


def _row(source, run, artifacts, errors):
    row = {"source": str(source), "run": run}
    for name in ARTIFACTS:
        data = artifacts.get(name)
        if data is None:
            errors.append(f"missing {name}")
            continue
        try:
            document = json.loads(data)
            if not isinstance(document, dict):
                raise ValueError("expected a JSON object")
            _flatten(document, name.removesuffix(".json"), row)
        except (ValueError, UnicodeError) as exc:
            errors.append(f"invalid {name}: {exc}")
    if errors:
        row["artifact_errors"] = "; ".join(errors)
    return row


def _directory(path):
    artifacts, errors = {}, []
    for name in ARTIFACTS:
        try:
            artifacts[name] = (path / name).read_bytes()
        except FileNotFoundError:
            pass
        except OSError as exc:
            errors.append(f"cannot read {name}: {exc}")
    return [_row(path, path.name, artifacts, errors)]


def _archive(path):
    groups = {}
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    name = PurePosixPath(member.filename)
                    if not member.is_dir() and name.name in ARTIFACTS:
                        groups.setdefault(str(name.parent), {})[name.name] = archive.read(member)
        else:
            with tarfile.open(path, "r:*") as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    if member.isfile() and name.name in ARTIFACTS:
                        file = archive.extractfile(member)
                        if file is not None:
                            with file:
                                groups.setdefault(str(name.parent), {})[name.name] = file.read()
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        if not groups:
            return [_row(path, path.name, {}, [f"cannot read archive: {exc}"])]
        return [_row(path, run, artifacts, [f"archive read error: {exc}"])
                for run, artifacts in sorted(groups.items())]
    if not groups:
        return [_row(path, path.name, {}, ["no benchmark artifacts in archive"])]
    return [_row(path, run if run != "." else path.name, artifacts, [])
            for run, artifacts in sorted(groups.items())]


def aggregate(paths):
    rows = []
    for path in map(Path, paths):
        if path.is_dir():
            rows.extend(_directory(path))
        elif path.is_file():
            rows.extend(_archive(path))
        else:
            rows.append(_row(path, path.name, {}, ["input does not exist or is inaccessible"]))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="run directories or ZIP/TAR archives")
    parser.add_argument("--output", required=True, type=Path, help="output .csv or .json")
    args = parser.parse_args(argv)
    suffix = args.output.suffix.lower()
    if suffix not in (".csv", ".json"):
        parser.error("--output must end in .csv or .json")
    rows = aggregate(args.paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        if suffix == ".json":
            json.dump(rows, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        else:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
