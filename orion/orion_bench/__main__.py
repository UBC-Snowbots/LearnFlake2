"""Run Orion with: python -m orion_bench run --config CONFIG.json."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="orion_bench")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run and package a single benchmark")
    run.add_argument("--config", required=True, help="JSON configuration path")
    args = parser.parse_args()
    if args.command == "run":
        from .runner import load_config, run_benchmark
        try:
            return run_benchmark(load_config(args.config))
        except (ValueError, OSError) as exc:
            print(f"Cannot start benchmark: {exc}", file=sys.stderr)
            return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
