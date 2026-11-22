#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CONFIG.json" >&2
    exit 2
fi
config=$(realpath -e -- "$1")
if [[ ! -f "$config" ]]; then
    echo "Not a regular config file: $config" >&2
    exit 2
fi
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"
python=$(command -v python3)
if [[ -n "${BENCH_GPU_WRAPPER:-}" ]]; then
    wrapper=$(command -v -- "$BENCH_GPU_WRAPPER")
    exec tsp "$wrapper" "$python" -m orion_bench run --config "$config"
fi
exec tsp "$python" -m orion_bench run --config "$config"
