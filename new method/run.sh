#!/usr/bin/env bash
# Run from any directory; choose the installed research interpreter with PYTHON.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python interpreter not found: %s\nSet PYTHON to your installed dlm environment interpreter.\n' "$python_bin" >&2
    exit 127
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
exec "$python_bin" -u "$script_dir/run.py" "$@"
