# Shared shell helpers for the setpoint ablation. Source this file; do not run it.
#
#   source results/ablation_setpoint/common.sh
#
# Resolves the repo root, picks the Python interpreter, and provides
# `run_one`, which runs a single controller invocation, times it, logs the
# outcome to timings.tsv, and skips runs whose output already exists so a
# failed job can be rerun without redoing finished conditions.

set -euo pipefail

ABL_DIR_REL="results/ablation_setpoint"

if [[ -n "${DLM_BIAS_ROOT:-}" ]]; then
  ROOT="$DLM_BIAS_ROOT"
else
  ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
fi
export DLM_BIAS_ROOT="$ROOT"
cd "$ROOT"

# Interpreter: honour $PY (the repo's sbatch convention), else whatever `python` is.
PY="${PY:-python}"

ABL="$ROOT/$ABL_DIR_REL"
CTRL="steering/denoise_pid.py"
ITEMS_DIR="results/balanced"
REF_DIR="results/balanced/results_balanced"
TIMINGS="$ABL/timings.tsv"
FAILURES="$ABL/failures.log"

# Fixed hyperparameters. Only --setpoint varies in this ablation.
PI_ARGS=(--cond PI --kp 3 --ki 0.1 --kd 0 --amax 6)

gpu_name() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | sed 's/^ *//;s/ *$//' || echo "unknown"
}

# setpoint_tag 0.9 -> dpid_PI_s0p9
setpoint_tag() { echo "dpid_PI_s$(echo "$1" | tr . p)"; }

ensure_timings_header() {
  if [[ ! -f "$TIMINGS" ]]; then
    mkdir -p "$ABL"
    printf 'tier\trot\ttag\tstart_utc\telapsed_s\texit_code\tgpu\n' > "$TIMINGS"
  fi
}

# run_one TIER ROT TAG OUT_DIR -- <denoise_pid.py args...>
# Skips when cond_<TAG>_samples.jsonl already exists in OUT_DIR.
run_one() {
  local tier="$1" rot="$2" tag="$3" out_dir="$4"; shift 4
  [[ "$1" == "--" ]] && shift
  ensure_timings_header
  mkdir -p "$out_dir"
  if [[ -s "$out_dir/cond_${tag}_samples.jsonl" && -s "$out_dir/cond_${tag}.json" ]]; then
    echo "[skip] $tier rot$rot $tag (output exists)"
    return 0
  fi
  local start; start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local t0; t0="$(date +%s)"
  local code=0
  echo "[run ] $tier rot$rot $tag :: $PY $CTRL $* --out-dir $out_dir --tag $tag"
  "$PY" "$CTRL" "$@" --out-dir "$out_dir" --tag "$tag" || code=$?
  local elapsed=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tier" "$rot" "$tag" "$start" "$elapsed" "$code" "$(gpu_name)" >> "$TIMINGS"
  if [[ $code -ne 0 ]]; then
    echo "$start $tier rot$rot $tag exit=$code" >> "$FAILURES"
    echo "[FAIL] $tier rot$rot $tag exit=$code (logged to $FAILURES)" >&2
    return $code
  fi
  echo "[done] $tier rot$rot $tag ${elapsed}s"
}
