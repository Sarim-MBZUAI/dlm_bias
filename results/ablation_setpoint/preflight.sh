#!/usr/bin/env bash
# Step 0: preflight. Records the environment to env.json, checks that every
# git-ignored dependency exists, then runs the controller's built-in selftest
# and a 2-item GPU smoke at the primary setting. Exits non-zero on any failure.
#
#   bash results/ablation_setpoint/preflight.sh

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "== repo root: $ROOT"
COMMIT="$(git rev-parse HEAD)"
DIRTY="$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
echo "== commit: $COMMIT (modified tracked files: $DIRTY)"

GPU="$(gpu_name)"
echo "== gpu: $GPU"
if [[ "$GPU" == "unknown" ]]; then
  echo "nvidia-smi did not report a GPU. Run this on the GPU node." >&2
  exit 1
fi

VERS="$("$PY" - <<'EOF'
import json, sys
import torch, transformers, numpy
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": numpy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
EOF
)"
echo "== versions: $VERS"
if ! echo "$VERS" | grep -q '"transformers": "4.46.2"'; then
  echo "transformers must be 4.46.2 (requirements.txt)." >&2
  exit 1
fi

echo "== checking git-ignored dependencies"
ARROWS="${DLM_ARROWS_PATH:-steering/arrows.pt}"
missing=0
for p in LLaDA-8B-Instruct "$ARROWS" \
         "$ITEMS_DIR/_sweep400_rot0.jsonl" "$ITEMS_DIR/_sweep400_rot1.jsonl" "$ITEMS_DIR/_sweep400_rot2.jsonl" \
         "$REF_DIR/rot0/cond_dpid_PI_samples.jsonl" "$REF_DIR/rot1/cond_dpid_PI_samples.jsonl" "$REF_DIR/rot2/cond_dpid_PI_samples.jsonl"; do
  if [[ -e "$p" ]]; then
    echo "   ok      $p"
  else
    echo "   MISSING $p" >&2; missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  echo "See migrate_local_data.sh and ask the repository owner for the blobs. Do NOT rebuild arrows.pt." >&2
  exit 1
fi
for r in 0 1 2; do
  n="$(wc -l < "$ITEMS_DIR/_sweep400_rot$r.jsonl")"
  [[ "$n" == "400" ]] || { echo "_sweep400_rot$r.jsonl has $n lines, expected 400" >&2; exit 1; }
done

mkdir -p "$ABL"
"$PY" - "$ABL/env.json" "$COMMIT" "$DIRTY" "$GPU" "$VERS" "$ARROWS" <<'EOF'
import hashlib, json, os, sys, datetime
out, commit, dirty, gpu, vers, arrows = sys.argv[1:7]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
env = {
    "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "commit": commit,
    "modified_tracked_files": int(dirty),
    "gpu": gpu,
    "hostname": os.uname().nodename,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "arrows_path": arrows,
    "arrows_sha256": sha(arrows),
    "items_sha256": {f"_sweep400_rot{r}.jsonl": sha(f"results/balanced/_sweep400_rot{r}.jsonl") for r in range(3)},
    "fixed_args": "--cond PI --kp 3 --ki 0.1 --kd 0 --amax 6 (64 steps, sensor upper, oracle mapping)",
    **json.loads(vers),
}
json.dump(env, open(out, "w"), indent=2)
print(f"== wrote {out}")
EOF

echo "== controller selftest"
"$PY" "$CTRL" --selftest

echo "== 2-item GPU smoke at the primary setting"
"$PY" "$CTRL" --smoke --smoke-items 2 --kp 3 --ki 0.1 --kd 0 --setpoint 0.9

echo "PREFLIGHT PASSED"
