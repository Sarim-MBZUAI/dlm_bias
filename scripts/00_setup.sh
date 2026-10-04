#!/usr/bin/env bash
# Models, BBQ source data, item/rotation checks, SocialStigmaQA preparation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
HF_CLI=${HF_CLI:-huggingface-cli}

# Models -> repo-root dirs expected by baselines/common.py, dream/common_dream.py,
# llada_moe/common_lladamoe.py.
dl_model() {  # $1 = HF repo id, $2 = local dir
    if [[ -f "$2/config.json" ]]; then echo "SKIP (exists): $2"; return; fi
    "$HF_CLI" download "$1" --local-dir "$2"
}
dl_model GSAI-ML/LLaDA-8B-Instruct               LLaDA-8B-Instruct
dl_model Dream-org/Dream-v0-Instruct-7B          Dream-v0-Instruct-7B
dl_model inclusionAI/LLaDA-MoE-7B-A1B-Instruct   LLaDA-MoE-7B-A1B-Instruct

# BBQ source jsonl (nyu-mll/BBQ) -> data/bbq_cache/<Category>.jsonl (eval/bbq_eval.py cache).
$PY - <<'EOF'
import os, urllib.request
base = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"
cats = ["Age", "Disability_status", "Gender_identity", "Nationality",
        "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
        "Religion", "SES", "Sexual_orientation"]
os.makedirs("data/bbq_cache", exist_ok=True)
for c in cats:
    p = os.path.join("data/bbq_cache", c + ".jsonl")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        print("SKIP (exists):", p); continue
    urllib.request.urlretrieve(f"{base}/{c}.jsonl", p); print("wrote", p)
EOF

# Item sets. The 400-item Black pool data/bbq_items/_sweep400.jsonl is included.
# Per-target pools are built by multirace/make_items.py.
[[ -f data/bbq_items/_sweep400.jsonl ]] || { echo "missing data/bbq_items/_sweep400.jsonl"; exit 1; }
[[ -f data/bbq_items/_sweep400_arab.jsonl ]] || $PY multirace/make_items.py
[[ -f data/bbq_items/_sweep400_old.jsonl  ]] || $PY multirace/make_items.py --category Age

# Black rotations -> results/balanced/_sweep400_rot{0,1,2}.jsonl, + offline oracle check.
[[ -f results/balanced/_sweep400_rot2.jsonl ]] || $PY eval/balanced/make_rotations.py
$PY eval/balanced/oracle_test.py

# Per-target rotations + oracle check.
ROTDIR=results/balanced_all/rotations
for T in arab asian latino white old; do
    [[ -f "$ROTDIR/_sweep400_${T}_rot2.jsonl" ]] || \
        $PY balanced_all/make_rotations.py --items "data/bbq_items/_sweep400_${T}.jsonl" --out-dir "$ROTDIR"
    $PY balanced_all/oracle_test.py --items-glob "$ROTDIR/_sweep400_${T}_rot*.jsonl" \
        --target "$T" --orig "data/bbq_items/_sweep400_${T}.jsonl"
done

# SocialStigmaQA source (pinned revision) -> data/socialstigma/source, then derive items.
if [[ ! -f data/socialstigma/source/yes_no/full_dataset_yes_no.csv ]]; then
    "$HF_CLI" download ibm-research/SocialStigmaQA --repo-type dataset \
        --revision e53d65d53bc74b7079a2a57f97e35c73d6f9fba4 --local-dir data/socialstigma/source
fi
$PY -m socialstigma.prepare --check

echo "DONE (setup)"
