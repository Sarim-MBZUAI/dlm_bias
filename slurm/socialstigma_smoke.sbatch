#!/bin/bash
#SBATCH --job-name=ssqasmoke
#SBATCH --account=nils
#SBATCH --partition=gpu
#SBATCH --nodelist=mbz-titan-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.out
#SBATCH --error=/shared/home/sarim.hashmi/dlm_bias/logs/slurm/%x-%j.err

set -euo pipefail
export HF_HOME=/shared/models/huggingface
export HF_HUB_CACHE=/shared/models/huggingface/hub
export HF_MODULES_CACHE=/shared/home/sarim.hashmi/.cache/hf_modules
export DLM_BIAS_ROOT=/shared/home/sarim.hashmi/dlm_bias
cd "$DLM_BIAS_ROOT"
PY=/shared/home/sarim.hashmi/anaconda3/envs/dlm/bin/python
OUT=results/socialstigma/smoke

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

CONDS=(clean decode_pi64 decode_pid64 decode_pi32 normal_a4 normal_eff layer_pi_a2 \
       caa_a16 actadd_a16 meanact_s2 linearact_s1 aura_inject_g4 aura_vanilla itic_k48_a8)
for polarity in yes no; do
    for cond in "${CONDS[@]}"; do
        extra=()
        if [[ "$cond" == normal_eff ]]; then
            extra=(--effort-alpha 3.0)
        fi
        $PY -m socialstigma.run_one --condition "$cond" --polarity "$polarity" \
            --rotation 0 --limit 2 --out-root "$OUT" "${extra[@]}"
    done
done

echo "ALL DONE (SocialStigmaQA all 14 conditions x both polarities x 2 items)"
