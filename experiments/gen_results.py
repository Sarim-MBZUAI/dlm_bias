import json, collections, subprocess, re, os
ROOT="/home/lukas/users/shashmi/dlm_bias"; os.chdir(ROOT)
PY="/home/lukas/miniconda3/envs/sarim_awm/bin/python"
M="experiments/results/matrix/"; S="experiments/results/sweep/"; UQ="datasets/unqover/results/"
OUT=[]; w=OUT.append

def cfg(path):
    try: return json.load(open(path)).get("config",{})
    except: return None
def rates(f, restrict=None):
    c=collections.Counter(); n=0
    for l in open(f):
        r=json.loads(l)
        if r.get("context_condition")!="ambig": continue
        if restrict is not None and (r.get("example_id"),r.get("question_index")) not in restrict: continue
        c[r["pred_class"]]+=1; n+=1
    return (n,c["target"]/n,c["non-target"]/n,c["unknown"]/n,c["no_answer"]/n) if n else (0,0,0,0,0)
def samp(jsonfile): return jsonfile.replace(".json","_samples.jsonl")

# clean baselines
n0,b0,nb0,ab0,na0=rates(M+"bbq_clean_samples.jsonl")

w("# Results — current runs (auto-generated; numbers read from result files)\n")
w("`[RUNNING]` = job not finished yet. Regenerate to refresh.\n")

# ---------- GLOBAL CONFIG ----------
import torch
d=torch.load("directional_steering/race_black_anchored_text.pt",map_location="cpu")
w("## Global configuration (identical across all runs)\n")
w("| item | value |")
w("|---|---|")
w("| model | LLaDA-8B-Instruct (frozen, bf16) |")
w("| decoding | gen_length=32, steps=64, block_length=32, temperature=0.0, remasking=low_confidence, cfg_scale=0.0 |")
w("| constants | MASK_ID=126336, PAD=126081 |")
w("| BBQ eval set | 1600 Black-referent ambiguous items (seed 42); `experiments/data/black_referent_ambig_eval.jsonl` |")
w("| UNQOVER | ethnicity, 2000 instances = 8000 records (limit 2000, seed 42) |")
w(f"| steering direction | `race_black_anchored_text.pt` — method={d['method']}, built at **layer {int(d['layer'])}**, n_pairs={int(d['n_pairs'])} |")
w(f"| direction quality | split-half cosine={d['splithalf_cosine']:.3f}, raw_norm={d['raw_norm']:.3f}, ‖mean_diff‖={d['mean_diff_norm']:.2f}, norm_ratio={d['norm_ratio']:.3f}, avg_act_norm={d['avg_act_norm']:.2f} |")
w(f"| black tags | {', '.join(d['black_tags'])} |")
w("| c* semantics | single-layer: **absolute** target; all-layer (`--layers all`): per-layer **offset** `c*_ℓ = a_nat_ℓ + offset` (a_nat range [−3.6,14.8], so offset 2 ⇒ c* mean ≈1.44) |")
w("| d_gap | (Δblack − Δnonblk) vs clean, on Black-referent ambiguous items |")
w(f"\nClean baseline (n={n0}): black={b0:.3f} nonblk={nb0:.3f} abstain={ab0:.3f} no_ans={na0:.3f}\n")

# ---------- BBQ n=1600 full matrix ----------
w("## 1. BBQ (n=1600) — every hyperparameter per row\n")
w("| method | site | steer_mode | α | c* | β | norm | black | nonblk | abstain | no_ans | d_gap |")
w("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|")
# (label, file, intended: site, mode, alpha, cstar, beta, norm)
BBQ=[("clean","bbq_clean.json","—","add","0","—","—","no"),
 ("additive","bbq_L14_add_a4.json","L14","add","4","—","—","no"),
 ("additive","bbq_L14_add_a8.json","L14","add","8","—","—","no"),
 ("clamp (P)","bbq_L14_clamp_c60.json","L14","clamp","—","60 (abs)","—","no"),
 ("cmom (P+EMA)","bbq_L14_cmom_c60.json","L14","cmom","—","60 (abs)","0.8","no"),
 ("additive","bbq_additive_full.json","all 32","add","0.234","—","—","no"),
 ("additive","bbq_add_o6.json","all 32","add","0.703","—","—","no"),
 ("additive","bbq_add_o12.json","all 32","add","1.405","—","—","no"),
 ("additive","bbq_add_o24.json","all 32","add","2.810","—","—","no"),
 ("clamp (P)","bbq_ours_clamp_full.json","all 32","clamp","—","offset 2","—","no"),
 ("clamp (P)","bbq_clamp_o6.json","all 32","clamp","—","offset 6","—","no"),
 ("clamp (P)","bbq_clamp_o12.json","all 32","clamp","—","offset 12","—","no"),
 ("clamp (P)","bbq_clamp_o24.json","all 32","clamp","—","offset 24","—","no"),
 ("cmom (P+EMA)","bbq_ours_cmom_full.json","all 32","cmom","—","offset 2","0.8","no"),
 ("cmom (P+EMA)","bbq_cmom_o6.json","all 32","cmom","—","offset 6","0.8","no"),
 ("cmom (P+EMA)","bbq_cmom_o12.json","all 32","cmom","—","offset 12","0.8","no"),
 ("cmom (P+EMA)","bbq_cmom_o24.json","all 32","cmom","—","offset 24","0.8","no")]
for lab,f,site,mode,al,cs,be,nm in BBQ:
    p=M+f; sf=samp(p)
    if os.path.exists(sf):
        n,b,nb,ab,na=rates(sf); dg=(b-b0)-(nb-nb0)
        w(f"| {lab} | {site} | {mode} | {al} | {cs} | {be} | {nm} | {b:.3f} | {nb:.3f} | {ab:.3f} | {na:.3f} | {dg:+.3f} |")
    else:
        w(f"| {lab} | {site} | {mode} | {al} | {cs} | {be} | {nm} | — | — | — | — | **[RUNNING]** |")
w("")

# ---------- All-layer sweep 400 ----------
keys=set()
for l in open("experiments/data/_sweep400.jsonl"):
    r=json.loads(l); keys.add((r.get("example_id"),r.get("question_index")))
cn,cb,cnb,cab,cna=rates(M+"bbq_clean_samples.jsonl",keys)
w("## 2. All-layer strength sweep (n=400 subset) — additive vs clamp, matched strength\n")
w(f"Clean(subset): black={cb:.3f} nonblk={cnb:.3f} abstain={cab:.3f}. offset o ↔ α=o/‖v‖ (‖v‖=8.54).\n")
w("| offset | method | site | α | c* | black | nonblk | abstain | no_ans | d_gap |")
w("|---:|---|---|---|---|---:|---:|---:|---:|---:|")
for o,a in [(2,0.234),(6,0.703),(12,1.405),(24,2.810)]:
    for meth,f,al,cs in [("additive",f"add_o{o}.json",str(a),"—"),("clamp",f"clamp_o{o}.json","—",f"offset {o}")]:
        sf=samp(S+f)
        if os.path.exists(sf):
            n,b,nb,ab,na=rates(sf); dg=(b-cb)-(nb-cnb)
            w(f"| {o} | {meth} | all 32 | {al} | {cs} | {b:.3f} | {nb:.3f} | {ab:.3f} | {na:.3f} | {dg:+.3f} |")
        else:
            w(f"| {o} | {meth} | all 32 | {al} | {cs} | — | — | — | — | **[RUNNING]** |")
w("")

# ---------- UNQOVER ----------
def uq(f):
    r=subprocess.run([PY,"datasets/unqover/unqover_metric.py","--results",UQ+f,"--target-subject","African"],capture_output=True,text=True).stdout
    g=lambda p:(re.search(p,r) or [None,"NA"])[1]
    return (g(r"mu \(bias intensity\)\s*=\s*([\d.]+)"),g(r"mean\|C\| \(asymmetry\)\s*=\s*([\d.]+)"),
            g(r"eta \(count-based\)\s*=\s*([\d.]+)"),g(r"delta \(positional err\)=\s*([\d.]+)"),g(r"debiased=([+\-][\d.]+)"))
w("## 3. UNQOVER — ethnicity (official metric, target=African)\n")
w("| method | site | steer_mode | α | c* | β | μ | mean\\|C\\| | η | δ | pref→African |")
w("|---|---|---|---|---|---|---:|---:|---:|---:|---:|")
UQROWS=[("clean","uq_clean.jsonl","—","add","0","—","—"),
 ("additive","uq_add_L14_a8.jsonl","L14","add","8","—","—"),
 ("additive","uq_add_all_a0234.jsonl","all 32","add","0.234","—","—"),
 ("clamp (P)","uq_ours_clamp_full.jsonl","all 32","clamp","—","offset 2","—"),
 ("cmom (P+EMA)","uq_ours_cmom_full.jsonl","all 32","cmom","—","offset 2","0.8"),
 ("clamp (P)","uq_L14_clamp_c60.jsonl","L14","clamp","—","60 (abs)","—"),
 ("cmom (P+EMA)","uq_L14_cmom_c60.jsonl","L14","cmom","—","60 (abs)","0.8")]
for lab,f,site,mode,al,cs,be in UQROWS:
    if os.path.exists(UQ+f):
        mu,mc,et,de,pr=uq(f)
        w(f"| {lab} | {site} | {mode} | {al} | {cs} | {be} | {mu} | {mc} | {et} | {de} | {pr} |")
    else:
        w(f"| {lab} | {site} | {mode} | {al} | {cs} | {be} | — | — | — | — | **[RUNNING]** |")
w("")

# ---------- Layer sweep ----------
w("## 4. Layer sweep — anchored direction built per layer, matched open-loop (unit-norm, α=8)\n")
w("Config per row: `--layer L --alpha 8 --normalize-direction` (unit push=8 at each layer, so d_gap-vs-layer is a fair site comparison). Eval on 400-subset; peak confirmed at n=1600.\n")
w("| layer | split-half | black | nonblk | abstain | d_gap |")
w("|---:|---:|---:|---:|---:|---:|")
LS="experiments/results/layersweep/"
for L in [4,8,10,12,14,16,20,24]:
    f=LS+f"L{L}_add_a8_samples.jsonl"
    ptf=f"directional_steering/race_black_anchored_text{'' if L==14 else '_L'+str(L)}.pt"
    sh="?"
    try: sh=f"{float(torch.load(ptf,map_location='cpu')['splithalf_cosine']):.3f}"
    except: pass
    if os.path.exists(f):
        n,b,nb,ab,na=rates(f); dg=(b-cb)-(nb-cnb)
        w(f"| {L} | {sh} | {b:.3f} | {nb:.3f} | {ab:.3f} | {dg:+.3f} |")
    else:
        w(f"| {L} | {sh} | — | — | — | **[RUNNING]** |")
w("")
w("## Notes")
w("- 'ours' = closed-loop (clamp=P / cmom=P+EMA). Single-layer L14 is the config with the promising pilot (d_gap +0.135/+0.216 on n=37); all-layer is the weaker ablation.")
w("- Per-sample files: `experiments/results/matrix/*_samples.jsonl`, `experiments/results/sweep/*_samples.jsonl`, `experiments/results/layersweep/*_samples.jsonl`, `datasets/unqover/results/*.jsonl`.")
open("experiments/RESULTS.md","w").write("\n".join(OUT)+"\n")
print("wrote experiments/RESULTS.md")
print("\n".join(OUT))
