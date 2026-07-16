#!/usr/bin/env python
"""Layer-space PID (Eq.18) + normal single-vector baseline (from pid_steering/pid_steer.py)
on UNQOVER.

THIN ADAPTER. All steering machinery is IMPORTED from pid_steering.pid_steer:
    build_u                 -- layer-depth PID combine (Eq.18): u(k) from per-layer rhat
    build_injection         -- alpha * build_u(unit_rows(r), Kp,Ki,Kd)  (per-block vectors)
    build_normal_injection  -- vhat=unit(r[source_layer]) broadcast to ALL 32 blocks as alpha*vhat
    unit_rows               -- per-layer unit-normalisation of raw arrows
    AddVec                  -- per-block forward hook adding a FIXED vector to the residual
    GAINS / N_LAYERS        -- reference gains {base,P,PI,PID} and block count (32)
Generation reuses pid_steer.bbq_eval.generate (LLaDA masked-diffusion sampler) verbatim.

KEY difference from the DECODE-space adapter (denoise_pid_unqover.py): the steering here is
ITEM-INDEPENDENT. A fixed per-block injection is added at EVERY block on EVERY denoising step,
with NO per-item target-letter feedback. So there is no p_black / closed-loop control loop --
we just attach the fixed vectors, run the sampler, parse_choice -> pred_subject -> record.

  * --mode normal --alpha A : one fixed vector vhat=unit(r[14]), SAME at all 32 blocks (alpha*vhat).
  * --mode pid --cond C --alpha A : DISTINCT per-block u(k) from build_u (base = no hook).

UNQOVER glue (item load + target-filter, build_input, make_record, output schema) is REUSED
from denoise_pid_unqover.py, so records are unqover_metric.py-compatible unchanged.

GPU RULE: run on an idle allowed GPU (0/2/3/4). --selftest is CPU-only (loads arrows on CPU).
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))               # datasets/unqover
_ROOT = "/home/lukas/users/shashmi/dlm_bias"

# Reuse the layer-space PID / normal-vector machinery verbatim (adds eval/ to path).
sys.path.insert(0, os.path.join(_ROOT, "pid_steering"))
import pid_steer as P  # noqa: E402
from pid_steer import (  # noqa: E402  -- explicit reuse, no reimplementation
    build_u, build_injection, build_normal_injection, unit_rows,
    AddVec, GAINS, N_LAYERS, MODEL_PATH, DEFAULT_ARROWS,
)

# Reuse the UNQOVER glue (item load+filter, prompt build, record schema, parser).
sys.path.insert(0, _HERE)
import denoise_pid_unqover as G  # noqa: E402
from denoise_pid_unqover import (  # noqa: E402
    load_target_items, build_input, make_record, target_letter, LETTERS,
)
import unqover_eval as U  # noqa: E402

DEFAULT_ITEMS = os.path.join(_HERE, "data", "ethnicity.items.jsonl")
DEFAULT_TARGET = "Black"
DEFAULT_OUT_DIR = os.path.join(_HERE, "results_pid_steer")


# --------------------------------------------------------------------------- #
# Build the fixed per-block injection for a given mode/condition.
# Returns (inject: (N,H) tensor or None, label: str, gains: (kp,ki,kd)).
# --------------------------------------------------------------------------- #
def build_inject(mode, cond, alpha, r, source_layer):
    if mode == "normal":
        return build_normal_injection(r, source_layer, alpha), f"normalL{source_layer}", (1.0, 0.0, 0.0)
    # mode == "pid"
    if cond == "base":
        return None, "base", (0.0, 0.0, 0.0)
    kp, ki, kd = GAINS[cond]
    return build_injection(r, kp, ki, kd, alpha), cond, (kp, ki, kd)


# --------------------------------------------------------------------------- #
# GPU eval (mirrors pid_steer.run with UNQOVER glue; NOT run by the offline gate).
# --------------------------------------------------------------------------- #
def run(mode, cond, alpha, source_layer, arrows_path, limit, items_path, target,
        out_path, gen_len, steps, blk):
    blob = torch.load(arrows_path, map_location="cpu")
    r = blob["r"].to(torch.float32)
    inject, label, (kp, ki, kd) = build_inject(mode, cond, alpha, r, source_layer)
    steer_on = inject is not None

    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    items = load_target_items(items_path, target, limit)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[{label}] mode={mode} items={items_path} target={target!r} n={len(items)} "
          f"dev={torch.cuda.get_device_name(0)} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"actuator=all{N_LAYERS} alpha={alpha} Kp={kp} Ki={ki} Kd={kd} "
          f"src_layer={source_layer if mode=='normal' else '-'} steer_on={steer_on}", flush=True)

    fired = {}
    steerers = []
    if inject is not None:
        blocks = P.bbq_eval.resolve_module(model, P.bbq_eval.BLOCKS_PATH)
        assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"
        for li in range(N_LAYERS):
            s = AddVec(li, inject[li].to(model.device), fired)
            s.attach(blocks[li])
            steerers.append(s)

    no_answer = 0
    t0 = time.time()
    try:
        with open(out_path, "w") as out_fh:
            for idx, item in enumerate(items):
                ids = build_input(tok, item, model.device)
                out = P.bbq_eval.generate(model, ids, steps=steps, gen_length=gen_len,
                                          block_length=blk, temperature=0.0, cfg_scale=0.0,
                                          remasking="low_confidence")
                gen = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
                pick = U.parse_choice(gen, item)
                if pick is None:
                    no_answer += 1
                diag = {"target_letter": target_letter(item, target)}   # informational only
                out_fh.write(json.dumps(make_record(item, target, pick, gen, diag)) + "\n")
                if (idx + 1) % 50 == 0:
                    print(f"[{label}] {idx+1}/{len(items)} no_answer={no_answer} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        for s in steerers:
            s.detach()

    n_fired = len(fired)
    fire_counts = sorted(set(fired.values())) if fired else []
    cfg = {"mode": mode, "condition": label,
           "gains": {"Kp": kp, "Ki": ki, "Kd": kd}, "alpha": alpha,
           "source_layer": source_layer if mode == "normal" else None,
           "actuator": f"all_{N_LAYERS}_layers", "item_independent": True,
           "target_subject": target, "items": items_path, "n_items": len(items),
           "no_answer": no_answer, "steer_on": steer_on,
           "hooks_fired_layers": n_fired, "hook_fire_counts": fire_counts,
           "arrows": {"n_items": blob.get("n_items"), "source": blob.get("source")},
           "gen_length": gen_len, "steps": steps, "block_length": blk,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "elapsed_s": time.time() - t0}
    cfg_path = out_path[:-6] + ".json" if out_path.endswith(".jsonl") else out_path + ".json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[{label}] DONE -> {out_path}  (no_answer={no_answer}/{len(items)}) "
          f"hooks_fired_layers={n_fired}/{N_LAYERS} fire_counts={fire_counts}", flush=True)
    print("Analyze with unqover_metric.py --results " + out_path, flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test (NO model, NO GPU; loads arrows on CPU).
# --------------------------------------------------------------------------- #
def selftest(arrows_path, items_path, target, alpha=4.0, source_layer=14):
    ok_all = True
    blob = torch.load(arrows_path, map_location="cpu")
    r = blob["r"].to(torch.float32)
    print(f"[selftest] arrows {arrows_path}: r shape={tuple(r.shape)}")

    # (1) normal mode: identical vector at all 32 blocks == alpha * unit(r[L]) --- #
    inj_n, label_n, _ = build_inject("normal", None, alpha, r, source_layer)
    vhat = r[source_layer] / r[source_layer].norm()
    shape_ok = tuple(inj_n.shape) == (N_LAYERS, r.shape[1])
    ident_ok = all(torch.allclose(inj_n[k], inj_n[0], atol=1e-6) for k in range(N_LAYERS))
    eq_ok = torch.allclose(inj_n[0], alpha * vhat, atol=1e-6)
    t1 = shape_ok and ident_ok and eq_ok
    ok_all &= t1
    print(f"[selftest] normal(L{source_layer},a{alpha:g}): shape==(32,{r.shape[1]}) "
          f"{'PASS' if shape_ok else 'FAIL'}; all 32 identical "
          f"{'PASS' if ident_ok else 'FAIL'}; ==alpha*vhat "
          f"{'PASS' if eq_ok else 'FAIL'}")

    # (2) pid mode: DISTINCT per-layer vectors matching alpha*build_u(unit_rows(r)) -- #
    for cond in ("P", "PI", "PID"):
        kp, ki, kd = GAINS[cond]
        inj_p, _, _ = build_inject("pid", cond, alpha, r, source_layer)
        want = alpha * build_u(unit_rows(r), kp, ki, kd)
        match = torch.allclose(inj_p, want, atol=1e-6)
        distinct = not all(torch.allclose(inj_p[k], inj_p[0], atol=1e-6) for k in range(N_LAYERS))
        neq_normal = not torch.allclose(inj_p, inj_n, atol=1e-3)
        cok = match and distinct and neq_normal
        ok_all &= cok
        print(f"[selftest] pid {cond:3s}: ==alpha*build_u {'PASS' if match else 'FAIL'}; "
              f"per-layer distinct {'PASS' if distinct else 'FAIL'}; "
              f"!= normal vector {'PASS' if neq_normal else 'FAIL'}")
    # base = no hook
    inj_base, _, _ = build_inject("pid", "base", alpha, r, source_layer)
    tbase = inj_base is None
    ok_all &= tbase
    print(f"[selftest] pid base: inject is None (no hook) {'PASS' if tbase else 'FAIL'}")

    # (3) record dict has all unqover_metric-required fields --------------------- #
    items = load_target_items(items_path, target, limit=0)
    n_inst = len({it["instance_id"] for it in items})
    print(f"[selftest] target-filter: {len(items)} items / {n_inst} instances contain "
          f"target={target!r}")
    sample = items[0]
    tgt = target_letter(sample, target)
    picked = U.parse_choice(tgt, sample)
    diag = {"target_letter": tgt}
    rec = make_record(sample, target, picked, tgt, diag)
    needed = ["instance_id", "subj0", "subj1", "qid", "act_cluster", "pred_subject"]
    missing = [k for k in needed if k not in rec]
    t3 = not missing and rec["pred_subject"] == target
    ok_all &= t3
    print(f"[selftest] record has metric fields {needed}: "
          f"{'PASS' if not missing else 'FAIL (missing %s)' % missing}; "
          f"round-trip pred_subject=={target!r}: "
          f"{'PASS' if rec['pred_subject']==target else 'FAIL'}")
    print(f"[selftest] record keys = {sorted(rec.keys())}")

    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Layer-space PID + normal single-vector baseline on UNQOVER.")
    ap.add_argument("--selftest", action="store_true", help="offline check; no GPU")
    ap.add_argument("--mode", choices=["normal", "pid"], default="pid")
    ap.add_argument("--cond", choices=list(GAINS.keys()), help="PID condition (mode=pid)")
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--source-layer", type=int, default=14,
                    help="layer whose arrow is the fixed normal vector (mode=normal)")
    ap.add_argument("--arrows", default=DEFAULT_ARROWS)
    ap.add_argument("--items", default=DEFAULT_ITEMS)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--limit", type=int, default=0, help="0 = all target items")
    ap.add_argument("--out", default=None,
                    help="output jsonl (default under results_pid_steer/)")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(args.arrows, args.items, args.target,
                               args.alpha, args.source_layer) else 1)

    if args.mode == "pid" and not args.cond:
        ap.error("--cond required for --mode pid (or use --selftest)")

    if args.out:
        out_path = args.out
    elif args.mode == "normal":
        tag = f"normalL{args.source_layer}_a{args.alpha:g}".replace(".", "p")
        out_path = os.path.join(DEFAULT_OUT_DIR, f"{tag}.jsonl")
    else:
        tag = (args.cond if args.cond == "base"
               else f"pid_{args.cond}_a{args.alpha:g}".replace(".", "p"))
        out_path = os.path.join(DEFAULT_OUT_DIR, f"{tag}.jsonl")

    run(args.mode, args.cond, args.alpha, args.source_layer, args.arrows, args.limit,
        args.items, args.target, out_path, args.gen_length, args.steps, args.block_length)


if __name__ == "__main__":
    main()
