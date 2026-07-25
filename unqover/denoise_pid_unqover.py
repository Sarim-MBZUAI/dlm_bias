#!/usr/bin/env python
"""Decode-space PID controller (from steering/denoise_pid.py) on UNQOVER.

THIN ADAPTER. All control machinery is IMPORTED from steering.denoise_pid:
    PID                  -- discrete PID on scalar observable, anti-windup
    AllLayerSteerer      -- adds alpha(t)*vhat to ALL 32 block residuals
    controlled_generate  -- LLaDA block-diffusion sampler + per-step decode PID
    p_black_from_logits  -- P(target letter) = P(plain)+P(space) at gen pos 0
    letter_token_ids     -- letter -> (plain_id, space_id)
    load_model_tok / load_vhat / attach_all_layers  -- model & actuator setup
    COND_MASK/SETPOINT/ALPHA_MAX/STEPS/N_LAYERS      -- controller config
Nothing about the controller/actuator changes; only the BBQ glue is swapped for
UNQOVER glue (2-choice A/B prompt, per-item target letter, UNQOVER record schema).

UNQOVER vs BBQ glue differences:
  * Prompt          : unqover_eval.build_prompt(item) -- 2-choice A/B, NO unknown.
  * Target letter   : "A" if subj0==TARGET else "B" (the pair order flips per item).
  * Observable      : P(target letter) at gen pos 0 via the IMPORTED
                      p_black_from_logits (full-softmax sum of that letter's
                      plain+space tokens -- identical measurement to BBQ, only the
                      letter that counts as "target" differs). NOT renormalised
                      over {A,B}: kept identical to BBQ so the controller is unchanged.
  * Pre-filter      : keep only instances whose subject pair CONTAINS the target
                      (subj0==TARGET or subj1==TARGET); ~2/15 subjects are the
                      target, so unfiltered generation is mostly wasted.
  * Records         : EXACTLY unqover_eval.py's schema (so unqover_metric.py
                      consumes them unchanged) + controller diagnostics as extras.

GPU RULE for the eventual run: use ONLY an idle GPU (this repo commonly runs on 0/2);
here the caller will pick the allowed device. This file's --selftest is CPU-only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))               # unqover
_ROOT = "/home/lukas/users/shashmi/dlm_bias"

# Reuse the decode-space PID machinery verbatim (adds eval/ to path via denoise_pid).
sys.path.insert(0, os.path.join(_ROOT, "steering"))
import denoise_pid as D  # noqa: E402
from denoise_pid import (  # noqa: E402  -- explicit reuse, no reimplementation
    PID, AllLayerSteerer, controlled_generate, p_black_from_logits,
    letter_token_ids, pid_alphas_closedform,
    load_model_tok, load_vhat, attach_all_layers,
    COND_MASK, SETPOINT, ALPHA_MAX, STEPS, N_LAYERS,
)

# Reuse the UNQOVER prompt + parser (2-choice A/B, no unknown).
sys.path.insert(0, _HERE)
import unqover_eval as U  # noqa: E402

LETTERS = ["A", "B"]                                             # UNQOVER: two subjects
DEFAULT_ITEMS = os.path.join(_ROOT, "data", "unqover", "ethnicity.items.jsonl")
DEFAULT_TARGET = "Black"
DEFAULT_OUT_DIR = os.path.join(_ROOT, "results", "unqover", "denoise_pid")

# Fields carried straight from the item into the record (unqover_eval.py schema).
ITEM_FIELDS = ("id", "instance_id", "uqid", "bias_class", "qid", "polarity",
               "subj0", "subj1", "tid", "act_cluster", "obj0", "obj1",
               "s_cluster0", "s_cluster1")


# --------------------------------------------------------------------------- #
# UNQOVER glue.
# --------------------------------------------------------------------------- #
def target_letter(item, target):
    """The answer letter that means TARGET for this item. subj0 is choice A,
    subj1 is choice B, and the pair order flips per item."""
    return "A" if item["subj0"] == target else "B"


def load_target_items(items_path, target, limit=0):
    """Read items, keep only those whose subject pair contains `target`
    (subj0==target or subj1==target), grouped so each instance's 4 records stay
    contiguous. limit>0 caps the number of items (use a multiple of 4 to keep
    whole instances for the metric)."""
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    kept = [r for r in rows if r["subj0"] == target or r["subj1"] == target]
    # group by instance_id to keep the 4 records of an instance contiguous
    order, buckets = [], {}
    for r in kept:
        iid = r["instance_id"]
        if iid not in buckets:
            buckets[iid] = []
            order.append(iid)
        buckets[iid].append(r)
    grouped = [r for iid in order for r in buckets[iid]]
    return grouped[:limit] if limit else grouped


def build_input(tok, item, device):
    """UNQOVER 2-choice prompt -> chat-templated token ids (same wrap as bbq)."""
    prompt = U.build_prompt(item)
    ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    return torch.tensor(tok(ptxt)["input_ids"], device=device).unsqueeze(0)


def make_record(item, target, pick, gen, diag):
    """Build ONE result record in unqover_eval.py's EXACT schema + diagnostics.
    unqover_metric.py needs: instance_id, subj0, subj1, qid, act_cluster,
    pred_subject (grouping + hard-pick score); everything else is passthrough."""
    rec = {k: item[k] for k in ITEM_FIELDS}
    rec.update({
        "pred_index": pick,                                       # 0=subj0/A, 1=subj1/B
        "pred_letter": (LETTERS[pick] if pick is not None else None),
        "pred_subject": (item[f"subj{pick}"] if pick is not None else None),
        "model_output": gen,
        # ---- human-readable item (so records are self-inspectable) ----
        "context": item.get("context"),
        "question": item.get("question"),
        "choices": item.get("choices"),
        "prompt": U.build_prompt(item),
        # ---- controller diagnostics (extra fields; ignored by the metric) ----
        "target_subject": target,
        "target_letter": diag.get("target_letter"),
        "alpha_sum": diag.get("alpha_sum"),
        "alpha_mean": diag.get("alpha_mean"),
        "alpha_final": diag.get("alpha_final"),
        "sat_frac": diag.get("sat_frac"),
        "p_target_final": diag.get("p_target_final"),
        "alpha_traj": diag.get("alpha_traj"),
        "ptarget_traj": diag.get("ptarget_traj"),
    })
    return rec


# --------------------------------------------------------------------------- #
# GPU eval (NOT run by the gate; mirrors denoise_pid.run with UNQOVER glue).
# --------------------------------------------------------------------------- #
def run(cond, kp, ki, kd, amax, limit, items_path, target, out_path):
    use_ki, use_kd = COND_MASK[cond]
    eff_ki = ki if use_ki else 0.0
    eff_kd = kd if use_kd else 0.0
    steer_on = cond != "base"

    model, tok = load_model_tok()
    vhat = load_vhat().to(model.device)
    plain, space = letter_token_ids(tok)
    steerer = attach_all_layers(model, vhat)
    ctrl = PID(kp, eff_ki, eff_kd, SETPOINT, amax, antiwindup=True)
    items = load_target_items(items_path, target, limit)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"[{cond}] items={items_path} target={target!r} n={len(items)} "
          f"dev={torch.cuda.get_device_name(0)} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"actuator=all{N_LAYERS} Kp={kp} Ki={eff_ki} Kd={eff_kd} s*={SETPOINT} "
          f"amax={amax} steer_on={steer_on} AW=on", flush=True)

    no_answer = 0
    t0 = time.time()
    try:
        with open(out_path, "w") as out_fh:
            for idx, item in enumerate(items):
                tgt = target_letter(item, target)
                ids = build_input(tok, item, model.device)
                x, a, pt, sat = controlled_generate(
                    model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=steer_on)
                gen = tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
                pick = U.parse_choice(gen, item)
                if pick is None:
                    no_answer += 1
                diag = {
                    "target_letter": tgt,
                    "alpha_sum": float(a.sum()), "alpha_mean": float(a.mean()),
                    "alpha_final": float(a[-1]), "sat_frac": float(sat.mean()),
                    "p_target_final": float(pt[-1]),
                    "alpha_traj": [round(v, 4) for v in a.tolist()],
                    "ptarget_traj": [round(v, 4) for v in pt.tolist()],
                }
                out_fh.write(json.dumps(make_record(item, target, pick, gen, diag)) + "\n")
                if (idx + 1) % 50 == 0:
                    print(f"[{cond}] {idx+1}/{len(items)} no_answer={no_answer} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        steerer.detach()

    cfg = {"condition": cond, "gains": {"Kp": kp, "Ki": eff_ki, "Kd": eff_kd},
           "actuator": f"all_{N_LAYERS}_layers", "anti_windup": True,
           "setpoint": SETPOINT, "alpha_max": amax, "target_subject": target,
           "items": items_path, "n_items": len(items), "no_answer": no_answer,
           "steer_on": steer_on, "gen_length": D.GEN_LENGTH, "steps": STEPS,
           "block_length": D.BLOCK_LENGTH,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "elapsed_s": time.time() - t0}
    cfg_path = out_path[:-6] + ".json" if out_path.endswith(".jsonl") else out_path + ".json"
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"[{cond}] DONE -> {out_path}  (no_answer={no_answer}/{len(items)})", flush=True)
    print("Analyze with unqover_metric.py --results " + out_path, flush=True)


# --------------------------------------------------------------------------- #
# Offline self-test (NO model, NO GPU).
# --------------------------------------------------------------------------- #
def selftest(items_path, target):
    ok_all = True

    # (1) target-letter logic ------------------------------------------------ #
    a_item = {"subj0": target, "subj1": "Other"}
    b_item = {"subj0": "Other", "subj1": target}
    la, lb = target_letter(a_item, target), target_letter(b_item, target)
    t1 = (la == "A" and lb == "B")
    ok_all &= t1
    print(f"[selftest] target-letter: subj0==target -> {la!r} (want 'A'); "
          f"subj1==target -> {lb!r} (want 'B'): {'PASS' if t1 else 'FAIL'}")

    # (2) real prompts + target letter --------------------------------------- #
    items = load_target_items(items_path, target, limit=0)
    n_items = len(items)
    n_inst = len({it["instance_id"] for it in items})
    print(f"[selftest] pre-filter: {n_items} items / {n_inst} instances contain "
          f"target={target!r} (of 8000 items / 2000 instances)")
    # verify every kept item actually contains the target, and letter is consistent
    contains_ok = all(it["subj0"] == target or it["subj1"] == target for it in items)
    letter_ok = all(
        (target_letter(it, target) == "A") == (it["subj0"] == target) for it in items)
    ok_all &= contains_ok and letter_ok
    print(f"[selftest] every kept item contains target: {'PASS' if contains_ok else 'FAIL'}; "
          f"letter matches subj-order: {'PASS' if letter_ok else 'FAIL'}")
    sample = items[0]
    tgt = target_letter(sample, target)
    prompt = U.build_prompt(sample)
    n_choices = sum(1 for ln in prompt.splitlines() if ln[:2] in ("A.", "B.", "C."))
    two_choice = n_choices == 2 and "C." not in prompt
    ok_all &= two_choice
    print(f"[selftest] example prompt (2-choice A/B: {'PASS' if two_choice else 'FAIL'}), "
          f"subj0={sample['subj0']!r} subj1={sample['subj1']!r} -> target letter {tgt!r} "
          f"({'A=subj0' if tgt=='A' else 'B=subj1'} is the target):")
    for ln in prompt.splitlines():
        print("           | " + ln)
    # round-trip: if the model 'picked' the target letter, parse -> target subject
    picked = U.parse_choice(tgt, sample)
    rt_ok = sample[f"subj{picked}"] == target
    ok_all &= rt_ok
    print(f"[selftest] parse_choice({tgt!r}) -> index {picked} -> "
          f"subject {sample[f'subj{picked}']!r} == target: {'PASS' if rt_ok else 'FAIL'}")

    # (3) record schema covers everything unqover_metric.py needs ------------ #
    diag = {"target_letter": tgt, "alpha_sum": 1.0, "alpha_mean": 0.5,
            "alpha_final": 0.4, "sat_frac": 0.1, "p_target_final": 0.8,
            "alpha_traj": [0.1, 0.2], "ptarget_traj": [0.3, 0.4]}
    rec = make_record(sample, target, picked, "A", diag)
    # keys unqover_metric.py touches: group_instances + instance_B_C + target_gap
    needed = ["instance_id", "subj0", "subj1", "qid", "act_cluster", "pred_subject"]
    missing = [k for k in needed if k not in rec]
    t3 = not missing
    ok_all &= t3
    print(f"[selftest] record has all metric-required fields {needed}: "
          f"{'PASS' if t3 else 'FAIL (missing %s)' % missing}")
    print(f"[selftest] record keys = {sorted(rec.keys())}")

    # (4) imported PID == denoise_pid closed-form on a synthetic e-sequence --- #
    rng = np.random.default_rng(0)
    e_synth = rng.uniform(-0.5, 0.9, size=STEPS)
    grids = {"P": (1.5, 0.0, 0.0), "PI": (1.5, 0.30, 0.0), "PID": (1.5, 0.30, 0.50)}
    for name, (kp, ki, kd) in grids.items():
        ctrl = PID(kp, ki, kd, SETPOINT, ALPHA_MAX, antiwindup=False)
        ctrl.reset()
        got = [ctrl.update(SETPOINT - e)[0] for e in e_synth]     # feed p=s*-e => sees e
        want = pid_alphas_closedform(e_synth, kp, ki, kd, ALPHA_MAX)
        match = np.allclose(np.array(got), want, atol=1e-9)
        ok_all &= match
        print(f"[selftest] imported PID {name:3s} == denoise_pid closed-form (no-AW): "
              f"{'PASS' if match else 'FAIL'}")

    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Decode-space PID controller on UNQOVER.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cond", choices=list(COND_MASK.keys()))
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--amax", type=float, default=ALPHA_MAX)
    ap.add_argument("--limit", type=int, default=0, help="0 = all target items")
    ap.add_argument("--items", default=DEFAULT_ITEMS)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--out", default=None, help="output jsonl (default under results/unqover/denoise_pid/)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(args.items, args.target) else 1)
    if not args.cond:
        ap.error("--cond required (or use --selftest)")
    out_path = args.out or os.path.join(DEFAULT_OUT_DIR, f"dpid_{args.cond}.jsonl")
    run(args.cond, args.kp, args.ki, args.kd, args.amax, args.limit,
        args.items, args.target, out_path)


if __name__ == "__main__":
    main()
