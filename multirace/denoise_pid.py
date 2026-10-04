#!/usr/bin/env python
"""multirace/denoise_pid.py -- target-parameterized decode-space PID for LLaDA-8B-Instruct.

Generalization of steering/denoise_pid.py from the fixed black target to any
--target in the registry. The PID math, the all-32-block actuator, the
observable, and the LLaDA sampler loop are IMPORTED from the original by file
path (basename clash -> importlib), not reimplemented:

    PID / COND_MASK / SETPOINT(0.9) / ALPHA_MAX(6.0) / pid_alphas_closedform
    p_black_from_logits   (target-agnostic: takes a letter -> used as p_target)
    letter_token_ids / AllLayerSteerer / controlled_generate
    load_model_tok / build_input / attach_all_layers

Differences from the original:
  * target letter = chr(65 + target_idx_of(row, target))   (targets.py registry)
  * items default  data/bbq_items/_sweep400_<target>.jsonl
  * arrows default multirace/arrows_<target>.pt (RAW (32,4096) key 'r';
    vhat = unit(r[14]), broadcast to all 32 blocks)
  * per-item classes: target / nontarget / abstain / unparseable
  * result JSON: target_pick_rate / nontarget_pick_rate, gap_within = target -
    nontarget, d_gap = target_pick_rate - baseline_rate (null unless
    --baseline-rate passed); config records the target and its tag set.
Default gains: Kp=3 Ki=0.1 Kd=1 amax=6. Rows without the target are skipped
(counted in n_skipped_no_target).

--sensor-case {upper,both} is passed through to letter_token_ids: 'upper'
(default) reads only uppercase letter tokens; 'both' also adds the lowercase
letter tokens so the loop sees lowercase answers.

MODES
    --selftest                          offline: imported PID == closed-form (no GPU).
    --smoke --dummy-arrows --limit 2    GPU smoke; prints alpha(t)/p_target(t).
    --target T --cond {base,P,PI,PID}   eval, e.g.
        python multirace/denoise_pid.py --target asian --cond PI
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common_eval as CE  # noqa: E402

_LD = CE.load_by_path("llada_denoise_pid", "steering/denoise_pid.py")
PID = _LD.PID
COND_MASK = _LD.COND_MASK
SETPOINT = _LD.SETPOINT
ALPHA_MAX = _LD.ALPHA_MAX
pid_alphas_closedform = _LD.pid_alphas_closedform
p_target_from_logits = _LD.p_black_from_logits      # target-agnostic: takes a letter
letter_token_ids = _LD.letter_token_ids
controlled_generate = _LD.controlled_generate
attach_all_layers = _LD.attach_all_layers
build_input = _LD.build_input
load_model_tok = _LD.load_model_tok

STEPS = _LD.STEPS
N_LAYERS = CE.N_LAYERS
LETTERS = CE.LETTERS

# Sensor-case passthrough. Sourced from the imported module so the option list
# cannot drift; if the steering module lacks SENSOR_CASES only 'upper' is offered.
SENSOR_CASES = tuple(getattr(_LD, "SENSOR_CASES", ("upper",)))


def sensor_ids(tok, sensor_case):
    """letter_token_ids with the sensor-case flag; 'upper' uses the 1-arg call
    so it also works with a steering module that lacks the flag."""
    if sensor_case == "upper":
        return letter_token_ids(tok)
    return letter_token_ids(tok, sensor_case)


def default_out(target):
    return os.path.join(CE.ROOT, "results", "multirace", target, "decode_pid")


# --------------------------------------------------------------------------- #
# Offline selftest: imported PID stateful == closed-form (no GPU).
# --------------------------------------------------------------------------- #
def selftest():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-mrpid] {name:48s} : {'PASS' if cond else 'FAIL'}")

    amax = ALPHA_MAX
    grids = {"P": (3.0, 0.0, 0.0), "PI": (3.0, 0.1, 0.0), "PID": (3.0, 0.1, 1.0)}
    e_synth = rng.uniform(-0.5, 0.9, size=STEPS)
    for name, (kp, ki, kd) in grids.items():
        ctrl = PID(kp, ki, kd, SETPOINT, amax, antiwindup=False)
        ctrl.reset()
        got = [ctrl.update(SETPOINT - e)[0] for e in e_synth]
        want = pid_alphas_closedform(e_synth, kp, ki, kd, amax)
        check(f"{name} stateful == closedform (no-AW)",
              np.allclose(np.array(got), want, atol=1e-9))

    hi = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); hi.reset()
    check("clamp HIGH -> amax", all(hi.update(SETPOINT - 0.9)[0] == amax for _ in range(10)))
    lo = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); lo.reset()
    check("clamp LOW  -> 0.0", all(lo.update(SETPOINT + 0.9)[0] == 0.0 for _ in range(10)))
    aw = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=True); aw.reset()
    for _ in range(20):
        aw.update(SETPOINT - 0.9)
    check(f"anti-windup FREEZES integral (iacc={aw.iacc:.3f}~0)", abs(aw.iacc) < 1e-9)
    print(f"[selftest-mrpid] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Eval / smoke driver (mirrors steering/denoise_pid.run, target-parameterized).
# --------------------------------------------------------------------------- #
def run(target, cond, kp, ki, kd, amax, limit, items_path, arrows_path, out_dir,
        tag, dummy_arrows, baseline_rate, smoke, sensor_case="upper"):
    CE.warn_if_fallback()
    use_ki, use_kd = COND_MASK[cond]
    eff_ki = ki if use_ki else 0.0
    eff_kd = kd if use_kd else 0.0
    steer_on = cond != "base"

    model, tok = load_model_tok()
    r, arrows_meta = CE.load_r(arrows_path, dummy_arrows)
    vhat = CE.vhat_of(r).to(model.device)
    plain, space = sensor_ids(tok, sensor_case)
    steerer = attach_all_layers(model, vhat)
    ctrl = PID(kp, eff_ki, eff_kd, SETPOINT, amax, antiwindup=True)
    rows = CE.load_items(items_path, limit)
    print(f"[{cond}:{target}] dev={torch.cuda.get_device_name(0)} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} n={len(rows)} "
          f"actuator=all{N_LAYERS} Kp={kp} Ki={eff_ki} Kd={eff_kd} s*={SETPOINT} "
          f"amax={amax} steer_on={steer_on} dummy={dummy_arrows} "
          f"sensor_case={sensor_case} items={items_path}",
          flush=True)

    per_item = []
    counts = {"target": 0, "nontarget": 0, "abstain": 0, "unparseable": 0}
    n_skipped = 0
    alpha_sums, alpha_means, alpha_finals, sat_fracs = [], [], [], []
    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            tidx, uidx = CE.target_idx_of(row, target), CE.unk_idx_of(row)
            if tidx is None:
                n_skipped += 1
                continue
            tgt = chr(65 + tidx)
            ids = build_input(tok, row, model.device)
            steerer.fires = 0
            x, a, pt, sat = controlled_generate(
                model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=steer_on)
            gen = tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = _LD.B.parse_letter(gen, row)
            pred = LETTERS.index(letter) if letter else None
            cls = CE.classify(pred, tidx, uidx)
            counts[cls] += 1
            alpha_sums.append(float(a.sum())); alpha_means.append(float(a.mean()))
            alpha_finals.append(float(a[-1])); sat_fracs.append(float(sat.mean()))
            per_item.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "category": row.get("category"),
                "question_polarity": row.get("question_polarity"),
                "context_condition": row.get("context_condition"),
                "context": row.get("context"), "question": row.get("question"),
                "options": {"A": row.get("ans0"), "B": row.get("ans1"), "C": row.get("ans2")},
                "ground_truth": row.get("label"),
                "prompt": _LD.B.build_prompt(row), "target": target,
                "target_idx": tidx, "unk_idx": uidx, "target_letter": tgt,
                "sensor_case": sensor_case,
                "pred_index": pred, "pred_letter": letter, "pred_class": cls,
                "model_output": gen, "alpha_sum": float(a.sum()),
                "alpha_mean": float(a.mean()), "alpha_final": float(a[-1]),
                "sat_frac": float(sat.mean()), "p_target_final": float(pt[-1]),
                "alpha_traj": [round(v, 4) for v in a.tolist()],
                "ptarget_traj": [round(v, 4) for v in pt.tolist()],
            })
            if smoke:
                fwd = steerer.fires // N_LAYERS
                print(f"\n[smoke] item {idx} ex={row.get('example_id')} target={target} "
                      f"letter={tgt} parsed='{letter}' cls={cls}", flush=True)
                print(f"[smoke]   forwards={fwd} (expect 1 probe + {STEPS} = {STEPS + 1}) "
                      f"gen={gen!r}", flush=True)
                if idx == 0:
                    print(f"[smoke]   alpha(t)   : {[round(v,2) for v in a.tolist()]}", flush=True)
                    print(f"[smoke]   p_target(t): {[round(v,3) for v in pt.tolist()]}", flush=True)
                    print(f"[smoke]   alpha min={a.min():.2f} max={a.max():.2f} "
                          f"sat={int(sat.sum())}/{STEPS} varies="
                          f"{'YES' if a.max()-a.min() > 1e-3 else 'NO'}", flush=True)
            elif (idx + 1) % 50 == 0:
                print(f"[{cond}:{target}] {idx+1}/{len(rows)} t={counts['target']} "
                      f"nt={counts['nontarget']} ab={counts['abstain']} "
                      f"un={counts['unparseable']} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        steerer.detach()

    if smoke:
        print("\n[smoke] DONE", flush=True)
        return None

    n = len(per_item)
    rates = CE.rates_block(counts, max(n, 1))
    result = {
        "condition": cond, "target": target,
        "config": {"target": target, "target_tags": sorted(CE.TARGET_TAGS[target]),
                   "fallback_registry": CE.USING_FALLBACK_TARGETS,
                   "arrows": arrows_meta, "items_path": items_path},
        "gains": {"Kp": kp, "Ki": eff_ki, "Kd": eff_kd},
        "actuator": f"all_{N_LAYERS}_layers", "anti_windup": True,
        "setpoint": SETPOINT, "alpha_max": amax, "vhat_layer": CE.LAYER,
        "sensor_case": sensor_case,
        "sensor_letter_ids": {"plain": plain, "space": space},
        "n": n, "n_skipped_no_target": n_skipped, "counts": counts, "rates": rates,
        "gap_within": rates["target_pick_rate"] - rates["nontarget_pick_rate"],
        "d_gap": (rates["target_pick_rate"] - baseline_rate
                  if baseline_rate is not None else None),
        "baseline_rate": baseline_rate,
        "mean_total_actuation": float(np.mean(alpha_sums)) if alpha_sums else 0.0,
        "mean_alpha": float(np.mean(alpha_means)) if alpha_means else 0.0,
        "mean_final_alpha": float(np.mean(alpha_finals)) if alpha_finals else 0.0,
        "mean_sat_frac": float(np.mean(sat_fracs)) if sat_fracs else 0.0,
        "gen_length": _LD.GEN_LENGTH, "steps": STEPS, "block_length": _LD.BLOCK_LENGTH,
        "dummy_arrows": bool(dummy_arrows),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(out_dir, exist_ok=True)
    stem = tag if tag else f"dpid_{target}_{cond}"
    with open(os.path.join(out_dir, f"cond_{stem}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{stem}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond}:{target}] DONE rates={rates} gap_within={result['gap_within']:.3f} "
          f"mean_alpha={result['mean_alpha']:.2f} sat={result['mean_sat_frac']:.2f} "
          f"skipped={n_skipped} -> cond_{stem}.json", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", choices=CE.TARGETS)
    ap.add_argument("--cond", choices=list(COND_MASK.keys()))
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--amax", type=float, default=ALPHA_MAX)
    ap.add_argument("--sensor-case", choices=list(SENSOR_CASES), default="upper",
                    help="passthrough to steering/denoise_pid.letter_token_ids: "
                         "'upper' (default, original case-blind sensor) or "
                         "'both' (+ lowercase letter tokens)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--items", default=None,
                    help="default data/bbq_items/_sweep400_<target>.jsonl")
    ap.add_argument("--arrows", default=None,
                    help="default multirace/arrows_<target>.pt")
    ap.add_argument("--out-dir", default=None,
                    help="default results/multirace/<target>/decode_pid")
    ap.add_argument("--tag", default=None, help="output stem override: cond_<tag>.json")
    ap.add_argument("--dummy-arrows", action="store_true")
    ap.add_argument("--baseline-rate", type=float, default=None,
                    help="base-condition target_pick_rate; enables d_gap")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if not args.target:
        ap.error("--target required (or use --selftest)")
    cond = args.cond or ("PI" if args.smoke else None)
    if not cond:
        ap.error("--cond required (or use --smoke)")
    run(args.target, cond, args.kp, args.ki, args.kd, args.amax,
        args.limit or (2 if args.smoke else 0),
        args.items or CE.default_items(args.target),
        args.arrows or CE.default_arrows(args.target),
        args.out_dir or default_out(args.target),
        args.tag, args.dummy_arrows, args.baseline_rate, smoke=args.smoke,
        sensor_case=args.sensor_case)


if __name__ == "__main__":
    main()
