#!/usr/bin/env python
"""dream/denoise_pid.py -- FAITHFUL decode-space PID controller for Dream-v0-Instruct-7B.

Port of steering/denoise_pid.py (LLaDA) to Dream. CONTROL AXIS = the DIFFUSION
DENOISING STEP t = 0 .. STEPS-1 (NOT layer depth). Per item we run Dream's OWN
sampler (model.diffusion_generate, alg="entropy") ONCE and close the loop through
its NATIVE per-step hooks:

    p_black(t) = P(target letter token) at the answer slot logits[0, plen, :]
    e(t)       = s* - p_black(t)                       (s* = SETPOINT = 0.9)
    alpha(t)   = clamp( Kp*e + Ki*Iacc + Kd*d , 0, amax )   (imported PID law)

ACTUATOR = ALL 28 DreamDecoderLayers. The SAME alpha(t)*vhat is added to every
block residual each step (vhat = unit(arrows["r"][14])). Direction from ONE layer,
actuated at all 28 -- mirrors the LLaDA design so decode-PID is comparable to the
`normal` open-loop baseline (constant alpha vs feedback alpha(t)).

REUSE (imported verbatim from steering/denoise_pid.py -- the PID math is NOT
reimplemented): PID, COND_MASK, SETPOINT, ALPHA_MAX, pid_alphas_closedform,
p_black_from_logits, letter_token_ids. Steering plumbing uses common_dream.

CLOSED LOOP via Dream's native hooks (verified against generation_utils.py):
  generation_logits_hook_func(step, x, logits) fires AFTER the sampler's next-token
  shift (generation_utils.py: `logits = cat([logits[:,:1], logits[:,:-1]])`), so the
  logits it receives are ALREADY SHIFTED: logits[0, plen] is the distribution FOR the
  first generated token. We read p_black there with NO extra shift, call
  controller.update(p_meas), and write steerer.alpha for the NEXT forward.
  alpha starts 0, so step 0 is a natural unsteered probe; each command responds to the
  measurement one step behind it (the LLaDA one-step lag). Forwards == steps.

TRAJECTORY convention: pblack_traj[t] is measured from forward t (t=0 = unsteered
probe); alpha_traj[t] is the command issued after observing pblack_traj[t] (it drives
forward t+1). So alpha_traj[t] is the controller's response to pblack_traj[t].

MODES
    --selftest                     offline PID-math + anti-windup check (no GPU).
    --smoke [--limit N]            GPU smoke; prints alpha(t)/p_black(t) trajectory.
    --cond {base,P,PI,PID} ...     limited/full eval.
GPU RULE: run ONLY on the GPU assigned to this lane (CUDA_VISIBLE_DEVICES=3).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # ensure `import common_dream` resolves here

import common_dream as C  # noqa: E402


# steering/denoise_pid.py shares this file's basename -> load it by absolute path to
# avoid the name clash (sys.path[0] is dream/), exactly like build_arrows.py does.
def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LD = _load("llada_denoise_pid", "steering/denoise_pid.py")
PID = _LD.PID
COND_MASK = _LD.COND_MASK
SETPOINT = _LD.SETPOINT
ALPHA_MAX = _LD.ALPHA_MAX
pid_alphas_closedform = _LD.pid_alphas_closedform
p_black_from_logits = _LD.p_black_from_logits
letter_token_ids = _LD.letter_token_ids

LAYER = 14                          # vhat SOURCE layer (direction only; actuator = all 28)
STEPS = C.GEN_DEFAULTS["steps"]     # 64 denoising steps == forward count
LETTERS = C.LETTERS
DEFAULT_ARROWS = os.path.join(ROOT, "dream", "arrows.pt")
RESULTS = os.path.join(ROOT, "results", "dream", "decode_pid")


# --------------------------------------------------------------------------- #
# All-layer steerer: add a SETTABLE alpha*vhat to EVERY block residual (all pos).
# The 28 hooks share ONE mutable alpha, so one assignment steers the whole stack.
# Uses common_dream's tuple contract (hidden_from_output / output_with_hidden).
# --------------------------------------------------------------------------- #
class AllLayerSteerer:
    def __init__(self, vhat):
        self.vhat = vhat            # (H,) unit
        self.alpha = 0.0
        self.fires = 0
        self.handles = []

    def _hook(self, mod, inp, out):
        self.fires += 1
        if self.alpha == 0.0:
            return None
        h = C.hidden_from_output(out)
        steer = (self.alpha * self.vhat).to(h.dtype).to(h.device)
        return C.output_with_hidden(out, h + steer)

    def attach(self, model):
        paths = [C.block_path(k) for k in range(C.N_LAYERS)]
        self.handles = C.attach(model, paths, self._hook, pre=False)
        assert len(self.handles) == C.N_LAYERS

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------- #
# Controlled generation: Dream sampler + per-step decode-space PID via native hook.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def controlled_generate(model, steerer, controller, ids, tgt_plain, tgt_space, steer_on):
    plen = ids.shape[1]
    controller.reset()
    alpha_traj = np.zeros(STEPS, dtype=np.float64)
    pblack_traj = np.zeros(STEPS, dtype=np.float64)
    sat_traj = np.zeros(STEPS, dtype=bool)

    steerer.alpha = 0.0             # step-0 forward is an unsteered probe
    steerer.fires = 0

    def logits_hook(step, x, logits):
        if step is None or step >= STEPS:
            return logits          # (token hook passes step=None; ignore)
        # logits are ALREADY next-token-shifted (see module docstring): NO extra shift.
        p_meas = p_black_from_logits(logits, plen, tgt_plain, tgt_space)
        alpha, e, _, _, sat = controller.update(p_meas)
        cmd = alpha if steer_on else 0.0
        pblack_traj[step] = p_meas         # measured from THIS forward
        alpha_traj[step] = cmd             # command issued after observing it (drives step+1)
        sat_traj[step] = sat
        steerer.alpha = cmd                # apply on the NEXT forward
        return logits

    cfg = dict(C.GEN_DEFAULTS)
    out = model.diffusion_generate(
        ids, attention_mask=None, output_history=False, return_dict_in_generate=True,
        generation_logits_hook_func=logits_hook, **cfg)
    seq = out.sequences[:, plen:]
    return seq, alpha_traj, pblack_traj, sat_traj


# --------------------------------------------------------------------------- #
# Offline self-test: imported PID stateful output == pid_alphas_closedform, clamp
# bounds, integral freeze under saturation.  No GPU, no model.
# --------------------------------------------------------------------------- #
def selftest():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-dpid] {name:48s} : {'PASS' if cond else 'FAIL'}")

    amax = ALPHA_MAX
    grids = {"P": (1.5, 0.0, 0.0), "PI": (1.5, 0.30, 0.0), "PID": (1.5, 0.30, 0.50)}
    e_synth = rng.uniform(-0.5, 0.9, size=STEPS)   # sign changes exercise D + clamp
    for name, (kp, ki, kd) in grids.items():
        ctrl = PID(kp, ki, kd, SETPOINT, amax, antiwindup=False)
        ctrl.reset()
        got = [ctrl.update(SETPOINT - e)[0] for e in e_synth]   # feed p=s*-e => sees e
        want = pid_alphas_closedform(e_synth, kp, ki, kd, amax)
        check(f"{name} stateful == closedform (no-AW)", np.allclose(np.array(got), want, atol=1e-9))

    hi = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); hi.reset()
    check("clamp HIGH -> amax", all(hi.update(SETPOINT - 0.9)[0] == amax for _ in range(10)))
    lo = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); lo.reset()
    check("clamp LOW  -> 0.0", all(lo.update(SETPOINT + 0.9)[0] == 0.0 for _ in range(10)))

    aw = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=True); aw.reset()
    for _ in range(20):
        aw.update(SETPOINT - 0.9)          # persistent saturating +error -> freeze
    naw = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); naw.reset()
    for _ in range(20):
        naw.update(SETPOINT - 0.9)
    check(f"anti-windup FREEZES integral (iacc={aw.iacc:.3f}~0)", abs(aw.iacc) < 1e-9)
    check(f"no-AW WINDS UP integral (iacc={naw.iacc:.3f}>15)", naw.iacc > 15.0)

    print(f"[selftest-dpid] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Arrows / items.
# --------------------------------------------------------------------------- #
def load_vhat(dummy_arrows):
    if dummy_arrows:
        torch.manual_seed(1234)
        r14 = torch.randn(C.D_MODEL, dtype=torch.float32)
    else:
        blob = torch.load(DEFAULT_ARROWS, map_location="cpu")
        r14 = blob["r"][LAYER].to(torch.float32)
    return r14 / r14.norm().clamp(min=1e-12)


def load_items(limit, items_path):
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    return rows[:limit] if limit else rows


def build_input(tok, row, device):
    prompt = C.build_prompt(row)
    ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    return torch.tensor(tok(ptxt)["input_ids"], device=device).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Eval / smoke shared driver.
# --------------------------------------------------------------------------- #
def _run(cond, kp, ki, kd, amax, limit, items_path, out_dir, tag, dummy_arrows, smoke):
    use_ki, use_kd = COND_MASK[cond]
    eff_ki = ki if use_ki else 0.0
    eff_kd = kd if use_kd else 0.0
    steer_on = cond != "base"

    model, tok = C.load_model_tok()
    vhat = load_vhat(dummy_arrows).to(model.device)
    plain, space = letter_token_ids(tok)
    steerer = AllLayerSteerer(vhat)
    steerer.attach(model)
    ctrl = PID(kp, eff_ki, eff_kd, SETPOINT, amax, antiwindup=True)
    rows = load_items(limit, items_path)
    print(f"[{cond}] dev={torch.cuda.get_device_name(0)} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} n={len(rows)} actuator=all{C.N_LAYERS} "
          f"Kp={kp} Ki={eff_ki} Kd={eff_kd} s*={SETPOINT} amax={amax} steer_on={steer_on} "
          f"dummy={dummy_arrows} steps={STEPS}", flush=True)

    per_item, counts = [], {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    dis_n = dis_correct = 0
    alpha_sums, alpha_means, alpha_finals, sat_fracs = [], [], [], []
    hook_fires = 0                  # run total (steerer.fires is reset per item)
    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            bidx = C.black_idx_of(row)
            uidx = C.unk_idx_of(row)
            tgt = chr(65 + bidx) if bidx is not None else "A"
            ids = build_input(tok, row, model.device)
            seq, a, pb, sat = controlled_generate(
                model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=steer_on)
            hook_fires += steerer.fires   # counts EVERY invocation (bump precedes the
                                          # alpha==0 short-circuit): expect n*steps*28
                                          # in ALL conditions, base included.
            gen = tok.batch_decode(seq, skip_special_tokens=True)[0].strip()
            letter = C.parse_letter(gen, row)
            pred = LETTERS.index(letter) if letter else None
            if pred is None:
                cls = "unparseable"
            elif pred == bidx:
                cls = "black"
            elif pred == uidx:
                cls = "abstain"
            else:
                cls = "nonblack"
            counts[cls] += 1
            if row.get("context_condition") == "disambig" and pred is not None:
                dis_n += 1
                dis_correct += int(pred == int(row["label"]))
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
                "ground_truth_text": ([row.get("ans0"), row.get("ans1"), row.get("ans2")][row["label"]]
                                      if isinstance(row.get("label"), int) and 0 <= row["label"] < 3 else None),
                "prompt": C.build_prompt(row),
                "black_idx": bidx, "unk_idx": uidx, "target_letter": tgt,
                "pred_index": pred, "pred_letter": letter, "model_output": gen, "pred_class": cls,
                "alpha_sum": float(a.sum()), "alpha_mean": float(a.mean()),
                "alpha_final": float(a[-1]), "sat_frac": float(sat.mean()),
                "p_black_final": float(pb[-1]),
                "alpha_traj": [round(v, 4) for v in a.tolist()],
                "pblack_traj": [round(v, 4) for v in pb.tolist()],
            })
            if smoke:
                fwd = steerer.fires // C.N_LAYERS
                print(f"\n[smoke] item {idx} ex={row.get('example_id')} target={tgt} "
                      f"parsed='{letter}' cls={cls}", flush=True)
                print(f"[smoke]   forwards={fwd} (expect {STEPS}) gen={gen!r}", flush=True)
                if idx == 0:
                    print(f"[smoke]   alpha(t) : {[round(v,2) for v in a.tolist()]}", flush=True)
                    print(f"[smoke]   pblack(t): {[round(v,3) for v in pb.tolist()]}", flush=True)
                    print(f"[smoke]   alpha min={a.min():.2f} max={a.max():.2f} "
                          f"sat={int(sat.sum())}/{STEPS} varies="
                          f"{'YES' if a.max()-a.min() > 1e-3 else 'NO'}", flush=True)
            elif (idx + 1) % 50 == 0:
                print(f"[{cond}] {idx+1}/{len(rows)} b={counts['black']} nb={counts['nonblack']} "
                      f"ab={counts['abstain']} un={counts['unparseable']} ({time.time()-t0:.0f}s)",
                      flush=True)
    finally:
        steerer.detach()

    if smoke:
        print("\n[smoke] DONE", flush=True)
        return None

    n = len(rows)
    rates = {
        "black_pick_rate": counts["black"] / n, "nonblack_pick_rate": counts["nonblack"] / n,
        "abstain_rate": counts["abstain"] / n, "unparseable_rate": counts["unparseable"] / n,
    }
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "condition": cond, "model": "Dream-v0-Instruct-7B", "n": n,
        "counts": counts, "rates": rates,
        # gap_within = within-run aim (black - nonblack). d_gap keeps the run_items /
        # baselines/common.py semantics (black - baseline_black_rate); no baseline is
        # passed here, so it is null -- compare gap_within across conditions instead.
        "gap_within": rates["black_pick_rate"] - rates["nonblack_pick_rate"],
        "d_gap": None, "baseline_black_rate": None,
        "acc_disambig": (dis_correct / dis_n) if dis_n else None, "n_disambig": dis_n,
        "gains": {"Kp": kp, "Ki": eff_ki, "Kd": eff_kd},
        "actuator": f"all_{C.N_LAYERS}_layers", "anti_windup": True,
        "setpoint": SETPOINT, "alpha_max": amax, "vhat_layer": LAYER,
        "mean_total_actuation": float(np.mean(alpha_sums)) if alpha_sums else 0.0,
        "mean_alpha": float(np.mean(alpha_means)) if alpha_means else 0.0,
        "mean_final_alpha": float(np.mean(alpha_finals)) if alpha_finals else 0.0,
        "mean_sat_frac": float(np.mean(sat_fracs)) if sat_fracs else 0.0,
        "hook_fire_count": hook_fires,
        "gen_defaults": C.GEN_DEFAULTS, "steps": STEPS, "dummy_arrows": bool(dummy_arrows),
        "items_path": items_path, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_s": time.time() - t0,
    }
    stem = tag if tag else f"dpid_{cond}"
    with open(os.path.join(out_dir, f"cond_{stem}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{stem}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond}] DONE rates={rates} gap_within={result['gap_within']:.3f} "
          f"mean_alpha={result['mean_alpha']:.2f} final_alpha={result['mean_final_alpha']:.2f} "
          f"sat={result['mean_sat_frac']:.2f} fires={hook_fires} -> cond_{stem}.json", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cond", choices=list(COND_MASK.keys()))
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--amax", type=float, default=ALPHA_MAX)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--items", default=C.SWEEP400)
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--tag", default=None, help="output stem override: cond_<tag>.json")
    ap.add_argument("--dummy-arrows", action="store_true",
                    help="use a random (3584,) vhat (GPU smoke only; arrows.pt not built yet)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.smoke:
        cond = args.cond or "PI"
        _run(cond, args.kp, args.ki, args.kd, args.amax, args.limit or 2, args.items,
             args.out_dir, args.tag, args.dummy_arrows, smoke=True)
        return
    if not args.cond:
        ap.error("--cond required (or use --selftest / --smoke)")
    _run(args.cond, args.kp, args.ki, args.kd, args.amax, args.limit, args.items,
         args.out_dir, args.tag, args.dummy_arrows, smoke=False)


if __name__ == "__main__":
    main()
