#!/usr/bin/env python
"""FAITHFUL decode-space PID controller for LLaDA-8B-Instruct.

CONTROL AXIS = the DIFFUSION DENOISING STEP  t = 0 .. STEPS-1  (NOT layer depth).

Per item we run the LLaDA block-diffusion sampler (verbatim commit logic reused
from bbq_eval.generate). At every denoising step t we close a loop on the
decode-space observable p_black(t) and set ONE scalar actuator alpha(t):

    p_black(t) = P(target letter token) at the answer position (gen pos 0)
                 (plain + space variant summed; target letter = chr(65+black_idx))
    e(t)       = s* - p_black(t)                       (s* = setpoint, default 0.9)
    Iacc      += e(t)                                  (integral, w/ anti-windup)
    d(t)       = e(t) - e(t-1)                          (derivative, e(-1):=0)
    alpha(t)   = clamp( Kp*e(t) + Ki*Iacc + Kd*d(t), 0, alpha_max )

ACTUATOR = ALL 32 TRANSFORMER BLOCKS. The SAME alpha(t)*vhat is broadcast to every
block's residual output each step (vhat = unit(arrows.pt r[14])). This gives real
control authority AND makes decode-PID directly comparable to the normal open-loop
baseline (constant-alpha * the same vhat at all 32 layers): the ONLY difference
becomes constant alpha (open-loop) vs feedback-modulated alpha(t) (closed-loop).

ANTI-WINDUP: ON by default (conditional integration -- the integral accumulator is
frozen on any step whose command saturates at 0 or alpha_max). NOTE: the paper's
LAYER-space PID had no anti-windup; the decode-space controller adds this standard
term because the plant carries persistent error. Labelled, not hidden.

alpha >= 0: we only ever push TOWARD Black. Conditions differ ONLY in the gains:
    base = no controller (clean, alpha == 0 always)
    P    = (Ki=0, Kd=0)     PI = (Ki>0, Kd=0)     PID = (Ki>0, Kd>0)

TIMING: a single steered model forward per step. alpha(t) is computed from the most
recent available measurement p_black; seeded by ONE unsteered probe forward on the
initial all-masked sequence so alpha(0) reflects a real deficit (not a cold 0). Each
step's steered forward both (a) commits tokens and (b) yields p_black for the NEXT
step -- the loop observes the actual steered plant output. Forwards = 1 probe + STEPS.
No post-commit gating in the primary run (pure PID over all steps); gating is a
future variant.

Reuses bbq_eval: build_prompt / parse_letter / add_gumbel_noise /
get_num_transfer_tokens / resolve_module / hidden_from_output / output_with_hidden.

MODES
    --selftest                     offline PID-math + anti-windup check (no GPU).
    --smoke [--smoke-items N]      GPU smoke; prints alpha(t) trajectory.
    --cond {base,P,PI,PID} ...     full/limited eval (--limit 0 = all 400).
GPU RULE: run ONLY with CUDA_VISIBLE_DEVICES=0 or 2.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval as B  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "data", "bbq_items", "_sweep400.jsonl")
# arrows.pt is gitignored in the worktree -> always read the MAIN-tree copy.
DEFAULT_ARROWS = os.path.join(ROOT, "steering", "arrows.pt")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "decode_pid")

N_LAYERS = 32
LAYER = 14                       # vhat source layer (direction only; actuator = all 32)
GEN_LENGTH, STEPS, BLOCK_LENGTH = 32, 64, 32
TEMPERATURE, REMASKING = 0.0, "low_confidence"
SETPOINT, ALPHA_MAX = 0.9, 6.0
LETTERS = ["A", "B", "C"]
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

COND_MASK = {  # (use_ki, use_kd)
    "base": (False, False),
    "P":    (False, False),
    "PI":   (True,  False),
    "PID":  (True,  True),
}


# --------------------------------------------------------------------------- #
# The controller: discrete PID on the scalar observable p_black, anti-windup.
# --------------------------------------------------------------------------- #
def clamp(a, amax):
    return float(min(amax, max(0.0, a)))


class PID:
    def __init__(self, kp, ki, kd, setpoint, amax, antiwindup=True):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint, self.amax = setpoint, amax
        self.antiwindup = antiwindup

    def reset(self):
        self.iacc = 0.0
        self.eprev = 0.0

    def update(self, p_black):
        """One control step -> (alpha, e, integral, derivative, saturated).

        Anti-windup = conditional integration: the tentative integral (iacc+e) is
        COMMITTED only when the resulting command is inside (0, amax); if it
        saturates, the accumulator is frozen at its previous value.
        """
        e = self.setpoint - p_black
        d = e - self.eprev
        iacc_try = self.iacc + e
        raw = self.kp * e + self.ki * iacc_try + self.kd * d
        saturated = raw > self.amax or raw < 0.0
        if (not self.antiwindup) or (not saturated):
            self.iacc = iacc_try                 # accept integration
        # else: freeze integral (keep previous self.iacc)
        alpha = clamp(raw, self.amax)
        self.eprev = e
        return alpha, e, self.iacc, d, saturated


def pid_alphas_closedform(e_seq, kp, ki, kd, amax):
    """Vectorized reference (NO anti-windup): clip(Kp*e+Ki*cumsum(e)+Kd*diff(e),0,amax)."""
    e = np.asarray(e_seq, dtype=np.float64)
    cum = np.cumsum(e)
    d = np.diff(e, prepend=0.0)
    return np.clip(kp * e + ki * cum + kd * d, 0.0, amax)


# --------------------------------------------------------------------------- #
# All-layer steerer: add a settable alpha*vhat to EVERY block's residual (all pos).
# All 32 hooks share ONE alpha, so one assignment steers the whole stack per step.
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
        hidden = B.hidden_from_output(out)
        steer = (self.alpha * self.vhat).to(hidden.dtype).to(hidden.device)
        return B.output_with_hidden(out, hidden + steer)

    def attach(self, blocks):
        for blk in blocks:
            self.handles.append(blk.register_forward_hook(self._hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------- #
# BBQ helpers.
# --------------------------------------------------------------------------- #
def black_idx_of(row):
    for k in range(3):
        p = (row.get("answer_info") or {}).get(f"ans{k}")
        if p and str(p[-1]).strip().lower() in BLACK_TAGS:
            return k
    return None


def unk_idx_of(row):
    for k in range(3):
        p = (row.get("answer_info") or {}).get(f"ans{k}")
        if p and str(p[-1]).strip().lower() == "unknown":
            return k
    return None


def letter_token_ids(tok):
    plain, space = {}, {}
    for L in LETTERS:
        plain[L] = tok(L, add_special_tokens=False)["input_ids"][-1]
        space[L] = tok(" " + L, add_special_tokens=False)["input_ids"][-1]
    return plain, space


def p_black_from_logits(logits, prompt_len, tgt_plain, tgt_space):
    """P(target letter) = P(plain)+P(space) at gen pos 0 (the natural letter slot)."""
    row = logits[0, prompt_len + 0, :].float()
    p = torch.softmax(row, dim=-1)
    return float(p[tgt_plain].item() + p[tgt_space].item())


# --------------------------------------------------------------------------- #
# Controlled generation: LLaDA sampler + per-step decode-space PID.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def controlled_generate(model, steerer, controller, prompt, tgt_plain, tgt_space,
                        steer_on):
    mask_id = B.MASK_ID
    plen = prompt.shape[1]
    x = torch.full((1, plen + GEN_LENGTH), mask_id, dtype=torch.long, device=model.device)
    x[:, :plen] = prompt.clone()

    assert GEN_LENGTH % BLOCK_LENGTH == 0
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    assert STEPS % num_blocks == 0
    steps_per_block = STEPS // num_blocks

    controller.reset()
    alpha_traj = np.zeros(STEPS, dtype=np.float64)
    pblack_traj = np.zeros(STEPS, dtype=np.float64)
    sat_traj = np.zeros(STEPS, dtype=bool)

    # Seed measurement: one unsteered probe on the initial all-masked sequence.
    steerer.alpha = 0.0
    p_meas = p_black_from_logits(model(x).logits, plen, tgt_plain, tgt_space)

    t = 0
    for nb in range(num_blocks):
        b0 = plen + nb * BLOCK_LENGTH
        b1 = plen + (nb + 1) * BLOCK_LENGTH
        block_mask_index = x[:, b0:b1] == mask_id
        ntt = B.get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            alpha, e, _, _, sat = controller.update(p_meas)
            steerer.alpha = alpha if steer_on else 0.0

            mask_index = x == mask_id
            logits = model(x).logits

            p_meas = p_black_from_logits(logits, plen, tgt_plain, tgt_space)
            alpha_traj[t], pblack_traj[t], sat_traj[t] = steerer.alpha, p_meas, sat

            # --- commit (verbatim bbq_eval.generate low-confidence logic) ---
            x0 = torch.argmax(B.add_gumbel_noise(logits, TEMPERATURE), dim=-1)
            p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, b1:] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float("-inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = int(ntt[0, i])
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                transfer_index[0, sel] = True
            x[transfer_index] = x0[transfer_index]
            t += 1

    return x, alpha_traj, pblack_traj, sat_traj


# --------------------------------------------------------------------------- #
# Offline PID-math + anti-windup self-test (no GPU, no model).
# --------------------------------------------------------------------------- #
def selftest():
    rng = np.random.default_rng(0)
    ok_all = True
    grids = {"P": (1.5, 0.0, 0.0), "PI": (1.5, 0.30, 0.0), "PID": (1.5, 0.30, 0.50)}
    e_synth = rng.uniform(-0.5, 0.9, size=STEPS)   # sign changes exercise D + clamp
    amax = ALPHA_MAX
    for name, (kp, ki, kd) in grids.items():
        ctrl = PID(kp, ki, kd, SETPOINT, amax, antiwindup=False)
        ctrl.reset()
        got = [ctrl.update(SETPOINT - e)[0] for e in e_synth]  # feed p=s*-e => sees e
        want = pid_alphas_closedform(e_synth, kp, ki, kd, amax)
        match = np.allclose(np.array(got), want, atol=1e-9)
        ok_all &= match
        print(f"[selftest] {name:3s} (Kp={kp},Ki={ki},Kd={kd}) stateful==closedform "
              f"(no-AW): {'PASS' if match else 'FAIL'}")

    # Clamp bounds.
    hi = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); hi.reset()
    hi_ok = all(hi.update(SETPOINT - 0.9)[0] == amax for _ in range(10))
    lo = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); lo.reset()
    lo_ok = all(lo.update(SETPOINT + 0.9)[0] == 0.0 for _ in range(10))
    print(f"[selftest] clamp HIGH -> amax={amax}       : {'PASS' if hi_ok else 'FAIL'}")
    print(f"[selftest] clamp LOW  -> 0.0               : {'PASS' if lo_ok else 'FAIL'}")

    # Anti-windup: persistent saturating +error must FREEZE the integral (iacc stays 0
    # since step 0 already saturates), whereas no-AW winds it up unboundedly.
    aw = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=True); aw.reset()
    for _ in range(20):
        aw.update(SETPOINT - 0.9)          # e=0.9 -> raw huge -> saturate -> freeze
    naw = PID(100.0, 10.0, 0.0, SETPOINT, amax, antiwindup=False); naw.reset()
    for _ in range(20):
        naw.update(SETPOINT - 0.9)
    aw_ok = abs(aw.iacc) < 1e-9            # frozen at 0
    naw_ok = naw.iacc > 15.0              # 20*0.9 = 18 wound up
    print(f"[selftest] anti-windup FREEZES integral (iacc={aw.iacc:.3f}~0): "
          f"{'PASS' if aw_ok else 'FAIL'}")
    print(f"[selftest] no-AW WINDS UP integral (iacc={naw.iacc:.3f}>15): "
          f"{'PASS' if naw_ok else 'FAIL'}")
    # Anti-windup recovers FASTER than no-AW after overshoot: because AW capped the
    # integral, the command drops back sooner once error reverses (no-AW stays pegged).
    def after_overshoot(aw_flag):
        c = PID(4.0, 0.5, 0.0, SETPOINT, amax, antiwindup=aw_flag); c.reset()
        for _ in range(30):
            c.update(0.0)                  # sustained +error (saturates repeatedly)
        return c.update(1.0)[0]            # overshoot: p=1 -> e=-0.1
    a_aw, a_naw = after_overshoot(True), after_overshoot(False)
    rel_ok = a_aw < a_naw
    print(f"[selftest] anti-windup recovers faster than no-AW "
          f"(alpha_AW={a_aw:.2f} < alpha_noAW={a_naw:.2f}): {'PASS' if rel_ok else 'FAIL'}")

    ok_all &= hi_ok and lo_ok and aw_ok and naw_ok and rel_ok
    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


# --------------------------------------------------------------------------- #
# Model / data loading.
# --------------------------------------------------------------------------- #
def load_model_tok():
    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tok


def load_vhat():
    blob = torch.load(DEFAULT_ARROWS, map_location="cpu")
    r14 = blob["r"][LAYER].to(torch.float32)
    return r14 / r14.norm().clamp(min=1e-12)


def load_items(limit, items_path=SWEEP400):
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    return rows[:limit] if limit else rows


def build_input(tok, row, device):
    prompt = B.build_prompt(row)
    ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    return torch.tensor(tok(ptxt)["input_ids"], device=device).unsqueeze(0)


def attach_all_layers(model, vhat):
    blocks = B.resolve_module(model, B.BLOCKS_PATH)
    assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"
    steerer = AllLayerSteerer(vhat)
    steerer.attach(blocks)
    return steerer


# --------------------------------------------------------------------------- #
# GPU smoke.
# --------------------------------------------------------------------------- #
def smoke(n_items, kp, ki, kd, amax):
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[smoke] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    model, tok = load_model_tok()
    vhat = load_vhat().to(model.device)
    plain, space = letter_token_ids(tok)
    print(f"[smoke] letter tokens plain={plain} space={space}", flush=True)
    steerer = attach_all_layers(model, vhat)
    ctrl = PID(kp, ki, kd, SETPOINT, amax, antiwindup=True)
    rows = [r for r in load_items(0) if black_idx_of(r) is not None][:n_items]
    try:
        for n, row in enumerate(rows):
            bidx = black_idx_of(row)
            tgt = chr(65 + bidx)
            ids = build_input(tok, row, model.device)
            steerer.fires = 0
            x, a, pb, sat = controlled_generate(
                model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=True)
            gen = tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = B.parse_letter(gen, row)
            print(f"\n[smoke] item {n} ex={row.get('example_id')} target={tgt}", flush=True)
            print(f"[smoke]   forwards this item = {steerer.fires // N_LAYERS} "
                  f"(x{N_LAYERS} hooks; expect 1 probe + {STEPS} = {STEPS + 1})", flush=True)
            print(f"[smoke]   steps={len(a)} parsed='{letter}' gen={gen!r}", flush=True)
            if n == 0:
                print(f"[smoke]   alpha(t) : {[round(v,2) for v in a.tolist()]}", flush=True)
                print(f"[smoke]   p_black(t): {[round(v,3) for v in pb.tolist()]}", flush=True)
                print(f"[smoke]   alpha varies? min={a.min():.2f} max={a.max():.2f} "
                      f"n_distinct={len(set(round(v,3) for v in a.tolist()))} "
                      f"sat_steps={int(sat.sum())}/{STEPS} "
                      f"-> {'YES' if a.max()-a.min() > 1e-3 else 'NO'}", flush=True)
    finally:
        steerer.detach()
    print("\n[smoke] DONE", flush=True)


# --------------------------------------------------------------------------- #
# Full / limited eval over the sweep400.
# --------------------------------------------------------------------------- #
def run(cond, kp, ki, kd, amax, limit, out_dir, tag, items_path=SWEEP400):
    os.makedirs(out_dir, exist_ok=True)
    use_ki, use_kd = COND_MASK[cond]
    eff_ki = ki if use_ki else 0.0
    eff_kd = kd if use_kd else 0.0
    steer_on = cond != "base"

    model, tok = load_model_tok()
    vhat = load_vhat().to(model.device)
    plain, space = letter_token_ids(tok)
    steerer = attach_all_layers(model, vhat)
    ctrl = PID(kp, eff_ki, eff_kd, SETPOINT, amax, antiwindup=True)
    rows = load_items(limit, items_path)
    print(f"[{cond}] items={items_path}", flush=True)
    print(f"[{cond}] dev={torch.cuda.get_device_name(0)} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} n={len(rows)} actuator=all{N_LAYERS} "
          f"Kp={kp} Ki={eff_ki} Kd={eff_kd} s*={SETPOINT} amax={amax} "
          f"steer_on={steer_on} AW=on", flush=True)

    per_item, counts = [], {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    alpha_sums, alpha_means, alpha_finals, sat_fracs = [], [], [], []
    t0 = time.time()
    try:
        for idx, row in enumerate(rows):
            bidx, uidx = black_idx_of(row), unk_idx_of(row)
            tgt = chr(65 + bidx) if bidx is not None else "A"
            ids = build_input(tok, row, model.device)
            x, a, pb, sat = controlled_generate(
                model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=steer_on)
            gen = tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = B.parse_letter(gen, row)
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
            alpha_sums.append(float(a.sum()))
            alpha_means.append(float(a.mean()))
            alpha_finals.append(float(a[-1]))
            sat_fracs.append(float(sat.mean()))
            per_item.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "category": row.get("category"),
                "question_polarity": row.get("question_polarity"),
                "context_condition": row.get("context_condition"),
                "context": row.get("context"),
                "question": row.get("question"),
                "options": {"A": row.get("ans0"), "B": row.get("ans1"), "C": row.get("ans2")},
                "ground_truth": row.get("label"),
                "ground_truth_text": ([row.get("ans0"), row.get("ans1"), row.get("ans2")][row["label"]]
                                      if isinstance(row.get("label"), int) and 0 <= row["label"] < 3 else None),
                "prompt": B.build_prompt(row),
                "black_idx": bidx, "unk_idx": uidx, "target_letter": tgt,
                "pred_index": pred, "pred_letter": letter, "pred_class": cls,
                "model_output": gen, "alpha_sum": float(a.sum()),
                "alpha_mean": float(a.mean()), "alpha_final": float(a[-1]),
                "sat_frac": float(sat.mean()), "p_black_final": float(pb[-1]),
                "alpha_traj": [round(v, 4) for v in a.tolist()],
                "pblack_traj": [round(v, 4) for v in pb.tolist()],
            })
            if (idx + 1) % 50 == 0:
                print(f"[{cond}] {idx+1}/{len(rows)} b={counts['black']} "
                      f"nb={counts['nonblack']} ab={counts['abstain']} "
                      f"un={counts['unparseable']} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        steerer.detach()

    n = len(rows)
    rates = {f"{k}_rate": v / n for k, v in counts.items()}
    result = {
        "condition": cond, "gains": {"Kp": kp, "Ki": eff_ki, "Kd": eff_kd},
        "actuator": f"all_{N_LAYERS}_layers", "anti_windup": True,
        "setpoint": SETPOINT, "alpha_max": amax, "vhat_layer": LAYER, "n": n,
        "counts": counts, "rates": rates,
        "d_gap": rates["black_rate"] - rates["nonblack_rate"],
        "mean_total_actuation": float(np.mean(alpha_sums)) if alpha_sums else 0.0,
        "mean_alpha": float(np.mean(alpha_means)) if alpha_means else 0.0,
        "mean_final_alpha": float(np.mean(alpha_finals)) if alpha_finals else 0.0,
        "mean_sat_frac": float(np.mean(sat_fracs)) if sat_fracs else 0.0,
        "gen_length": GEN_LENGTH, "steps": STEPS, "block_length": BLOCK_LENGTH,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_s": time.time() - t0,
    }
    stem = tag if tag else f"dpid_{cond}"
    with open(os.path.join(out_dir, f"cond_{stem}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{stem}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond}] DONE rates={rates} d_gap={result['d_gap']:.3f} "
          f"mean_alpha={result['mean_alpha']:.2f} final_alpha={result['mean_final_alpha']:.2f} "
          f"sat={result['mean_sat_frac']:.2f} -> cond_{stem}.json", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-items", type=int, default=2)
    ap.add_argument("--cond", choices=list(COND_MASK.keys()))
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--amax", type=float, default=ALPHA_MAX)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 400")
    ap.add_argument("--items", default=SWEEP400,
                    help="input jsonl (default sweep400; point at a rotation file)")
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--tag", default=None, help="output stem override: cond_<tag>.json")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.smoke:
        smoke(args.smoke_items, args.kp, args.ki, args.kd, args.amax)
        return
    if not args.cond:
        ap.error("--cond required (or use --selftest / --smoke)")
    run(args.cond, args.kp, args.ki, args.kd, args.amax, args.limit,
        args.out_dir, args.tag, args.items)


if __name__ == "__main__":
    main()
