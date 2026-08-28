#!/usr/bin/env python
"""llada_moe/denoise_pid.py -- FAITHFUL decode-space PID controller for
LLaDA-MoE-7B-A1B-Instruct.

Port of steering/denoise_pid.py (LLaDA) / dream/denoise_pid.py to the MoE model.
CONTROL AXIS = the DIFFUSION DENOISING STEP t = 0 .. STEPS-1 (NOT layer depth).
LLaDA-MoE samples with the EXTERNAL LLaDA block-diffusion loop (no native
per-step hooks like Dream's diffusion_generate), so the closed loop is the
LLaDA-style controlled_generate: verbatim commit logic (bbq_eval helpers, this
model's mask id) with the PID update inline per step:

    p_black(t) = P(target letter token) at the answer slot logits[0, plen, :]
    e(t)       = s* - p_black(t)                       (s* = SETPOINT = 0.9)
    alpha(t)   = clamp( Kp*e + Ki*Iacc + Kd*d , 0, amax )   (imported PID law)

ACTUATOR (default) = ALL 16 LLaDAMoEDecoderLayers. The SAME alpha(t)*vhat is added
to every block residual each step (vhat = unit(arrows["r"][8])). Direction from ONE
layer, actuated at all 16 -- mirrors the LLaDA/Dream design so decode-PID is
comparable to the `normal` open-loop baseline (constant alpha vs feedback alpha(t)).

ACTUATOR VARIANTS (round-4 collapse forensics; defaults keep the behavior above):
    --layers "8" | "4-11" | "0,2,8"   restrict injection to a block subset
                                      (e.g. L8-only = closed-loop CAA parity).
    --layer-scale raw                 inject alpha(t)*r[k] with the RAW per-layer
                                      diff-in-means norms ([0.12..10.8]) instead of
                                      the flat unit broadcast.  Rationale: pooled
                                      hidden norms grow 0.37 -> 150 across the 16
                                      blocks, so a constant unit vector is ~135%
                                      of ||h|| at block 0 but 0.3% at block 15;
                                      raw arrows are a flat 7-32% of the local
                                      ||h|| at EVERY block (alpha=1 == exactly the
                                      natural Black-vs-other mean shift).

ALPHA_MAX: default 1.0 -- the CONSERVATIVE Dream-parity starting point (LLaDA's
amax 6 was 97% garbage on Dream; the MoE residual is 2048-d and its activation
scale is unknown a priori).  The round-4 dose job (round4_jobAB) sweeps
--amax {0.5,1,2,4} and the balanced job picks the calibrated value via
$MOE_AMAX; do NOT hardcode a different default without re-running the sweep.

REUSE (imported verbatim from steering/denoise_pid.py -- the PID math is NOT
reimplemented): PID, COND_MASK, SETPOINT, pid_alphas_closedform,
p_black_from_logits, letter_token_ids.  Steering plumbing uses common_lladamoe.

TIMING (identical to steering/denoise_pid.py): one unsteered probe forward on
the initial all-masked sequence seeds p_black, then each step computes
alpha(t) from the latest measurement, runs ONE steered forward that both
commits tokens and yields the next measurement.  Forwards = 1 probe + STEPS.

MODES
    --selftest                     offline PID-math + anti-windup check (no GPU).
    --smoke [--limit N]            GPU smoke; prints alpha(t)/p_black(t) trajectory.
    --cond {base,P,PI,PID} ...     limited/full eval.
GPU RULE: SLURM assigns the GPU; never set CUDA_VISIBLE_DEVICES yourself.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # ensure `import common_lladamoe` resolves here

import common_lladamoe as C  # noqa: E402


# steering/denoise_pid.py shares this file's basename -> load it by absolute path to
# avoid the name clash (sys.path[0] is llada_moe/), exactly like dream/ does.
def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LD = _load("llada_denoise_pid", "steering/denoise_pid.py")
PID = _LD.PID
COND_MASK = _LD.COND_MASK
SETPOINT = _LD.SETPOINT
pid_alphas_closedform = _LD.pid_alphas_closedform
p_black_from_logits = _LD.p_black_from_logits
letter_token_ids = _LD.letter_token_ids

LAYER = 8                           # vhat SOURCE layer (direction only; actuator = all 16)
ALPHA_MAX = 1.0                     # conservative Dream-parity default (see docstring)
GEN_LENGTH = C.GEN_DEFAULTS["gen_length"]     # 32
STEPS = C.GEN_DEFAULTS["steps"]               # 64
BLOCK_LENGTH = C.GEN_DEFAULTS["block_length"]  # 32
TEMPERATURE = C.GEN_DEFAULTS["temperature"]    # 0.0
LETTERS = C.LETTERS
DEFAULT_ARROWS = os.path.join(ROOT, "llada_moe", "arrows.pt")
RESULTS = os.path.join(ROOT, "results", "lladamoe", "decode_pid")


# --------------------------------------------------------------------------- #
# Steerer: add a SETTABLE alpha*V[k] to block k's residual (all pos) for every
# k in `layers` (default all 16; V rows identical in the default unit mode, so
# that is exactly the original all-16 broadcast).  The hooks share ONE mutable
# alpha, so one assignment steers the whole actuated set.
# Uses common_lladamoe's tuple contract (hidden_from_output / output_with_hidden).
# --------------------------------------------------------------------------- #
class AllLayerSteerer:
    def __init__(self, V, layers=None):
        self.V = V                  # (N_LAYERS, H) per-block injection rows
        self.layers = list(range(C.N_LAYERS)) if layers is None else list(layers)
        self.alpha = 0.0
        self.fires = 0
        self.handles = []

    def _mk_hook(self, k):
        def hook(mod, inp, out):
            self.fires += 1
            if self.alpha == 0.0:
                return None
            h = C.hidden_from_output(out)
            steer = (self.alpha * self.V[k]).to(h.dtype).to(h.device)
            return C.output_with_hidden(out, h + steer)
        return hook

    def attach(self, model):
        for k in self.layers:
            self.handles += C.attach(model, [C.block_path(k)], self._mk_hook(k), pre=False)
        assert len(self.handles) == len(self.layers)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------- #
# Controlled generation: external LLaDA sampler + per-step decode-space PID.   #
# Commit logic is verbatim bbq_eval.generate (low_confidence remasking), with  #
# THIS model's mask id.  Same structure as steering/denoise_pid.py:251-304.    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def controlled_generate(model, steerer, controller, prompt, tgt_plain, tgt_space,
                        steer_on, steps=STEPS):
    mask_id = C.MASK_ID
    plen = prompt.shape[1]
    x = torch.full((1, plen + GEN_LENGTH), mask_id, dtype=torch.long, device=model.device)
    x[:, :plen] = prompt.clone()

    assert GEN_LENGTH % BLOCK_LENGTH == 0
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    controller.reset()
    alpha_traj = np.zeros(steps, dtype=np.float64)
    pblack_traj = np.zeros(steps, dtype=np.float64)
    sat_traj = np.zeros(steps, dtype=bool)

    # Seed measurement: one unsteered probe on the initial all-masked sequence.
    steerer.alpha = 0.0
    p_meas = p_black_from_logits(model(x).logits, plen, tgt_plain, tgt_space)

    t = 0
    for nb in range(num_blocks):
        b0 = plen + nb * BLOCK_LENGTH
        b1 = plen + (nb + 1) * BLOCK_LENGTH
        block_mask_index = x[:, b0:b1] == mask_id
        ntt = C.get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            alpha, e, _, _, sat = controller.update(p_meas)
            steerer.alpha = alpha if steer_on else 0.0

            mask_index = x == mask_id
            logits = model(x).logits

            p_meas = p_black_from_logits(logits, plen, tgt_plain, tgt_space)
            alpha_traj[t], pblack_traj[t], sat_traj[t] = steerer.alpha, p_meas, sat

            # --- commit (verbatim bbq_eval.generate low-confidence logic) ---
            x0 = torch.argmax(C.add_gumbel_noise(logits, TEMPERATURE), dim=-1)
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
# Offline self-test: imported PID stateful output == pid_alphas_closedform, clamp
# bounds, integral freeze under saturation.  No GPU, no model.                  #
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

    # Loop-plumbing math (no model): step bookkeeping used by controlled_generate.
    check("GEN_LENGTH % BLOCK_LENGTH == 0", GEN_LENGTH % BLOCK_LENGTH == 0)
    nb = GEN_LENGTH // BLOCK_LENGTH
    check("STEPS % num_blocks == 0", STEPS % nb == 0)
    mi = torch.tensor([[True] * BLOCK_LENGTH])
    ntt = C.get_num_transfer_tokens(mi, STEPS // nb)
    check("get_num_transfer_tokens sums to block width",
          int(ntt.sum()) == BLOCK_LENGTH and ntt.shape == (1, STEPS // nb))

    # --layers / --layer-scale plumbing (offline; dummy arrows are seeded randn).
    check("parse_layers 'all' -> 0..15", parse_layers("all") == list(range(C.N_LAYERS)))
    check("parse_layers '8' -> [8]", parse_layers("8") == [8])
    check("parse_layers '4-6,8' -> [4,5,6,8]", parse_layers("4-6,8") == [4, 5, 6, 8])
    Vu = build_injection_matrix(True, layer_scale="unit")
    torch.manual_seed(1234)
    rr = torch.randn(C.N_LAYERS, C.D_MODEL, dtype=torch.float32)
    check("unit matrix: all rows == unit(r[LAYER])",
          tuple(Vu.shape) == (C.N_LAYERS, C.D_MODEL)
          and torch.allclose(Vu, (rr[LAYER] / rr[LAYER].norm()).expand_as(Vu), atol=1e-6))
    check("unit matrix rows are unit-norm",
          torch.allclose(Vu.norm(dim=1), torch.ones(C.N_LAYERS), atol=1e-5))
    Vr = build_injection_matrix(True, layer_scale="raw")
    check("raw matrix == raw arrows (natural per-layer norms)", torch.allclose(Vr, rr))
    check("raw matrix norms NOT flat (natural profile preserved)",
          float(Vr.norm(dim=1).std()) > 0.0)
    # Steerer subset wiring: hooks only on requested blocks, alpha*V[k] applied.
    st = AllLayerSteerer(Vr, layers=[8])
    st.alpha = 2.0
    out = st._mk_hook(8)(None, None, (torch.zeros(1, 3, C.D_MODEL),))
    check("subset steerer adds alpha*V[8] at every position",
          torch.allclose(out[0], (2.0 * Vr[8]).expand(1, 3, -1)))
    check("steerer default layers == all 16",
          AllLayerSteerer(Vr).layers == list(range(C.N_LAYERS)))

    print(f"[selftest-dpid] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Arrows / items.
# --------------------------------------------------------------------------- #
def parse_layers(spec):
    """'all' -> 0..15; '8' -> [8]; '4-11' -> [4..11]; '0,2,8' -> [0,2,8]."""
    if spec is None or spec == "all":
        return list(range(C.N_LAYERS))
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    out = sorted(set(out))
    assert out and all(0 <= k < C.N_LAYERS for k in out), f"bad --layers spec {spec!r}"
    return out


def build_injection_matrix(dummy_arrows, arrows_path=DEFAULT_ARROWS, layer=LAYER,
                           layer_scale="unit"):
    """(N_LAYERS, H) rows the steerer scales by alpha(t).

    unit: every row = unit(r[layer])  -- the original flat broadcast.
    raw : row k = r[k] RAW (natural per-layer diff-in-means norm profile)."""
    if dummy_arrows:
        torch.manual_seed(1234)
        r = torch.randn(C.N_LAYERS, C.D_MODEL, dtype=torch.float32)
    else:
        r = torch.load(arrows_path, map_location="cpu")["r"].to(torch.float32)
    assert r.shape == (C.N_LAYERS, C.D_MODEL)
    if layer_scale == "raw":
        return r.clone()
    rL = r[layer]
    vhat = rL / rL.norm().clamp(min=1e-12)
    return vhat.unsqueeze(0).expand(C.N_LAYERS, -1).contiguous()


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
def _run(cond, kp, ki, kd, amax, limit, items_path, out_dir, tag, dummy_arrows, smoke,
         layers_spec="all", layer_scale="unit"):
    use_ki, use_kd = COND_MASK[cond]
    eff_ki = ki if use_ki else 0.0
    eff_kd = kd if use_kd else 0.0
    steer_on = cond != "base"
    layers = parse_layers(layers_spec)

    model, tok = C.load_model_tok()
    V = build_injection_matrix(dummy_arrows, layer_scale=layer_scale).to(model.device)
    plain, space = letter_token_ids(tok)
    steerer = AllLayerSteerer(V, layers=layers)
    steerer.attach(model)
    ctrl = PID(kp, eff_ki, eff_kd, SETPOINT, amax, antiwindup=True)
    rows = load_items(limit, items_path)
    print(f"[{cond}] dev={torch.cuda.get_device_name(0)} "
          f"CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} n={len(rows)} "
          f"actuator={len(layers)}of{C.N_LAYERS}({layers_spec}) scale={layer_scale} "
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
            steerer.fires = 0
            x, a, pb, sat = controlled_generate(
                model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=steer_on)
            hook_fires += steerer.fires   # counts EVERY invocation (bump precedes the
                                          # alpha==0 short-circuit): expect
                                          # n*(1+steps)*len(layers), base incl.
            gen = tok.batch_decode(x[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
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
                fwd = steerer.fires // len(layers)
                print(f"\n[smoke] item {idx} ex={row.get('example_id')} target={tgt} "
                      f"parsed='{letter}' cls={cls}", flush=True)
                print(f"[smoke]   forwards={fwd} (expect 1 probe + {STEPS} = {STEPS + 1}) "
                      f"gen={gen!r}", flush=True)
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
        "condition": cond, "model": C.MODEL_NAME, "n": n,
        "counts": counts, "rates": rates,
        # gap_within = within-run aim (black - nonblack). d_gap keeps the run_items /
        # baselines/common.py semantics (black - baseline_black_rate); no baseline is
        # passed here, so it is null -- compare gap_within across conditions instead.
        "gap_within": rates["black_pick_rate"] - rates["nonblack_pick_rate"],
        "d_gap": None, "baseline_black_rate": None,
        "acc_disambig": (dis_correct / dis_n) if dis_n else None, "n_disambig": dis_n,
        "gains": {"Kp": kp, "Ki": eff_ki, "Kd": eff_kd},
        "actuator": (f"all_{C.N_LAYERS}_layers" if len(layers) == C.N_LAYERS
                     else "layers_" + ",".join(str(k) for k in layers)),
        "steer_layers": layers, "layer_scale": layer_scale,
        "anti_windup": True,
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
    ap.add_argument("--layers", default="all",
                    help='actuated blocks: "all" (default), "8", "4-11", "0,2,8"')
    ap.add_argument("--layer-scale", choices=["unit", "raw"], default="unit",
                    help="unit: alpha*unit(r[8]) at every actuated block (default); "
                         "raw: alpha*r[k] with the natural per-layer norms")
    ap.add_argument("--dummy-arrows", action="store_true",
                    help="use random (16,2048) arrows (GPU smoke only; arrows.pt not built yet)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.smoke:
        cond = args.cond or "PI"
        _run(cond, args.kp, args.ki, args.kd, args.amax, args.limit or 2, args.items,
             args.out_dir, args.tag, args.dummy_arrows, smoke=True,
             layers_spec=args.layers, layer_scale=args.layer_scale)
        return
    if not args.cond:
        ap.error("--cond required (or use --selftest / --smoke)")
    _run(args.cond, args.kp, args.ki, args.kd, args.amax, args.limit, args.items,
         args.out_dir, args.tag, args.dummy_arrows, smoke=False,
         layers_spec=args.layers, layer_scale=args.layer_scale)


if __name__ == "__main__":
    main()
