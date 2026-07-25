#!/usr/bin/env python
"""PID-Steering (arXiv:2510.04309) ported to LLaDA-8B-Instruct, control axis = LAYER DEPTH.

Paper Eq. 18. For each transformer block k = 0..31, with per-layer diff-in-means
arrow r(k) and unit direction rhat(k) = r(k)/||r(k)||:

    u(k) = Kp * rhat(k)
         + Ki * sum_{j=0}^{k-1} rhat(j)          # integral over shallower layers
         + Kd * ( rhat(k) - rhat(k-1) )          # derivative,  rhat(-1) := 0

The residual stream is steered at EVERY block k, at EVERY denoising step:

    hidden  <-  hidden + alpha * u(k)

alpha is the single intervention-strength knob. Conditions differ ONLY in gains:
    base = no hook           P = (Ki=0,  Kd=0)
    PI   = (Ki=0.05, Kd=0)   PID = (Ki=0.05, Kd=0.02)     [Kp=1.0 throughout]

This is ALL 32 layers -- no single-layer anchor, no noise term, no +0.1 offset
(those were the reference repo's steady-state-error demo hacks, NOT the method).

Porting decisions for the masked-diffusion LM: (a) inject at all 32 blocks on every
denoising step (the hook fires once per model forward, and generate() calls the
model steps*num_blocks times); (b) unit-normalize each layer's arrow independently;
(c) arrows are built by answer-text-span pooling (see build_arrows.py).

NORMAL-STEERING-VECTOR BASELINE (--mode normal)
    The classic single diff-in-means direction: take the layer-`source_layer` arrow
    r(L) (default L=14), unit-normalize it once -> vhat = r(L)/||r(L)||, and inject the
    SAME fixed vector alpha*vhat at ALL 32 blocks, every denoising step. No per-layer
    directions, no I/D terms -- a plain proportional push of a single global vector.
    Contrast with PID (--mode pid) which uses a DIFFERENT per-layer rhat(k) at each block.

MODES
    --selftest            offline math check (PID Eq.18 + normal-vector props); prints PASS/FAIL.
    --mode pid  --cond {base,P,PI,PID} --alpha A       PID layer-depth eval.
    --mode normal --source-layer 14 --alpha A          normal single-vector all-layer baseline.
      [--limit N] [--dummy-arrows] [--tag-prefix STR] [--out-dir DIR]
Reuses bbq_eval.generate / build_prompt / parse_letter verbatim.
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
import bbq_eval  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "data", "bbq_items", "_sweep400.jsonl")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARROWS = os.path.join(ROOT, "steering", "arrows.pt")
RESULTS = os.path.join(ROOT, "results", "layer_pid")

N_LAYERS = 32
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}

# Paper reference gains. Conditions select which terms are active.
GAINS = {
    "base": None,                       # no hook
    "P":   (1.0, 0.0,  0.0),
    "PI":  (1.0, 0.05, 0.0),
    "PID": (1.0, 0.05, 0.02),
}


# --------------------------------------------------------------------------- #
# Core PID combine (Eq. 18).  rhat: (N,H) ALREADY unit-normed per layer.
# --------------------------------------------------------------------------- #
def build_u(rhat, kp, ki, kd):
    """Return u: (N,H), u(k)=Kp*rhat(k)+Ki*sum_{j<k}rhat(j)+Kd*(rhat(k)-rhat(k-1))."""
    n = rhat.shape[0]
    cum = torch.zeros_like(rhat[0])        # sum_{j<k} rhat(j)
    U = []
    for k in range(n):
        p = kp * rhat[k]
        integ = ki * cum.clone()
        # Reference-code boundary (llama_many_layers.py: shifted_ref_dir[0]=ref_dir[0]):
        # rhat(-1) := rhat(0) so the derivative term is ZERO at k=0.
        deriv = kd * (rhat[k] - (rhat[k - 1] if k > 0 else rhat[k]))
        U.append(p + integ + deriv)
        cum = cum + rhat[k]
    return torch.stack(U, 0)


def unit_rows(r, eps=1e-12):
    """Unit-normalize each row (layer) of r: (N,H)."""
    return torch.stack([r[k] / r[k].norm().clamp(min=eps) for k in range(r.shape[0])], 0)


def build_injection(r, kp, ki, kd, alpha):
    """Full per-block injection vectors alpha*u(k) from RAW arrows r: (N,H)."""
    return alpha * build_u(unit_rows(r), kp, ki, kd)


def build_normal_injection(r, source_layer, alpha, eps=1e-12):
    """Normal single-vector baseline: vhat = unit(r[source_layer]), broadcast to ALL
    N_LAYERS blocks as the SAME fixed injection alpha*vhat (per-block index k is ignored).
    Returns (N_LAYERS, H): every row is identical and equals alpha*vhat."""
    rL = r[source_layer]
    vhat = rL / rL.norm().clamp(min=eps)
    return alpha * vhat.unsqueeze(0).expand(N_LAYERS, -1).contiguous()


# --------------------------------------------------------------------------- #
# Hook: add a fixed per-block vector to the residual stream (all positions,
# every denoising step). Fires once per model forward; counts invocations so the
# smoke test can verify all 32 blocks are steered.
# --------------------------------------------------------------------------- #
class AddVec:
    def __init__(self, li, vec, fired):
        self.li = li
        self.vec = vec
        self.fired = fired      # shared dict {li: count}
        self.h = None

    def _hook(self, mod, inp, out):
        hidden = bbq_eval.hidden_from_output(out)
        self.fired[self.li] = self.fired.get(self.li, 0) + 1
        steer = self.vec.to(hidden.dtype).to(hidden.device)
        return bbq_eval.output_with_hidden(out, hidden + steer)

    def attach(self, module):
        self.h = module.register_forward_hook(self._hook)

    def detach(self):
        if self.h is not None:
            self.h.remove()
            self.h = None


def black_idx_of(row):
    for k in range(3):
        p = row["answer_info"].get(f"ans{k}")
        if p and str(p[-1]).strip().lower() in BLACK_TAGS:
            return k
    return None


def unk_idx_of(row):
    for k in range(3):
        p = row["answer_info"].get(f"ans{k}")
        if p and str(p[-1]).strip().lower() == "unknown":
            return k
    return None


# --------------------------------------------------------------------------- #
# Offline PID-math self-test (no GPU, no model).
# --------------------------------------------------------------------------- #
def selftest():
    torch.manual_seed(0)
    N, H = 7, 5
    rhat = torch.randn(N, H)  # synthetic per-layer unit dirs (values arbitrary; math only)

    # Reference: explicit exclusive cumsum + first difference.
    cumexcl = torch.zeros(N, H)
    for k in range(1, N):
        cumexcl[k] = cumexcl[k - 1] + rhat[k - 1]
    diff = torch.zeros(N, H)  # diff[0]=0 (rhat(-1):=rhat(0), matches reference code)
    for k in range(1, N):
        diff[k] = rhat[k] - rhat[k - 1]

    ok_all = True
    for name, (kp, ki, kd) in {"P": GAINS["P"], "PI": GAINS["PI"], "PID": GAINS["PID"]}.items():
        u = build_u(rhat, kp, ki, kd)
        expect = kp * rhat + ki * cumexcl + kd * diff
        p_ok = torch.allclose(u, expect, atol=1e-6)
        ok_all &= p_ok
        print(f"[selftest] {name:3s} (Kp={kp},Ki={ki},Kd={kd})  full formula: "
              f"{'PASS' if p_ok else 'FAIL'}")

    # Term isolation: P<-base, I<-(PI-P), D<-(PID-PI).
    uP = build_u(rhat, *GAINS["P"])
    uPI = build_u(rhat, *GAINS["PI"])
    uPID = build_u(rhat, *GAINS["PID"])
    p_term = torch.allclose(uP, GAINS["P"][0] * rhat, atol=1e-6)
    i_term = torch.allclose(uPI - uP, GAINS["PI"][1] * cumexcl, atol=1e-6)
    d_term = torch.allclose(uPID - uPI, GAINS["PID"][2] * diff, atol=1e-6)
    print(f"[selftest] P term  == Kp*rhat            : {'PASS' if p_term else 'FAIL'}")
    print(f"[selftest] I term  == Ki*cumsum_excl     : {'PASS' if i_term else 'FAIL'}")
    print(f"[selftest] D term  == Kd*(rhat_k-rhat_k-1): {'PASS' if d_term else 'FAIL'}")

    # Reference boundary: deriv(0)=0 and integral empty -> u(0)==Kp*rhat(0).
    b_ok = torch.allclose(uPID[0], GAINS["PID"][0] * rhat[0], atol=1e-6)
    print(f"[selftest] k=0 boundary  u(0)==Kp*rhat(0)      : {'PASS' if b_ok else 'FAIL'}")

    ok_all &= p_term and i_term and d_term and b_ok
    print(f"[selftest] OVERALL: {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def selftest_normal(source_layer=14, alpha=2.0):
    """Offline check of the normal single-vector baseline (no GPU, no model)."""
    torch.manual_seed(0)
    r = torch.randn(N_LAYERS, 4096)                      # synthetic raw arrows
    vhat = r[source_layer] / r[source_layer].norm()
    inj = build_normal_injection(r, source_layer, alpha)

    shape_ok = tuple(inj.shape) == (N_LAYERS, 4096)
    norm_ok = abs(vhat.norm().item() - 1.0) < 1e-6
    ident_ok = all(torch.allclose(inj[k], inj[0], atol=1e-6) for k in range(N_LAYERS))
    eq_ok = torch.allclose(inj[0], alpha * vhat, atol=1e-6)
    # sanity: it is genuinely a single fixed vector, NOT the per-layer PID/P injection
    p_inj = build_injection(r, *GAINS["P"], alpha)        # per-layer alpha*rhat(k)
    distinct_ok = not torch.allclose(inj, p_inj, atol=1e-3)

    print(f"[selftest-normal] L={source_layer} alpha={alpha:g}")
    print(f"[selftest-normal] shape == (32,4096)             : {'PASS' if shape_ok else 'FAIL'}")
    print(f"[selftest-normal] ||vhat|| == 1                   : {'PASS' if norm_ok else 'FAIL'}")
    print(f"[selftest-normal] all 32 layers identical         : {'PASS' if ident_ok else 'FAIL'}")
    print(f"[selftest-normal] each layer == alpha*vhat         : {'PASS' if eq_ok else 'FAIL'}")
    print(f"[selftest-normal] distinct from per-layer P inject : {'PASS' if distinct_ok else 'FAIL'}")
    ok = shape_ok and norm_ok and ident_ok and eq_ok and distinct_ok
    print(f"[selftest-normal] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Eval runner.
# --------------------------------------------------------------------------- #
def run(mode, cond, alpha, arrows_path, source_layer, limit, gen_len, steps, blk,
        dummy_arrows, out_dir, tag_prefix=None, items_path=SWEEP400):
    os.makedirs(out_dir, exist_ok=True)

    # arrows -> per-block injection vectors (base needs none; dummy handled below).
    need_arrows = not (mode == "pid" and cond == "base")
    if not need_arrows:
        r, arrows_meta = None, {"n_items": None, "source": None, "per_layer_raw_norm": None}
    elif dummy_arrows:
        torch.manual_seed(1234)
        r = torch.randn(N_LAYERS, 4096, dtype=torch.float32)
        arrows_meta = {"n_items": -1, "source": "DUMMY_RANDOM_SMOKE", "per_layer_raw_norm": None}
    else:
        blob = torch.load(arrows_path, map_location="cpu")
        r = blob["r"].to(torch.float32)
        arrows_meta = {"n_items": blob.get("n_items"), "source": blob.get("source"),
                       "per_layer_raw_norm": blob.get("per_layer_raw_norm")}

    if mode == "normal":
        # Single fixed diff-in-means vector at source_layer, same at all 32 blocks.
        cond_label = f"normalL{source_layer}"
        kp, ki, kd = 1.0, 0.0, 0.0   # proportional-only push of ONE fixed unit vector
        inject = build_normal_injection(r, source_layer, alpha)
    else:
        cond_label = cond
        kp = ki = kd = 0.0
        if cond != "base":
            kp, ki, kd = GAINS[cond]
        inject = None if cond == "base" else build_injection(r, kp, ki, kd, alpha)

    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    use_rows = rows[:limit] if limit else rows
    print(f"[{cond_label}] items={items_path}", flush=True)

    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to("cuda").eval())
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[{cond_label}] mode={mode} model on {model.device} | CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')} | alpha={alpha} "
          f"Kp={kp} Ki={ki} Kd={kd} | n={len(use_rows)} dummy={dummy_arrows}", flush=True)

    fired = {}
    steerers = []
    if inject is not None:
        blocks = bbq_eval.resolve_module(model, bbq_eval.BLOCKS_PATH)
        assert len(blocks) == N_LAYERS, f"expected {N_LAYERS} blocks, got {len(blocks)}"
        for li in range(N_LAYERS):
            s = AddVec(li, inject[li], fired)
            s.attach(blocks[li])
            steerers.append(s)

    per_item = []
    counts = {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    dis_n = dis_correct = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(use_rows):
            prompt = bbq_eval.build_prompt(row)
            ptxt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           add_generation_prompt=True, tokenize=False)
            ids = torch.tensor(tok(ptxt)["input_ids"], device=model.device).unsqueeze(0)
            out = bbq_eval.generate(model, ids, steps=steps, gen_length=gen_len,
                                    block_length=blk, temperature=0.0, cfg_scale=0.0,
                                    remasking="low_confidence")
            gen = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = bbq_eval.parse_letter(gen, row)
            pred = bbq_eval.LETTERS.index(letter) if letter else None
            bidx = black_idx_of(row)
            uidx = unk_idx_of(row)
            if pred is None:
                cls = "unparseable"; counts["unparseable"] += 1
            elif pred == bidx:
                cls = "black"; counts["black"] += 1
            elif pred == uidx:
                cls = "abstain"; counts["abstain"] += 1
            else:
                cls = "nonblack"; counts["nonblack"] += 1
            if row.get("context_condition") == "disambig" and pred is not None:
                dis_n += 1
                dis_correct += int(pred == int(row["label"]))
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
                "prompt": prompt,
                "black_idx": bidx, "unk_idx": uidx, "pred_index": pred,
                "pred_letter": letter, "model_output": gen, "pred_class": cls,
            })
            if (idx + 1) % 50 == 0:
                print(f"[{cond_label}] {idx+1}/{len(use_rows)} b={counts['black']} "
                      f"nb={counts['nonblack']} ab={counts['abstain']} "
                      f"un={counts['unparseable']} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for s in steerers:
            s.detach()

    n = len(use_rows)
    rates = {
        "black_pick_rate": counts["black"] / n,
        "nonblack_pick_rate": counts["nonblack"] / n,
        "abstain_rate": counts["abstain"] / n,
        "unparseable_rate": counts["unparseable"] / n,
    }
    acc_disambig = (dis_correct / dis_n) if dis_n else None
    n_fired = len(fired)
    fire_counts = sorted(set(fired.values())) if fired else []
    result = {
        "condition": cond_label, "mode": mode,
        "source_layer": source_layer if mode == "normal" else None,
        "alpha": alpha, "n": n, "counts": counts, "rates": rates,
        "acc_disambig": acc_disambig, "n_disambig": dis_n,
        "gains": {"Kp": kp, "Ki": ki, "Kd": kd},
        "gen_length": gen_len, "steps": steps, "block_length": blk,
        "temperature": 0.0, "dummy_arrows": bool(dummy_arrows),
        "hooks_fired_layers": n_fired, "hook_fire_counts": fire_counts,
        "arrows": arrows_meta, "elapsed_s": time.time() - t0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if mode == "pid" and cond == "base":
        tag = "base"
    else:
        prefix = tag_prefix or cond_label
        tag = f"{prefix}_a{alpha:g}".replace(".", "p")
    outp = os.path.join(out_dir, f"cond_{tag}.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{tag}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{cond_label}] DONE rates={rates} acc_dis={acc_disambig} "
          f"hooks_fired_layers={n_fired}/{N_LAYERS} fire_counts={fire_counts} -> {outp}",
          flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="offline math check (PID Eq.18 + normal-vector props); no GPU")
    ap.add_argument("--mode", choices=["pid", "normal"], default="pid")
    ap.add_argument("--cond", choices=list(GAINS.keys()),
                    help="PID condition (mode=pid only)")
    ap.add_argument("--source-layer", type=int, default=14,
                    help="layer whose arrow becomes the fixed normal vector (mode=normal)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--arrows", default=DEFAULT_ARROWS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 400")
    ap.add_argument("--gen-length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--dummy-arrows", action="store_true",
                    help="use a random (32,4096) arrow set (GPU smoke test only)")
    ap.add_argument("--tag-prefix", default=None,
                    help="override output filename prefix (default = condition label)")
    ap.add_argument("--items", default=SWEEP400,
                    help="input jsonl (default sweep400; point at a rotation file)")
    ap.add_argument("--out-dir", default=RESULTS)
    args = ap.parse_args()

    if args.selftest:
        ok_pid = selftest()
        print()
        ok_norm = selftest_normal()
        sys.exit(0 if (ok_pid and ok_norm) else 1)

    if args.mode == "pid" and not args.cond:
        ap.error("--cond required for --mode pid (or use --selftest)")
    run(args.mode, args.cond, args.alpha, args.arrows, args.source_layer, args.limit,
        args.gen_length, args.steps, args.block_length, args.dummy_arrows, args.out_dir,
        args.tag_prefix, args.items)


if __name__ == "__main__":
    main()
