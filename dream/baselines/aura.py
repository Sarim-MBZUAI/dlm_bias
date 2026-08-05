#!/usr/bin/env python
"""dream/baselines/aura.py -- AURA (AurA, Suau et al., arXiv:2407.12824) on
Dream-v0-Instruct-7B.  Port of baselines/aura.py, hooked at the MLP-HIDDEN units
(the 18944-d gated activation, the finest-grained per-neuron bank in the block).

AurA computes per-neuron the AUROC of that neuron at classifying a concept (here
"Black", label 1), then applies a per-neuron MULTIPLICATIVE gate:

    alpha = 1 where auroc<=0.5, else 1 - 2*(auroc-0.5)   (aura_hook.py:65-67)
    output <- output * alpha                             (per-neuron gate)

AUROC is directions.auroc_per_neuron (exact sklearn roc_auc_score).  On Dream the
richest per-neuron bank is the 18944-wide gated MLP activation, which has NO module
whose OUTPUT it is -- it is the INPUT to down_proj (exactly what
calib.collect_activations('mlp_hidden') captures).  So we GATE it by multiplying
down_proj's INPUT via a forward_PRE_hook (common_dream.mul_vec_pre_hook) -- the
faithful equivalent of AURA's output*alpha for Dream's fused MLP.

TWO CONDITIONS (a multiplicative gate's sign/strength cannot flip its effect):
  (1) vanilla  (SUPPRESSION, labeled NEGATIVE CONTROL, LOWERS Black rate): the
      faithful AURA gate  alpha_supp = 1 - 2*max(auroc-0.5,0)  in [0,1].
  (2) inject   (AMPLIFICATION, the headline number): the mirrored gate
      alpha_amp  = 1 + gamma*2*max(auroc-0.5,0)  >= 1  (gamma>=0; 0=identity).
Both leave non-selective neurons (auroc<=0.5) at exactly 1.0.

CLI:
    python dream/baselines/aura.py --selftest                       # offline
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/aura.py --fit      # NEEDS GPU
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/aura.py --run --mode vanilla
    CUDA_VISIBLE_DEVICES=4 python dream/baselines/aura.py --run --mode inject --gamma 2
"""
import argparse
import os
import sys

import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # common_dream
sys.path.insert(0, _HERE)

import common_dream as C  # noqa: E402
import calib             # noqa: E402
import directions        # noqa: E402

WHERE = "mlp_hidden"                 # AURA hooked at the 18944-d MLP-hidden units
AUROC_CACHE = os.path.join(calib.CACHE_DIR, "aura_auroc.pt")
DEFAULT_OUT = os.path.join(ROOT, "results", "dream", "aura")
MODES = ("vanilla", "inject")


def suppression_gate(auroc):
    """Faithful AURA suppression gate  alpha = 1 - 2*max(auroc-0.5,0)  in [0,1]."""
    auroc = torch.as_tensor(auroc, dtype=torch.float32)
    return 1.0 - 2.0 * (auroc - 0.5).clamp(min=0.0)


def amplification_gate(auroc, gamma):
    """Injection gate  alpha = 1 + gamma*2*max(auroc-0.5,0)  >= 1 (gamma=0=identity)."""
    auroc = torch.as_tensor(auroc, dtype=torch.float32)
    return 1.0 + float(gamma) * 2.0 * (auroc - 0.5).clamp(min=0.0)


def gate_for(auroc, mode, gamma):
    assert mode in MODES, f"mode must be one of {MODES}"
    return suppression_gate(auroc) if mode == "vanilla" else amplification_gate(auroc, gamma)


def fit(model=None, tok=None, cap=calib.CAP, save=True, out_path=AUROC_CACHE):
    """Build the per-neuron AUROC over the MLP-hidden units and cache it.  NEEDS GPU."""
    blob = calib.collect_activations(WHERE, model=model, tok=tok, cap=cap, save=False)
    acts = blob["acts"]; labels = blob["labels"]
    assert acts.shape[-1] == C.D_MLP, f"expected MLP-hidden width {C.D_MLP}, got {acts.shape[-1]}"

    auroc = torch.empty(C.N_LAYERS, C.D_MLP, dtype=torch.float32)
    for k in range(C.N_LAYERS):
        a_k, _gate_k = directions.auroc_per_neuron(acts[:, k, :], labels)
        auroc[k] = a_k
        print(f"[aura:fit] layer {k+1}/{C.N_LAYERS} auroc done", flush=True)

    out = {"auroc": auroc, "where": WHERE, "feat": C.D_MLP, "n_layers": C.N_LAYERS,
           "n_items": blob["n_items"], "source": blob["source"]}
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[aura:fit] SAVED auroc {tuple(auroc.shape)} -> {out_path}", flush=True)
    return auroc


def load_auroc(path=AUROC_CACHE):
    assert os.path.exists(path), \
        f"missing {path}; run `python dream/baselines/aura.py --fit` first (NEEDS GPU)."
    a = torch.load(path, map_location="cpu")["auroc"].to(torch.float32)
    assert tuple(a.shape) == (C.N_LAYERS, C.D_MLP), f"bad auroc shape {tuple(a.shape)}"
    return a


def attach_fn(mode, gamma=1.0, auroc=None):
    """attach(model)->handles gating each layer's down_proj INPUT (18944-d)."""
    a = load_auroc() if auroc is None else torch.as_tensor(auroc, dtype=torch.float32)
    gate = gate_for(a, mode, gamma)          # (28,18944)

    def _attach(model):
        handles = []
        for k in range(C.N_LAYERS):
            hook = C.mul_vec_pre_hook(gate[k])
            handles += C.attach(model, [C.down_proj_path(k)], hook, pre=True)
        return handles

    return _attach


def run(mode, gamma=1.0, out_dir=DEFAULT_OUT, tag=None, items_path=None, limit=0,
        baseline_black_rate=None, model=None, tok=None, auroc=None):
    """Evaluate one AURA condition on the sweep-400 BBQ items.  NEEDS A GPU."""
    assert mode in MODES, f"mode must be one of {MODES}"
    if tag is None:
        tag = mode if mode == "vanilla" else f"inject_g{gamma:g}"
    cfg = {"method": "aura", "where": WHERE, "mode": mode, "gamma": float(gamma),
           "multiplicative": True, "auroc_cache": AUROC_CACHE,
           "hook": "mul_vec_pre_hook on layers[k].mlp.down_proj INPUT (18944-d)",
           "note": "vanilla=suppression(1-2max(auroc-.5,0)); inject=amplify(1+g*2max(auroc-.5,0))"}
    return C.run_items(
        attach_fn=attach_fn(mode, gamma=gamma, auroc=auroc),
        items_path=items_path or C.SWEEP400,
        out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        config_extra=cfg)


def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-aura] {name:52s} : {'PASS' if cond else 'FAIL'}")

    auroc = torch.tensor([0.5, 0.75, 1.0, 0.3, 0.6])
    supp = suppression_gate(auroc)
    check("suppression gate == 1 - 2*max(auroc-.5,0)",
          torch.allclose(supp, 1.0 - 2.0 * (auroc - 0.5).clamp(min=0.0)))
    check("suppression: auroc<=.5 -> gate==1", supp[0] == 1.0 and supp[3] == 1.0)
    check("suppression: auroc==1 -> gate==0", torch.isclose(supp[2], torch.tensor(0.0)))
    check("suppression gate in [0,1]", bool((supp >= 0).all() and (supp <= 1).all()))
    _a, dir_gate = directions.auroc_per_neuron(
        torch.stack([torch.cat([torch.randn(200) + 8, torch.randn(200) - 8])], 1),
        torch.cat([torch.ones(200), torch.zeros(200)]))
    check("suppression gate matches directions.auroc gate formula",
          torch.allclose(suppression_gate(_a), dir_gate))

    gamma = 2.0
    amp = amplification_gate(auroc, gamma)
    check("amplification gate == 1 + g*2*max(auroc-.5,0)",
          torch.allclose(amp, 1.0 + gamma * 2.0 * (auroc - 0.5).clamp(min=0.0)))
    check("amplification: auroc<=.5 -> gate==1", amp[0] == 1.0 and amp[3] == 1.0)
    check("amplification: auroc>.5 -> gate>1", bool((amp[[1, 2, 4]] > 1.0).all()))
    check("amplification gamma=0 -> identity",
          torch.allclose(amplification_gate(auroc, 0.0), torch.ones_like(auroc)))

    check("gate_for vanilla == suppression", torch.allclose(gate_for(auroc, "vanilla", 9), supp))
    check("gate_for inject == amplification", torch.allclose(gate_for(auroc, "inject", gamma), amp))

    N = 800
    labels = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)])
    sep = torch.cat([torch.randn(N // 2) + 8.0, torch.randn(N // 2) - 8.0])
    rnd = torch.randn(N)
    a2, _ = directions.auroc_per_neuron(torch.stack([sep, rnd], dim=1), labels)
    check("separable neuron: vanilla gate near 0", suppression_gate(a2)[0] < 0.02)
    check("separable neuron: inject gate above 1", amplification_gate(a2, 2.0)[0] > 1.5)

    # multiplicative PRE-hook on synthetic input.
    B, S, H = 2, 4, 5
    x = torch.randn(B, S, H); gate5 = torch.rand(H) + 0.1
    C.reset_fire_count()
    r = C.mul_vec_pre_hook(gate5)(None, (x.clone(), "extra"))
    check("mul_vec_pre_hook: input[0]*gate, extras kept",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "extra"
          and torch.allclose(r[0], x * gate5))
    check("mul_vec_pre_hook incremented fire counter", C.get_fire_count() == 1)
    kill = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    r2 = C.mul_vec_pre_hook(kill)(None, (x.clone(),))
    check("suppression zeros the dampened neuron, leaves others",
          torch.allclose(r2[0][..., 0], torch.zeros(B, S))
          and torch.allclose(r2[0][..., 1:], x[..., 1:]))

    fake_auroc = torch.rand(C.N_LAYERS, C.D_MLP)
    check("gate_for on (28,18944) preserves shape",
          tuple(gate_for(fake_auroc, "inject", 1.0).shape) == (C.N_LAYERS, C.D_MLP))

    print(f"[selftest-aura] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fit", action="store_true", help="AUROC over MLP-hidden (NEEDS GPU)")
    ap.add_argument("--run", action="store_true", help="evaluate one condition (NEEDS GPU)")
    ap.add_argument("--mode", choices=MODES, default="inject")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--items", default=None,
                    help="BBQ items jsonl (default: sweep400; e.g. a position-balance rotation file)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline_black_rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit(); return
    if args.run:
        run(args.mode, gamma=args.gamma, out_dir=args.out_dir, tag=args.tag,
            items_path=args.items, limit=args.limit,
            baseline_black_rate=args.baseline_black_rate)
        return
    ap.error("nothing to do: pass --selftest, --fit (GPU), or --run (GPU)")


if __name__ == "__main__":
    main()
