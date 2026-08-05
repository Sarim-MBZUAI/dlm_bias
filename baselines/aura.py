#!/usr/bin/env python
"""baselines/aura.py -- AURA (AurA, Suau et al., arXiv:2407.12824) ported to the
LLaDA-8B-Instruct masked-diffusion BBQ bias harness, hooked at the MLP-HIDDEN
units (max faithfulness: AURA operates per-NEURON, and the 12288-d gated MLP
activation is the finest-grained neuron bank in the block).

WHAT AURA IS (reference: act/hooks/aura_hook.py + act/utils/auroc.py)
--------------------------------------------------------------------
AurA computes, for every neuron, the AUROC of that neuron's activation at
classifying a concept (here concept = "Black" answer, label 1). It then applies
a per-neuron MULTIPLICATIVE dampening gate to the module output:

    aura_hook.py:65-67   alpha = ones_like(auroc)
                         mask  = auroc > 0.5
                         alpha[mask] = 1 - 2*(auroc[mask] - 0.5)
    aura_hook.py:104     output_aura = output * alpha         # per-neuron gate
    aura_hook.py:107     output = (1-strength)*output + strength*output_aura

Neurons that do not separate the classes (auroc <= 0.5) keep gate = 1 (untouched);
strongly concept-selective neurons (auroc -> 1) get gate -> 0 (fully dampened).
AUROC itself is the exact sklearn per-column roc_auc_score of auroc.py:27-31 /
_compute_auroc_chunk, reproduced faithfully in directions.auroc_per_neuron
(directions.py:154-182), whose `gate` output IS the suppression alpha above.

WHERE WE HOOK IT ON LLaDA (max faithfulness)
--------------------------------------------
The AURA paper dampens per-neuron OUTPUTS of a module. On LLaDA the richest
per-neuron bank is the 12288-wide gated MLP hidden activation
    x = act(ff_proj(x)) * up_proj(x)                  # modeling_llada.py:924-929
    x = ff_out(x)                                      # modeling_llada.py:930
There is NO module whose OUTPUT is that gated activation -- it is computed inline
and immediately consumed as the INPUT to ff_out. So, exactly as
calib.collect_activations('mlp_hidden') CAPTURES it (calib.py:177,187 -> inp[0]
of ff_out), we GATE it by multiplying ff_out's INPUT via a forward_PRE_hook
(common.mul_vec_pre_hook). This is the faithful equivalent of AURA's
output * alpha for LLaDA's fused MLP -- same neurons, same multiply -- and is the
same pre-hook rationale already documented in common.py:152-166. (common.mul_vec_hook,
the OUTPUT twin, cannot be used here: ff_out's output is the 4096-d residual, not
the 12288-d neuron bank the 12288-d gate indexes -- the dims would not match.)

TWO CONDITIONS (the sign/strength of a multiplicative gate cannot flip its effect)
----------------------------------------------------------------------------------
A multiplicative gate cannot be turned into an injector by a sign or a "strength":
in the reference, strength only interpolates toward 1
    (1-s)*out + s*(out*alpha) = out*(1 - s*(1-alpha)),
so with the suppression alpha (<= 1) any strength in [0,1] keeps the effective
gate in [alpha, 1] -- it can only dampen. A NEGATIVE gate does not "reverse" the
concept, it just flips activation signs (meaningless). To AMPLIFY the concept you
need a DIFFERENT GATE FORMULA (gate > 1). Hence we ship two gate formulas:

  (1) vanilla  (SUPPRESSION, a labeled NEGATIVE CONTROL -- expected to LOWER the
      Black pick-rate): the faithful AURA gate
          alpha_supp = 1 - 2*max(auroc - 0.5, 0)        in [0, 1]
      This suppresses Black-predictive neurons. It is NOT the injection headline;
      it exists to show AURA's own mechanism runs and moves the metric the
      OPPOSITE way (down).

  (2) inject   (AMPLIFICATION, the headline injection number): the mirrored gate
          alpha_amp  = 1 + gamma * 2*max(auroc - 0.5, 0)   >= 1  (gamma >= 0)
      This AMPLIFIES the same Black-predictive neurons (gamma sweeps the strength;
      gamma=0 is identity). Amplifying the neurons AURA would dampen is the
      natural multiplicative dual of AURA-as-injector.

Both gates leave non-selective neurons (auroc <= 0.5) at exactly 1.0.

FIT / RUN split
---------------
    fit()            -> calib.collect_activations('mlp_hidden')  [GPU]
                        directions.auroc_per_neuron              -> auroc (32,12288)
                        caches cache/aura_auroc.pt               [GPU; DO NOT run here]
    run(mode,gamma)  -> load auroc, build per-layer gate, attach, common.run_baseline [GPU]

CLI:
    python aura.py --selftest                      # offline math, no GPU
    python aura.py --fit                           # NEEDS GPU (do NOT run here)
    python aura.py --run --mode vanilla            # NEEDS GPU
    python aura.py --run --mode inject --gamma 2   # NEEDS GPU
"""
import argparse
import os
import sys

import torch

# Harness import (ROOT = MAIN tree; mirrors steering/pid_steer.py:49-51).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import common  # noqa: E402
import calib   # noqa: E402
import directions  # noqa: E402
from common import N_LAYERS, H_MLP, CACHE_DIR  # noqa: E402

WHERE = "mlp_hidden"                 # AURA hooked at the 12288-d MLP-hidden units
AUROC_CACHE = os.path.join(CACHE_DIR, "aura_auroc.pt")
DEFAULT_OUT = os.path.join(ROOT, "results", "aura")
MODES = ("vanilla", "inject")


# --------------------------------------------------------------------------- #
# Gate formulas (per-neuron). auroc: any (...,) tensor of AUROC values.        #
# --------------------------------------------------------------------------- #
def suppression_gate(auroc):
    """Faithful AURA suppression gate  alpha = 1 - 2*max(auroc-0.5, 0)  in [0,1].

    Reproduces aura_hook.py:65-67 exactly (alpha=1 where auroc<=0.5, else
    1-2*(auroc-0.5)); identical to directions.auroc_per_neuron's `gate`."""
    auroc = torch.as_tensor(auroc, dtype=torch.float32)
    return 1.0 - 2.0 * (auroc - 0.5).clamp(min=0.0)


def amplification_gate(auroc, gamma):
    """Injection (amplification) gate  alpha = 1 + gamma*2*max(auroc-0.5, 0)  >= 1.

    The multiplicative dual of the suppression gate: amplifies exactly the
    Black-predictive neurons AURA would dampen. gamma>=0 sweeps strength;
    gamma=0 -> identity. Non-selective neurons (auroc<=0.5) stay at 1.0."""
    auroc = torch.as_tensor(auroc, dtype=torch.float32)
    return 1.0 + float(gamma) * 2.0 * (auroc - 0.5).clamp(min=0.0)


def gate_for(auroc, mode, gamma):
    """Per-neuron gate for a given condition. auroc: (32,12288)."""
    assert mode in MODES, f"mode must be one of {MODES}"
    if mode == "vanilla":
        return suppression_gate(auroc)
    return amplification_gate(auroc, gamma)


# --------------------------------------------------------------------------- #
# FIT: AUROC over the 12288-d MLP-hidden activations (NEEDS GPU).              #
# --------------------------------------------------------------------------- #
def fit(model=None, tok=None, cap=calib.CAP, save=True, out_path=AUROC_CACHE):
    """Build the per-neuron AUROC over the MLP-hidden units and cache it.

    Steps (all reused, none reimplemented):
      1. calib.collect_activations('mlp_hidden')  -> acts (2n,32,12288), labels
         (2n,) with Black=1 / other=0, from the contamination-safe held-out
         BBQ-Race contrast set (calib.py).
      2. For each of the 32 layers, directions.auroc_per_neuron(acts[:,k,:],
         labels) -> auroc (12288,)  (exact sklearn roc_auc_score, auroc.py:27-31).
      3. Stack -> auroc (32,12288); cache {'auroc', 'where', 'n_items', ...}.

    NEEDS A GPU (step 1 runs the model). --selftest validates the pure math only.
    Returns the auroc tensor (32,12288)."""
    blob = calib.collect_activations(WHERE, model=model, tok=tok, cap=cap, save=False)
    acts = blob["acts"]              # (2n, 32, 12288)
    labels = blob["labels"]          # (2n,)
    assert acts.shape[-1] == H_MLP, f"expected MLP-hidden width {H_MLP}, got {acts.shape[-1]}"

    auroc = torch.empty(N_LAYERS, H_MLP, dtype=torch.float32)
    for k in range(N_LAYERS):
        a_k, _gate_k = directions.auroc_per_neuron(acts[:, k, :], labels)
        auroc[k] = a_k
        print(f"[aura:fit] layer {k+1}/{N_LAYERS} auroc done", flush=True)

    out = {
        "auroc": auroc, "where": WHERE, "feat": H_MLP, "n_layers": N_LAYERS,
        "n_items": blob["n_items"], "source": blob["source"],
    }
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        print(f"[aura:fit] SAVED auroc {tuple(auroc.shape)} -> {out_path}", flush=True)
    return auroc


def load_auroc(path=AUROC_CACHE):
    """Load the cached per-neuron AUROC (32,12288). Run fit() first (GPU)."""
    assert os.path.exists(path), (
        f"missing {path}; run `python aura.py --fit` first (NEEDS GPU).")
    blob = torch.load(path, map_location="cpu")
    a = blob["auroc"].to(torch.float32)
    assert tuple(a.shape) == (N_LAYERS, H_MLP), f"bad auroc shape {tuple(a.shape)}"
    return a


# --------------------------------------------------------------------------- #
# ATTACH: per-layer multiplicative gate on the ff_out INPUT (12288-d).         #
# --------------------------------------------------------------------------- #
def attach_fn(mode, gamma=1.0, auroc=None):
    """Return an attach(model)->handles closure that gates the MLP-hidden units.

    For each of the 32 blocks, multiplies ff_out's INPUT (the 12288-d gated MLP
    activation) by that layer's per-neuron gate via common.mul_vec_pre_hook
    (forward_pre_hook, pre=True). The gate is the suppression or amplification
    formula for `mode` (see module docstring). All positions, bidirectional,
    fires once per denoising step (~steps*num_blocks per item)."""
    a = load_auroc() if auroc is None else torch.as_tensor(auroc, dtype=torch.float32)
    gate = gate_for(a, mode, gamma)          # (32,12288)

    def _attach(model):
        handles = []
        for k in range(N_LAYERS):
            path = common.submodule_paths("ff_out", [k])   # 1-element list
            hook = common.mul_vec_pre_hook(gate[k])         # h <- h * gate[k]
            handles += common.attach(model, path, hook, pre=True)
        return handles

    return _attach


# --------------------------------------------------------------------------- #
# RUN: one full BBQ condition end-to-end (NEEDS GPU).                          #
# --------------------------------------------------------------------------- #
def run(mode, gamma=1.0, out_dir=DEFAULT_OUT, tag=None, limit=0,
        baseline_black_rate=None, model=None, tok=None, auroc=None,
        items_path=common.SWEEP400):
    """Evaluate one AURA condition on the sweep-400 BBQ items.

    mode='vanilla' -> suppression gate (negative control, lowers Black rate).
    mode='inject'  -> amplification gate (gamma sweeps strength; headline number).
    Writes out_dir/cond_<tag>.json (+ _samples.jsonl). NEEDS A GPU."""
    assert mode in MODES, f"mode must be one of {MODES}"
    if tag is None:
        tag = mode if mode == "vanilla" else f"inject_g{gamma:g}"
    af = attach_fn(mode, gamma=gamma, auroc=auroc)
    cfg = {"method": "aura", "where": WHERE, "mode": mode, "gamma": float(gamma),
           "multiplicative": True, "auroc_cache": AUROC_CACHE,
           "hook": "mul_vec_pre_hook on blocks[k].ff_out INPUT (12288-d MLP-hidden)",
           "note": "vanilla=suppression(1-2max(auroc-.5,0)); inject=amplify(1+g*2max(auroc-.5,0))"}
    return common.run_baseline(
        af, items_path=items_path, out_dir=out_dir, tag=tag, limit=limit,
        baseline_black_rate=baseline_black_rate, model=model, tok=tok,
        config_extra=cfg)


# --------------------------------------------------------------------------- #
# Offline self-test: gate formulas + multiplicative pre-hook on synthetic.     #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-aura] {name:52s} : {'PASS' if cond else 'FAIL'}")

    # --- gate formula exactness --------------------------------------------- #
    auroc = torch.tensor([0.5, 0.75, 1.0, 0.3, 0.6])
    supp = suppression_gate(auroc)
    man_supp = 1.0 - 2.0 * (auroc - 0.5).clamp(min=0.0)
    check("suppression gate == 1 - 2*max(auroc-.5,0)", torch.allclose(supp, man_supp))
    check("suppression: auroc<=.5 -> gate==1", supp[0] == 1.0 and supp[3] == 1.0)
    check("suppression: auroc==1 -> gate==0", torch.isclose(supp[2], torch.tensor(0.0)))
    check("suppression gate in [0,1]", bool((supp >= 0).all() and (supp <= 1).all()))
    # matches directions (the faithful AURA gate)
    _a, dir_gate = directions.auroc_per_neuron(
        torch.stack([torch.cat([torch.randn(200) + 8, torch.randn(200) - 8])], 1),
        torch.cat([torch.ones(200), torch.zeros(200)]))
    check("suppression gate matches directions.auroc gate formula",
          torch.allclose(suppression_gate(_a), dir_gate))

    gamma = 2.0
    amp = amplification_gate(auroc, gamma)
    man_amp = 1.0 + gamma * 2.0 * (auroc - 0.5).clamp(min=0.0)
    check("amplification gate == 1 + g*2*max(auroc-.5,0)", torch.allclose(amp, man_amp))
    check("amplification: auroc<=.5 -> gate==1", amp[0] == 1.0 and amp[3] == 1.0)
    check("amplification: auroc>.5 -> gate>1", bool((amp[[1, 2, 4]] > 1.0).all()))
    check("amplification gamma=0 -> identity (all 1)",
          torch.allclose(amplification_gate(auroc, 0.0), torch.ones_like(auroc)))

    # sign/strength cannot flip a multiplicative gate: interpolating the
    # suppression gate toward 1 by any strength s in [0,1] stays in [alpha,1].
    for s in (0.0, 0.3, 1.0):
        eff = 1.0 - s * (1.0 - supp)            # (1-s)*1 + s*supp
        check(f"strength s={s}: effective suppression gate stays in [alpha,1]",
              bool((eff >= supp - 1e-6).all() and (eff <= 1 + 1e-6).all()))

    # --- gate_for dispatch --------------------------------------------------- #
    check("gate_for vanilla == suppression", torch.allclose(gate_for(auroc, "vanilla", 9), supp))
    check("gate_for inject  == amplification", torch.allclose(gate_for(auroc, "inject", gamma), amp))

    # --- separable neuron end-to-end (auroc from directions -> both gates) --- #
    N = 800
    labels = torch.cat([torch.ones(N // 2), torch.zeros(N // 2)])
    sep = torch.cat([torch.randn(N // 2) + 8.0, torch.randn(N // 2) - 8.0])  # Black-selective
    rnd = torch.randn(N)                                                    # uninformative
    acts = torch.stack([sep, rnd], dim=1)
    a2, _ = directions.auroc_per_neuron(acts, labels)
    gv = suppression_gate(a2)
    gi = amplification_gate(a2, gamma=2.0)
    check("separable neuron: vanilla gate near 0", gv[0] < 0.02)
    check("separable neuron: inject gate above 1", gi[0] > 1.5)
    # uninformative neuron: auroc~0.5 so both gates ~1 (tolerance scales with the
    # gate's slope: 2 for suppression, 2*gamma for amplification).
    dev = abs(float(a2[1]) - 0.5)
    check("uninformative neuron: auroc ~0.5", dev < 0.06)
    check("uninformative neuron: both gates ~1",
          abs(float(gv[1]) - 1.0) <= 2.0 * dev + 1e-6
          and abs(float(gi[1]) - 1.0) <= 2.0 * 2.0 * dev + 1e-6)

    # --- multiplicative PRE-hook on synthetic (B,seq,H) input --------------- #
    B, S, H = 2, 4, 5
    x = torch.randn(B, S, H)
    gate5 = torch.rand(H) + 0.1
    common.reset_fire_count()
    r = common.mul_vec_pre_hook(gate5)(None, (x.clone(), "extra"))
    check("mul_vec_pre_hook: input[0] * gate, extras kept",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "extra"
          and torch.allclose(r[0], x * gate5))
    check("mul_vec_pre_hook incremented fire counter", common.get_fire_count() == 1)
    # applying a near-0 suppression gate on a selective neuron kills it
    kill = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
    r2 = common.mul_vec_pre_hook(kill)(None, (x.clone(),))
    check("suppression zeros the dampened neuron, leaves others",
          torch.allclose(r2[0][..., 0], torch.zeros(B, S))
          and torch.allclose(r2[0][..., 1:], x[..., 1:]))
    # amplification gate >1 grows the neuron
    grow = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0])
    r3 = common.mul_vec_pre_hook(grow)(None, (x.clone(),))
    check("amplification scales the target neuron by gate",
          torch.allclose(r3[0][..., 0], 3.0 * x[..., 0]))

    # --- shapes -------------------------------------------------------------- #
    fake_auroc = torch.rand(N_LAYERS, H_MLP)
    check("gate_for on (32,12288) preserves shape",
          tuple(gate_for(fake_auroc, "inject", 1.0).shape) == (N_LAYERS, H_MLP))

    print(f"[selftest-aura] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                    help="offline gate-formula + multiplicative-hook check (no GPU)")
    ap.add_argument("--fit", action="store_true",
                    help="collect MLP-hidden acts + AUROC, cache aura_auroc.pt (NEEDS GPU)")
    ap.add_argument("--run", action="store_true",
                    help="evaluate one condition on sweep-400 (NEEDS GPU)")
    ap.add_argument("--mode", choices=MODES, default="inject",
                    help="vanilla=suppression (neg. control); inject=amplification (headline)")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="amplification strength for --mode inject (>=0; 0=identity)")
    ap.add_argument("--items", default=common.SWEEP400,
                    help="BBQ items jsonl (e.g. a position-balance rotation file)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baseline-black-rate", type=float, default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.fit:
        fit()
        return
    if args.run:
        run(args.mode, gamma=args.gamma, out_dir=args.out_dir, tag=args.tag,
            limit=args.limit, baseline_black_rate=args.baseline_black_rate,
            items_path=args.items)
        return
    ap.error("nothing to do: pass --selftest (offline), --fit (GPU), or --run (GPU)")


if __name__ == "__main__":
    main()
