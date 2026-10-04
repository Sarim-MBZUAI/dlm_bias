#!/usr/bin/env python
"""dream/pid_steer.py -- layer-space PID (arXiv:2510.04309) + normal-vector baseline,
ported to Dream-v0-Instruct-7B.  OPEN-LOOP, no feedback (prior-work comparison).

CONTROL AXIS = LAYER DEPTH.  Reuses steering/pid_steer.py's pure math VERBATIM
(imported, NOT reimplemented): build_u (Eq.18), unit_rows, GAINS, black_idx_of,
unk_idx_of.  The only Dream-specific work is: 28 blocks (not 32), 3584-d arrows,
and driving generation through common_dream (Dream's native diffusion_generate).

    --mode pid --cond {base,P,PI,PID} --alpha A
        u(k) = Kp*rhat(k) + Ki*sum_{j<k}rhat(j) + Kd*(rhat(k)-rhat(k-1))   (all 28 blocks)
        fixed additive injection alpha*u(k) at every block, every denoising step.
    --mode normal --source-layer 14 --alpha A
        single diff-in-means vhat = unit(r[L]) broadcast to ALL 28 blocks (alpha*vhat).

Open-loop => we PREFER common_dream.run_items(attach_fn=...) directly: it does the
Dream-native generation, BBQ classification, fire-count assertion, and writes the
shared cond_<tag>.json + _samples.jsonl schema.  attach_fn builds one add_vec_hook
per block (common_dream factory -> shared fire counter increments).

MODES
    --selftest                     offline math check (build_u boundary + unit rows).
    --mode {pid,normal} ...        eval, e.g.
        python dream/pid_steer.py --mode normal --source-layer 14 --alpha 4
"""
import argparse
import importlib.util
import os
import sys

import torch

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # ensure `import common_dream` resolves here

import common_dream as C  # noqa: E402


# steering/pid_steer.py shares this file's basename -> load by file path (sys.path[0]
# is dream/, so a plain `import pid_steer` would re-import THIS file).
def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LP = _load("llada_pid_steer", "steering/pid_steer.py")
build_u = _LP.build_u
unit_rows = _LP.unit_rows
GAINS = _LP.GAINS

DEFAULT_ARROWS = os.path.join(ROOT, "dream", "arrows.pt")
RESULTS = os.path.join(ROOT, "results", "dream", "layer_pid")


# --------------------------------------------------------------------------- #
# Arrows -> per-block (28, 3584) injection matrix.
# --------------------------------------------------------------------------- #
def load_arrows(dummy_arrows, arrows_path):
    if dummy_arrows:
        torch.manual_seed(1234)
        return torch.randn(C.N_LAYERS, C.D_MODEL, dtype=torch.float32)
    blob = torch.load(arrows_path, map_location="cpu")
    r = blob["r"].to(torch.float32)
    assert r.shape == (C.N_LAYERS, C.D_MODEL), f"arrows {tuple(r.shape)} != {(C.N_LAYERS, C.D_MODEL)}"
    return r


def build_injection(mode, cond, r, source_layer, alpha):
    """Return (28, 3584) per-block injection, or None for the un-hooked base."""
    if mode == "normal":
        rL = r[source_layer]
        vhat = rL / rL.norm().clamp(min=1e-12)
        return alpha * vhat.unsqueeze(0).expand(C.N_LAYERS, -1).contiguous()
    if cond == "base":
        return None
    kp, ki, kd = GAINS[cond]
    return alpha * build_u(unit_rows(r), kp, ki, kd)


def make_attach_fn(inject):
    """attach_fn(model) -> handles: one common_dream add_vec_hook per block (28)."""
    def attach_fn(model):
        handles = []
        for k in range(C.N_LAYERS):
            handles += C.attach(model, [C.block_path(k)], C.add_vec_hook(inject[k]), pre=False)
        return handles
    return attach_fn


# --------------------------------------------------------------------------- #
# Offline self-test: build_u boundary conditions + unit-norm rows (28-layer dummy).
# --------------------------------------------------------------------------- #
def selftest():
    torch.manual_seed(0)
    N, H = C.N_LAYERS, 8
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-pid] {name:52s} : {'PASS' if cond else 'FAIL'}")

    r = torch.randn(N, H)
    rhat = unit_rows(r)
    check("unit_rows: every row unit-norm",
          torch.allclose(rhat.norm(dim=1), torch.ones(N), atol=1e-6))

    # Reference exclusive cumsum + first difference (rhat(-1):=rhat(0) => diff[0]=0).
    cumexcl = torch.zeros(N, H)
    for k in range(1, N):
        cumexcl[k] = cumexcl[k - 1] + rhat[k - 1]
    diff = torch.zeros(N, H)
    for k in range(1, N):
        diff[k] = rhat[k] - rhat[k - 1]

    for name, (kp, ki, kd) in {"P": GAINS["P"], "PI": GAINS["PI"], "PID": GAINS["PID"]}.items():
        u = build_u(rhat, kp, ki, kd)
        check(f"{name} == Kp*rhat + Ki*cumexcl + Kd*diff",
              torch.allclose(u, kp * rhat + ki * cumexcl + kd * diff, atol=1e-6))

    uPID = build_u(rhat, *GAINS["PID"])
    check("k=0 boundary: derivative zero -> u(0)==Kp*rhat(0)",
          torch.allclose(uPID[0], GAINS["PID"][0] * rhat[0], atol=1e-6))
    check("shape == (28, H)", tuple(uPID.shape) == (N, H))

    # normal-vector injection: all 28 rows identical == alpha*unit(r[L]).
    inj = build_injection("normal", None, r, 14, 2.0)
    vhat = r[14] / r[14].norm()
    check("normal: 28 rows identical == alpha*vhat",
          all(torch.allclose(inj[k], 2.0 * vhat, atol=1e-6) for k in range(N)))

    print(f"[selftest-pid] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mode", choices=["pid", "normal"], default="pid")
    ap.add_argument("--cond", choices=list(GAINS.keys()), help="PID condition (mode=pid)")
    ap.add_argument("--source-layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--arrows", default=DEFAULT_ARROWS)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--items", default=C.SWEEP400)
    ap.add_argument("--steps", type=int, default=C.GEN_DEFAULTS["steps"])
    ap.add_argument("--max-new-tokens", type=int, default=C.GEN_DEFAULTS["max_new_tokens"])
    ap.add_argument("--dummy-arrows", action="store_true",
                    help="random (28,3584) arrows (GPU smoke; arrows.pt not built yet)")
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--tag", default=None, help="output stem override: cond_<tag>.json")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.mode == "pid" and not args.cond:
        ap.error("--cond required for --mode pid (or use --selftest)")

    need_arrows = not (args.mode == "pid" and args.cond == "base")
    r = load_arrows(args.dummy_arrows, args.arrows) if need_arrows else None
    inject = build_injection(args.mode, args.cond, r, args.source_layer, args.alpha)

    if args.mode == "normal":
        cond_label = f"normalL{args.source_layer}"
        kp, ki, kd = 1.0, 0.0, 0.0
    else:
        cond_label = args.cond
        kp, ki, kd = (0.0, 0.0, 0.0) if args.cond == "base" else GAINS[args.cond]

    if args.tag:
        tag = args.tag
    elif args.mode == "pid" and args.cond == "base":
        tag = "base"
    else:
        tag = f"{cond_label}_a{args.alpha:g}".replace(".", "p")

    attach_fn = make_attach_fn(inject) if inject is not None else None
    gen_overrides = {"steps": args.steps, "max_new_tokens": args.max_new_tokens}
    config_extra = {
        "mode": args.mode, "condition": cond_label,
        "source_layer": args.source_layer if args.mode == "normal" else None,
        "alpha": args.alpha, "gains": {"Kp": kp, "Ki": ki, "Kd": kd},
        "dummy_arrows": bool(args.dummy_arrows), "arrows_path": None if not need_arrows else args.arrows,
    }
    print(f"[{cond_label}] mode={args.mode} CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"alpha={args.alpha} Kp={kp} Ki={ki} Kd={kd} dummy={args.dummy_arrows} "
          f"steps={args.steps}", flush=True)

    C.run_items(attach_fn=attach_fn, items_path=args.items, out_dir=args.out_dir,
                tag=tag, limit=args.limit, config_extra=config_extra,
                gen_overrides=gen_overrides)


if __name__ == "__main__":
    main()
