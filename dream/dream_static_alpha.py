#!/usr/bin/env python
"""Per-item static-alpha control for Dream decode-time PI.

Same design as steering/static_alpha_control.py, on dream/denoise_pid.py, which
is imported and not modified. Dream's first forward is the unsteered probe, so
p0 is logged as pblack_traj[0] in the primary PI samples.

  p0    alpha_i = clip(scale * (s* - p0_i), 0, amax), scale solved so the mean
        alpha over the primary PI prompts equals PI's mean command (all steps).
  hind  alpha_i = the primary PI run's own mean command over all 64 steps.

Dream's sampler does not commit exactly one token per step, so both arms use the
full-run mean rather than a token-committing window.

Run: python dream/dream_static_alpha.py --arm p0 --rot 0
         --model-path Dream-v0-Instruct-7B --out-dir OUT_DIR
"""
import argparse
import functools
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("dream_denoise_pid", os.path.join(HERE, "denoise_pid.py"))
DP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DP)


def key(row):
    return (str(row.get("category")), str(row.get("example_id")), str(row.get("question_index")))


class StaticController:
    def __init__(self, arm, setpoint, amax, scale, table):
        self.arm, self.setpoint, self.amax = arm, setpoint, amax
        self.scale, self.table = scale, table
        self.row_key = None

    def reset(self):
        self.alpha = None

    def update(self, p_meas):
        if self.alpha is None:
            raw = self.scale * (self.setpoint - p_meas) if self.arm == "p0" else self.table[self.row_key]
            self.alpha = float(min(self.amax, max(0.0, raw)))
        return self.alpha, self.setpoint - p_meas, 0.0, 0.0, self.alpha >= self.amax


def solve_scale(gaps, target, amax):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        mean = float(np.mean(np.clip(mid * gaps, 0.0, amax)))
        lo, hi = (mid, hi) if mean < target else (lo, mid)
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("p0", "hind"), required=True)
    ap.add_argument("--rot", type=int, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--amax", type=float, default=1.0)
    ap.add_argument("--pi-dir", default=os.path.join(DP.ROOT, "results/dream_balanced/decode_pid"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    load = lambda r: [json.loads(l) for l in open(
        os.path.join(args.pi_dir, f"rot{r}", "cond_dpid_PI_samples.jsonl")) if l.strip()]
    all_rows = [s for r in range(3) for s in load(r)]
    gaps = np.array([DP.SETPOINT - s["pblack_traj"][0] for s in all_rows])
    target = float(np.mean([np.mean(s["alpha_traj"]) for s in all_rows]))
    scale = solve_scale(gaps, target, args.amax)
    table = {key(s): float(np.mean(s["alpha_traj"])) for s in load(args.rot)}
    ctrl = StaticController(args.arm, DP.SETPOINT, args.amax, scale, table)
    print(f"[dream-static] arm={args.arm} rot={args.rot} scale={scale:.3f} target_mean={target:.3f} "
          f"p0_arm_mean={np.mean(np.clip(scale * gaps, 0, args.amax)):.3f} "
          f"hind_mean={np.mean(list(table.values())):.3f} model={args.model_path}", flush=True)

    DP.C.load_model_tok = functools.partial(DP.C.load_model_tok, model_dir=args.model_path)
    DP.PID = lambda *a, **k: ctrl
    orig_build = DP.build_input

    def build_input(tok, row, device):
        ctrl.row_key = key(row)
        return orig_build(tok, row, device)

    DP.build_input = build_input
    items = os.path.join(DP.ROOT, f"results/balanced/_sweep400_rot{args.rot}.jsonl")
    DP._run("PI", 3.0, 0.1, 0.0, args.amax, args.limit, items, args.out_dir,
            f"static_{args.arm}", False, False)


if __name__ == "__main__":
    main()
