#!/usr/bin/env python
"""Per-item static-alpha control for decode-time PI.

Question: does PI win because it re-measures during denoising, or only because it
gives each item its own dose? This script holds one alpha per item for all steps,
with everything else identical to steering/denoise_pid.py (sampler, vhat, all-32-
block actuator, sensor, probe forward, output schema). Two arms:

  p0    alpha_i = clip(scale * (s* - p0_i), 0, amax), fixed from the unsteered
        probe p0 before any steered forward. scale is chosen so the mean alpha over
        the primary PI prompts equals the matched open-loop constant (3.28).
  hind  alpha_i = the primary PI run's own mean command over steps 1-32 for the
        same prompt (hindsight upper bound on any static per-item dose).

Only the controller is replaced; denoise_pid.py is imported, not modified.

Usage:
    python steering/static_alpha_control.py --arm {p0,hind} --rot R
        --model-path PATH --out-dir DIR [--pi-dir DIR] [--limit N]
"""
import argparse
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("denoise_pid", os.path.join(HERE, "denoise_pid.py"))
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

TOKEN_STEPS = 32  # steps that commit tokens (gen_length 32, one per step)


def key(row):
    return (str(row.get("category")), str(row.get("example_id")), str(row.get("question_index")))


class StaticController:
    """Same interface as D.PID. The first update() sees p0 and fixes alpha."""

    def __init__(self, arm, setpoint, amax, scale=None, table=None):
        self.arm, self.setpoint, self.amax = arm, setpoint, amax
        self.scale, self.table = scale, table
        self.row_key = None

    def reset(self):
        self.alpha = None

    def update(self, p_meas):
        if self.alpha is None:
            if self.arm == "p0":
                raw = self.scale * (self.setpoint - p_meas)
            else:
                raw = self.table[self.row_key]
            self.alpha = D.clamp(raw, self.amax, 0.0)
        e = self.setpoint - p_meas
        return self.alpha, e, 0.0, 0.0, self.alpha >= self.amax


def pi_samples(pi_dir, rot):
    path = os.path.join(pi_dir, f"rot{rot}", "cond_dpid_PI_samples.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("p0", "hind"), required=True)
    ap.add_argument("--rot", type=int, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--pi-dir", default=os.path.join(D.ROOT, "results/balanced/results_balanced"))
    ap.add_argument("--match-alpha", type=float, default=3.28)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    D.MODEL_PATH = args.model_path
    items = os.path.join(D.ROOT, f"results/balanced/_sweep400_rot{args.rot}.jsonl")

    kp, ki = 3.0, 0.1  # primary PI gains; its first command is (kp+ki)*(s*-p0)
    first = [s["alpha_traj"][0] for r in range(3) for s in pi_samples(args.pi_dir, r)]
    scale = args.match_alpha * (kp + ki) / float(np.mean(first))
    table = {key(s): float(np.mean(s["alpha_traj"][:TOKEN_STEPS]))
             for s in pi_samples(args.pi_dir, args.rot)}
    ctrl = StaticController(args.arm, D.SETPOINT, D.ALPHA_MAX, scale=scale, table=table)
    print(f"[static] arm={args.arm} rot={args.rot} scale={scale:.4f} "
          f"hind_mean={np.mean(list(table.values())):.3f} model={D.MODEL_PATH}", flush=True)

    D.PID = lambda *a, **k: ctrl
    orig_generate = D.generate_item

    def generate_item(model, tok, steerer, controller, row, **kw):
        controller.row_key = key(row)
        return orig_generate(model, tok, steerer, controller, row, **kw)

    D.generate_item = generate_item
    D.run("PI", kp, ki, 0.0, D.ALPHA_MAX, args.limit, args.out_dir,
          f"static_{args.arm}", items_path=items)


if __name__ == "__main__":
    main()
