#!/usr/bin/env python
"""Fig-3-on-the-denoising-axis diagnostic for LLaDA-8B-Instruct.

PID-Steering's Fig. 3 sweeps the control axis and asks whether a proportional
controller leaves a nonzero steady-state error (justifying an integral term).
Here we transplant that test onto the DENOISING-STEP axis of the diffusion LM.

At the L14 block hook (which fires ONCE per denoising-step forward pass) we
record, BEFORE any steering correction is applied, the incoming projection of
the hidden state onto the unit steering direction, averaged over the GENERATED
answer-region positions (the appended gen_length block = the last gen_length
sequence positions). That a_t is the state the closed-loop controller sees each
step. We run 4 conditions on the SAME 16 items with ONE model load:

  1. clean       (add, alpha=0)            -> natural a_t trajectory
  2. open-loop   (add, raw vector alpha=8) -> drift/growth of a_t
  3. clamp       (P,  cstar=60)            -> residual error cstar - a_t
  4. cmom        (PI, cstar=60, beta=0.8)  -> residual error cstar - a_t

Verdict logic: if clamp's residual error cstar - a_t decays to ~0 over steps,
the model naturally holds the setpoint (no persistent per-step disturbance ->
integral action useless, closed-loop framing unjustified on this axis). If it
plateaus at a nonzero value, a persistent disturbance exists and integral
action is justified.

Outputs (written next to this script):
  fig3_denoise_error.json  -- per-step mean/std a_t + derived error series
  fig3_denoise_error.png   -- projection trajectory + residual-error panels
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the real sampler / prompt / model-load machinery from the BBQ eval.
_EVAL_DIR = "/home/lukas/users/shashmi/dlm_bias/eval"
sys.path.insert(0, _EVAL_DIR)
import bbq_eval as B  # noqa: E402

# ----- Fixed assets (absolute paths into the MAIN repo; gitignored in worktree) -----
MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DIRECTION_PATH = "/home/lukas/users/shashmi/dlm_bias/directional_steering/race_black_anchored_text.pt"
EVAL_ITEMS = "/home/lukas/users/shashmi/dlm_bias/experiments/data/black_referent_ambig_eval.jsonl"
LAYER = 14
DEVICE = "cuda"

# Sampler defaults (match bbq_eval): 1 block -> exactly STEPS forward passes/item.
GEN_LENGTH = 32
STEPS = 64
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
REMASKING = "low_confidence"

N_ITEMS = 16
CSTAR = 60.0
BETA = 0.8
OPEN_LOOP_ALPHA = 8.0  # raw-vector open-loop strength


class LoggingSteerer(B.BiasSteerer):
    """BiasSteerer that logs the pre-correction gen-region mean projection a_t
    at every hook firing (= every denoising step), then applies the normal
    steering behavior for its mode."""

    def __init__(self, *args, gen_length=GEN_LENGTH, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_length = gen_length
        self.trajectory = []   # list[per-item list[a_t per step]]
        self._cur = None

    def start_item(self):
        """Begin a new item: fresh per-item log + reset cmom integrator."""
        self._cur = []
        self.trajectory.append(self._cur)
        self.reset()

    def _hook(self, module, inputs, output):
        hidden = B.hidden_from_output(output)
        # Gen answer-region = the appended gen_length block = last gen_length
        # positions (prompt-length-agnostic; matches setpoint semantics).
        gen = hidden[:, -self.gen_length:, :].float()
        vh = self.vhat.float().to(hidden.device)
        a_t = float((gen * vh).sum(-1).mean().item())
        if self._cur is not None:
            self._cur.append(a_t)
        # Now apply the real steering correction for this mode (unchanged).
        return super()._hook(module, inputs, output)


def load_items(n):
    rows = []
    with open(EVAL_ITEMS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


@torch.no_grad()
def run_condition(model, tok, rows, direction, mode, alpha, cstar, beta):
    steerer = LoggingSteerer(direction, alpha=alpha, mode=mode, cstar=cstar, beta=beta)
    block = B.resolve_module(model, B.BLOCKS_PATH)[LAYER]
    steerer.attach(block)
    try:
        for row in rows:
            steerer.start_item()
            base_prompt = B.build_prompt(row)
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": base_prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            input_ids = torch.tensor(tok(prompt_text)["input_ids"], device=DEVICE).unsqueeze(0)
            B.generate(
                model, input_ids, steps=STEPS, gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH, temperature=TEMPERATURE,
                cfg_scale=0.0, remasking=REMASKING,
            )
    finally:
        steerer.detach()
    return steerer.trajectory


def summarize(trajectory):
    arr = np.array(trajectory, dtype=np.float64)  # (n_items, n_steps)
    return arr.mean(axis=0), arr.std(axis=0), arr.shape


def main():
    print("Loading model/tokenizer (once) ...")
    model = B.AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tok = B.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    saved = torch.load(DIRECTION_PATH, map_location="cpu")
    direction = saved["direction"].float().to(DEVICE)
    raw_norm = float(direction.norm().item())
    print(f"direction: layer={saved.get('layer')} raw_norm={raw_norm:.4f} "
          f"(open-loop alpha={OPEN_LOOP_ALPHA:g} adds {OPEN_LOOP_ALPHA*raw_norm:.1f} to projection)")

    rows = load_items(N_ITEMS)
    print(f"Loaded {len(rows)} eval items (first {N_ITEMS} from {os.path.basename(EVAL_ITEMS)}).")

    conditions = [
        ("clean",           dict(mode="add",   alpha=0.0,             cstar=0.0,   beta=0.0)),
        ("open_loop_add_a8", dict(mode="add",   alpha=OPEN_LOOP_ALPHA, cstar=0.0,   beta=0.0)),
        ("clamp_c60",       dict(mode="clamp", alpha=0.0,             cstar=CSTAR, beta=0.0)),
        ("cmom_c60_b0.8",   dict(mode="cmom",  alpha=0.0,             cstar=CSTAR, beta=BETA)),
    ]

    out = {
        "meta": {
            "model_path": MODEL_PATH, "direction_path": DIRECTION_PATH,
            "eval_items": EVAL_ITEMS, "layer": LAYER, "n_items": len(rows),
            "gen_length": GEN_LENGTH, "steps": STEPS, "block_length": BLOCK_LENGTH,
            "temperature": TEMPERATURE, "remasking": REMASKING,
            "cstar": CSTAR, "beta": BETA, "open_loop_alpha": OPEN_LOOP_ALPHA,
            "raw_norm": raw_norm,
            "projection_region": "last gen_length positions (appended answer block), pre-correction",
        },
        "conditions": {},
    }

    for name, kw in conditions:
        print(f"\n=== condition: {name}  ({kw}) ===")
        traj = run_condition(model, tok, rows, direction, **kw)
        mean, std, shape = summarize(traj)
        print(f"  collected trajectory shape {shape}; a_t start={mean[0]:.3f} "
              f"mid={mean[STEPS//2]:.3f} end={mean[-1]:.3f}")
        entry = {
            "mode": kw["mode"], "alpha": kw["alpha"], "cstar": kw["cstar"], "beta": kw["beta"],
            "a_t_mean": mean.tolist(), "a_t_std": std.tolist(),
        }
        if kw["mode"] in ("clamp", "cmom"):
            err = (kw["cstar"] - mean).tolist()
            entry["residual_error_mean"] = err  # cstar - a_t per step
            print(f"  residual error cstar-a_t: step0={err[0]:.3f} "
                  f"mid={err[STEPS//2]:.3f} final={err[-1]:.3f}")
        out["conditions"][name] = entry

    json_path = os.path.join(_HERE, "fig3_denoise_error.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    make_figure(out, os.path.join(_HERE, "fig3_denoise_error.png"))
    print("Done.")


def make_figure(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(out["meta"]["steps"])
    cstar = out["meta"]["cstar"]
    C = out["conditions"]
    colors = {
        "clean": "#4C72B0", "open_loop_add_a8": "#DD8452",
        "clamp_c60": "#55A868", "cmom_c60_b0.8": "#C44E52",
    }
    labels = {
        "clean": "clean (add a=0)",
        "open_loop_add_a8": "open-loop add (raw a=8)",
        "clamp_c60": "clamp (P, c*=60)",
        "cmom_c60_b0.8": "cmom (PI, c*=60, b=0.8)",
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # --- Panel 1: incoming projection a_t ---
    for name in ["clean", "open_loop_add_a8", "clamp_c60", "cmom_c60_b0.8"]:
        m = np.array(C[name]["a_t_mean"])
        s = np.array(C[name]["a_t_std"])
        ax1.plot(steps, m, color=colors[name], label=labels[name], lw=2)
        ax1.fill_between(steps, m - s, m + s, color=colors[name], alpha=0.15)
    ax1.axhline(cstar, color="gray", ls="--", lw=1, label=f"setpoint c*={cstar:g}")
    ax1.set_ylabel("incoming projection  a_t = <h, v_hat>  (gen region)")
    ax1.set_title("Fig.3-on-denoising-axis: incoming projection per denoising step (pre-correction)")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.3)

    # --- Panel 2: residual error cstar - a_t for closed-loop conditions ---
    for name in ["clamp_c60", "cmom_c60_b0.8"]:
        err = np.array(C[name]["residual_error_mean"])
        ax2.plot(steps, err, color=colors[name], label=labels[name], lw=2)
    ax2.axhline(0.0, color="gray", ls="--", lw=1, label="zero error (setpoint held)")
    ax2.set_xlabel("denoising step t (0..63)")
    ax2.set_ylabel("residual error  c* - a_t")
    ax2.set_title("Residual steady-state error the controller faces each step")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"Saved PNG  -> {png_path}")


if __name__ == "__main__":
    main()
