#!/usr/bin/env python
"""One steered UNQOVER-race condition, end to end.  Phase-2 sweep runner.

Every method -- ours and every prior-work baseline -- goes through THIS file, so
the decoding budget, the prompt, the parser, the metric and the result schema are
identical across methods by construction.  The ONLY thing that differs between
two runs is which intervention is attached.

  decoding      : LLaDA masked-diffusion sampler, gen_length=32, steps=64,
                  block_length=32, temperature=0, remasking="low_confidence"
                  (eval/bbq_eval.generate for the open-loop methods,
                  steering/denoise_pid.controlled_generate for ours).
  prompt        : unqover_hf.eval_harness.build_prompt (2 options, no "unknown")
  parser        : unqover_hf.eval_harness.strict_letter -- STRICT anchored only,
                  invalid kept in the denominator
  metric        : unqover_hf.eval_harness.summarize -> unqover_hf.metric.compute
  fit artifacts : unqover_hf/arrows_race_black.pt + unqover_hf/cache_race/*,
                  all built by build_arrows_race.py on the BUILD split ONLY.
                  NOTHING here reads steering/arrows.pt or baselines/cache/.

HOOK-FIRE ASSERTION.  Every intervention is asserted to have actually fired
(baselines/common.py:441-444's pattern), and the count is stamped into the run
summary as `hook_fire_count`.  A method whose hooks silently fail to attach
fails the run instead of quietly producing a clean-model number.

REALIZED ACTUATION.  For the additive all-layer methods (ours, open-loop normal,
mean-act, caa, actadd) the summary records the per-layer L2 norm of the vector
actually added and its mean over the 32 blocks, in the SAME units, so the
open-loop dose that matches the controller's effort can be computed without
re-running anything.  For ours it additionally records, per item, the mean
alpha(t) over the FULL run (all 64 steps) and over the COMMITTING WINDOW (steps
up to and including the one that commits the answer-letter slot).

  CPU:  python -m unqover_hf.run_race --selftest
  GPU:  CUDA_VISIBLE_DEVICES=6 python -m unqover_hf.run_race --method caa \
            --alpha 16 --split-path data/unqover_hirundo/build_race.jsonl \
            --out-dir results/unqover_hf/build_sweep
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "steering"))

from unqover_hf.loader import FAMILY_MEMBERS, load_jsonl, write_jsonl  # noqa: E402
from unqover_hf.splits import DEFAULT_BUILD, DEFAULT_EVAL  # noqa: E402
from unqover_hf import eval_harness as EH  # noqa: E402
from unqover_hf import metric as M  # noqa: E402
from unqover_hf.build_arrows_race import (  # noqa: E402
    ARROWS_OUT, CACHE_DIR, target_instances, git_sha,
)

TARGET = "black"
N_LAYERS = 32
GEN_LENGTH, STEPS, BLOCK_LENGTH = 32, 64, 32
LETTERS = ("A", "B")

DEFAULT_OUT_DIR = os.path.join(_ROOT, "results", "unqover_hf", "build_sweep")

# The methods, and which knobs each one takes.  Used by --selftest and by the
# sweep driver to check that every method really is swept.
METHOD_KNOBS = {
    "base":         (),
    "ours_PI":      ("amax",),
    "ours_PID":     ("amax",),
    "normal":       ("alpha",),
    "caa":          ("alpha",),
    "actadd":       ("alpha", "pair_index"),
    "aura_vanilla": ("gamma",),
    "aura_inject":  ("gamma",),
    "itic":         ("topk", "alpha"),
    "linearact":    ("variant", "strength"),
    "meanact_raw":  ("strength",),
    "meanact_unit": ("strength",),
}


# --------------------------------------------------------------------------- #
# Items: the target-family instances of a split, expanded to 4 prompts each.
# --------------------------------------------------------------------------- #
def load_items(split_path, family=TARGET, n_instances=0, seed=42):
    """Target-family instances of `split_path`, expanded to their 4 items.

    n_instances>0 takes a DETERMINISTIC seeded subsample of the instances (the
    pre-registered sweep subset), never a prefix -- a prefix would correlate
    with template/attribute file order."""
    recs = target_instances(load_jsonl(split_path), family)
    recs = sorted(recs, key=lambda r: r["instance_id"])
    if n_instances and n_instances < len(recs):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.permutation(len(recs))[:n_instances].tolist())
        recs = [recs[i] for i in idx]
    return recs, EH.build_items(recs, family)


# --------------------------------------------------------------------------- #
# Traced controlled generation.
#
# VERBATIM copy of steering/denoise_pid.controlled_generate (the control law,
# the sampler and the commit rule are byte-identical and reuse the SAME imported
# helpers), with ONE addition: it records `commit_step`, the denoising step at
# which the answer-letter slot (generation position 0) stops being MASK.  That
# step bounds the window in which actuation can still change the answer, and the
# effort-matching report needs it.  `--selftest` asserts this function is
# output-identical to the imported denoise_pid.controlled_generate on a
# deterministic stub model.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def controlled_generate_traced(model, steerer, controller, prompt, tgt_plain,
                               tgt_space, steer_on, steps=STEPS):
    import denoise_pid as D
    import bbq_eval as B

    mask_id = B.MASK_ID
    plen = prompt.shape[1]
    x = torch.full((1, plen + GEN_LENGTH), mask_id, dtype=torch.long,
                   device=model.device)
    x[:, :plen] = prompt.clone()

    assert GEN_LENGTH % BLOCK_LENGTH == 0
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    controller.reset()
    alpha_traj = np.zeros(steps, dtype=np.float64)
    pblack_traj = np.zeros(steps, dtype=np.float64)
    sat_traj = np.zeros(steps, dtype=bool)
    commit_step = None

    steerer.alpha = 0.0
    p_meas = D.p_black_from_logits(model(x).logits, plen, tgt_plain, tgt_space)

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

            p_meas = D.p_black_from_logits(logits, plen, tgt_plain, tgt_space)
            alpha_traj[t], pblack_traj[t], sat_traj[t] = steerer.alpha, p_meas, sat

            x0 = torch.argmax(B.add_gumbel_noise(logits, 0.0), dim=-1)
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
            if commit_step is None and bool(transfer_index[0, plen]):
                commit_step = t          # the answer-letter slot just committed
            t += 1

    return x, alpha_traj, pblack_traj, sat_traj, commit_step


# --------------------------------------------------------------------------- #
# Interventions.  Each builder returns
#   (attach_fn(model) -> handles, per-layer added-vector norms or None, config)
# and NEVER reads a BBQ artifact.
# --------------------------------------------------------------------------- #
def _arrows(path=ARROWS_OUT):
    blob = torch.load(path, map_location="cpu")
    return blob["r"].to(torch.float32)


def _unit_rows(r):
    import common as C
    return C.unit_rows(r)


def build_intervention(method, args):
    """-> (attach_fn or None, delta_norms (32,) or None, config dict)."""
    import common as C  # noqa: F401  (shared fire counter lives here)

    if method == "base":
        return None, None, {"method": "base", "intervention": "none"}

    if method in ("ours_PI", "ours_PID"):
        # The controllers are NOT an item-independent attach: run() builds the
        # AllLayerSteerer, the PID and the per-item target letter itself, and
        # fills in the config there.  Nothing to attach here.  Realized actuation
        # is alpha(t)*vhat with ||vhat||=1, so the per-layer added-vector norm IS
        # alpha(t); it is time-varying and is reported in the `actuation` block.
        return None, None, {}

    if method == "normal":
        import pid_steer as P
        r = _arrows(args.arrows)
        inj = P.build_normal_injection(r, args.layer, args.alpha)   # (32,H)
        def attach_fn(model, inj=inj):
            hs = []
            for k in range(N_LAYERS):
                hs += C.attach(model, C.block_paths([k]), C.add_vec_hook(inj[k]))
            return hs
        return (attach_fn, inj.norm(dim=1),
                {"method": "open_loop_normal", "alpha": args.alpha,
                 "source_layer": args.layer, "arrows": args.arrows,
                 "granularity": "block_residual_all32",
                 "note": "alpha * unit(r[source_layer]) added at all 32 blocks, "
                         "every denoising step -- the constant-alpha twin of ours"})

    if method == "caa":
        import caa
        u = _unit_rows(_arrows(args.arrows))
        vec = caa.build_injection(layer=args.layer, alpha=args.alpha, arrows=u)
        norms = torch.zeros(N_LAYERS)
        norms[args.layer] = vec.norm()
        return (caa.make_attach_fn(layer=args.layer, alpha=args.alpha, arrows=u),
                norms,
                {"method": "caa", "alpha": args.alpha, "layer": args.layer,
                 "arrows": args.arrows,
                 "granularity": "block_residual_single_layer"})

    if method == "actadd":
        import actadd
        path = os.path.join(args.cache_dir, "actadd_dir_pair%d.pt" % args.pair_index)
        blob = torch.load(path, map_location="cpu")
        r = blob["r"].to(torch.float32)
        vec = actadd.build_injection(alpha=args.alpha, layer=args.layer, r=r)
        norms = torch.zeros(N_LAYERS)
        norms[args.layer] = vec.norm()
        return (actadd.make_attach_fn(alpha=args.alpha, layer=args.layer, r=r),
                norms,
                {"method": "actadd_single_pair", "alpha": args.alpha,
                 "layer": args.layer, "pair_index": args.pair_index,
                 "pair_target": blob["target_subject"],
                 "pair_other": blob["other_subject"], "artifact": path,
                 "granularity": "block_residual_single_layer"})

    if method in ("aura_vanilla", "aura_inject"):
        import aura
        path = os.path.join(args.cache_dir, "aura_auroc.pt")
        auroc = torch.load(path, map_location="cpu")["auroc"]
        mode = "vanilla" if method == "aura_vanilla" else "inject"
        eff = auroc
        if mode == "vanilla":
            # AurA's published suppression gate is  1 - 2*relu(auroc-0.5), which
            # has NO dose knob, so the method could not be swept on an equal
            # footing.  We give it the one-parameter family
            #     gate_gamma = 1 - gamma*2*relu(auroc-0.5),
            # whose gamma=1 member IS the published gate exactly.  It is applied
            # through aura's OWN suppression_gate by feeding it the reparameter-
            # ised score  auroc' = 0.5 + gamma*relu(auroc-0.5)  (identity at
            # gamma=1), so aura.py's gate and hook code are untouched.
            eff = 0.5 + args.gamma * torch.clamp(auroc - 0.5, min=0)
        gate = aura.gate_for(eff, mode, args.gamma)
        return (aura.attach_fn(mode, gamma=args.gamma, auroc=eff), None,
                {"method": "aura", "mode": mode, "gamma": args.gamma,
                 "gamma_semantics": ("dampening strength; gamma=1 is the published "
                                     "AurA suppression gate" if mode == "vanilla"
                                     else "amplification strength (aura.py default 1.0)"),
                 "artifact": path, "granularity": "mlp_hidden_all32",
                 "gate_min": float(gate.min()), "gate_max": float(gate.max()),
                 "gate_mean": float(gate.mean()),
                 "note": ("multiplicative per-neuron gate; not an additive "
                          "vector, so no alpha-equivalent actuation")})

    if method == "itic":
        import itic
        path = os.path.join(args.cache_dir, "itic_probes.pt")
        probes = itic.load_probes(path=path)
        inj = itic.build_injection(K=args.topk, alpha=args.alpha, probes=probes)
        d = inj["delta_heads"].to(torch.float32).reshape(N_LAYERS, -1)
        return (lambda m: itic.attach_fn(m, inj), d.norm(dim=1),
                {"method": "iti_c", "topk": args.topk, "alpha": args.alpha,
                 "artifact": path, "granularity": "attn_head",
                 "n_selected_heads": int(inj["mask"].sum()),
                 "layers_touched": inj["layers"]})

    if method == "linearact":
        import linearact
        path = os.path.join(args.cache_dir, "linearact_stats.pt")
        stats = linearact.load_stats(path=path)
        return (lambda m: linearact.attach_fn(m, variant=args.variant,
                                              strength=args.strength, stats=stats),
                None,
                {"method": "linear_act", "variant": args.variant,
                 "strength": args.strength, "artifact": path,
                 "granularity": "mlp_hidden_all32",
                 "note": ("per-neuron affine OT map; not an additive vector, so "
                          "no alpha-equivalent actuation")})

    if method in ("meanact_raw", "meanact_unit"):
        import meanact
        direction = "raw" if method == "meanact_raw" else "unit"
        r = _arrows(args.arrows)
        arr = r if direction == "raw" else _unit_rows(r)
        inj = meanact.build_injection(args.strength, direction=direction, arrows=arr)
        return (meanact.build_attach_fn(args.strength, direction=direction,
                                        arrows=arr),
                inj.norm(dim=1),
                {"method": "mean_act", "direction": direction,
                 "strength": args.strength, "gain": meanact.GAIN,
                 "arrows": args.arrows, "granularity": "block_residual_all32"})

    raise ValueError("unknown method %r" % method)


# --------------------------------------------------------------------------- #
def make_tag(args):
    m = args.method
    if m == "base":
        return "base"
    if m in ("ours_PI", "ours_PID"):
        return "%s_amax%s" % (m, ("%g" % args.amax).replace(".", "p"))
    if m in ("normal", "caa"):
        return "%s_a%s" % (m, ("%g" % args.alpha).replace(".", "p").replace("-", "m"))
    if m == "actadd":
        return "actadd_p%d_a%s" % (args.pair_index,
                                   ("%g" % args.alpha).replace(".", "p"))
    if m == "aura_vanilla":
        return "aura_vanilla_g%s" % ("%g" % args.gamma).replace(".", "p")
    if m == "aura_inject":
        return "aura_inject_g%s" % ("%g" % args.gamma).replace(".", "p")
    if m == "itic":
        return "itic_K%d_a%s" % (args.topk, ("%g" % args.alpha).replace(".", "p"))
    if m == "linearact":
        return "linearact_%s_s%s" % (args.variant,
                                     ("%g" % args.strength).replace(".", "p"))
    if m in ("meanact_raw", "meanact_unit"):
        return "%s_s%s" % (m, ("%g" % args.strength).replace(".", "p"))
    return m


# --------------------------------------------------------------------------- #
def run(args, model=None, tok=None):
    import common as C
    import bbq_eval as B
    import denoise_pid as D

    tag = args.tag or make_tag(args)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, tag + "_raw.jsonl")
    sum_path = os.path.join(out_dir, tag + "_summary.json")
    if os.path.exists(sum_path) and not args.force:
        print("[%s] SKIP (summary exists)" % tag, flush=True)
        return

    recs, items = load_items(args.split_path, TARGET, args.n_instances, args.seed)
    if args.limit:
        items = items[:args.limit]
    print("[%s] split=%s instances=%d items=%d CVD=%s"
          % (tag, os.path.basename(args.split_path), len(recs), len(items),
             os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)

    if model is None or tok is None:
        model, tok = C.load_model()
    attach_fn, delta_norms, cfg = build_intervention(args.method, args)
    is_ours = args.method in ("ours_PI", "ours_PID")

    t0 = time.time()
    raw = []
    actuation = []

    if is_ours:
        r = _arrows(args.arrows)
        vhat = (r[args.layer] / r[args.layer].norm().clamp(min=1e-12)).to(model.device)
        plain, space = D.letter_token_ids(tok)
        steerer = D.AllLayerSteerer(vhat)
        blocks = B.resolve_module(model, B.BLOCKS_PATH)
        assert len(blocks) == N_LAYERS
        steerer.attach(blocks)
        use_ki, use_kd = D.COND_MASK["PI" if args.method == "ours_PI" else "PID"]
        ctrl = D.PID(args.kp, args.ki if use_ki else 0.0,
                     args.kd if use_kd else 0.0,
                     D.SETPOINT, args.amax, antiwindup=True)
        cfg.update({"method": "decode_%s" % ("PI" if args.method == "ours_PI" else "PID"),
                    "gains": {"Kp": args.kp, "Ki": args.ki if use_ki else 0.0,
                              "Kd": args.kd if use_kd else 0.0},
                    "setpoint": D.SETPOINT, "alpha_max": args.amax,
                    "anti_windup": True, "actuator": "all_32_blocks",
                    "vhat_layer": args.layer, "arrows": args.arrows,
                    "granularity": "block_residual_all32"})
        try:
            for i, it in enumerate(items):
                tgt = it["target_present_at"]
                assert tgt in LETTERS, "item has no target option: %r" % it["id"]
                ptxt = tok.apply_chat_template(
                    [{"role": "user", "content": it["prompt"]}],
                    add_generation_prompt=True, tokenize=False)
                ids = torch.tensor(tok(ptxt)["input_ids"],
                                   device=model.device).unsqueeze(0)
                x, a, pt, sat, cstep = controlled_generate_traced(
                    model, steerer, ctrl, ids, plain[tgt], space[tgt], steer_on=True)
                gen = tok.batch_decode(x[:, ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()
                win = int(cstep) + 1 if cstep is not None else len(a)
                act = {"alpha_mean_full": float(a.mean()),
                       "alpha_mean_commit_window": float(a[:win].mean()),
                       "commit_step": (int(cstep) if cstep is not None else None),
                       "commit_window_steps": win,
                       "alpha_final": float(a[-1]), "sat_frac": float(sat.mean()),
                       "p_target_final": float(pt[-1])}
                actuation.append(act)
                raw.append(dict(it, model_output=gen, **act))
                if (i + 1) % 100 == 0:
                    print("[%s] %d/%d (%.0fs)" % (tag, i + 1, len(items),
                                                  time.time() - t0), flush=True)
        finally:
            steerer.detach()
        n_fired = steerer.fires
    else:
        C.reset_fire_count()
        handles = []
        if attach_fn is not None:
            handles = attach_fn(model)
            if not isinstance(handles, (list, tuple)):
                handles = [handles]
        try:
            for i, it in enumerate(items):
                ptxt = tok.apply_chat_template(
                    [{"role": "user", "content": it["prompt"]}],
                    add_generation_prompt=True, tokenize=False)
                ids = torch.tensor(tok(ptxt)["input_ids"],
                                   device=model.device).unsqueeze(0)
                out = B.generate(model, ids, steps=STEPS, gen_length=GEN_LENGTH,
                                 block_length=BLOCK_LENGTH, temperature=0.0,
                                 cfg_scale=0.0, remasking="low_confidence")
                gen = tok.batch_decode(out[:, ids.shape[1]:],
                                       skip_special_tokens=True)[0].strip()
                raw.append(dict(it, model_output=gen))
                if (i + 1) % 100 == 0:
                    print("[%s] %d/%d (%.0fs)" % (tag, i + 1, len(items),
                                                  time.time() - t0), flush=True)
        finally:
            for h in handles:
                h.remove()
        n_fired = C.get_fire_count()

    elapsed = time.time() - t0

    # ---- hook-fire assertion (baselines/common.py:441-444 pattern) --------- #
    if args.method != "base":
        assert n_fired > 0, (
            "[%s] hooks never fired (fire_count=0): the intervention attached "
            "nothing, or its hooks were not built with the common.py factories."
            % tag)
    else:
        assert n_fired == 0, "[base] something attached a hook to the clean run"

    # ---- score (STRICT parser only; invalid stays in the denominator) ------ #
    scored = EH.score_records(raw, target=TARGET)
    summary = EH.summarize(scored, target=TARGET, n_boot=args.n_boot,
                           n_noise=args.n_noise)

    act_summary = None
    if actuation:
        act_summary = {
            "mean_alpha_full_run": float(np.mean([a["alpha_mean_full"] for a in actuation])),
            "mean_alpha_commit_window": float(
                np.mean([a["alpha_mean_commit_window"] for a in actuation])),
            "mean_commit_step": float(np.mean(
                [a["commit_step"] for a in actuation if a["commit_step"] is not None])),
            "mean_commit_window_steps": float(
                np.mean([a["commit_window_steps"] for a in actuation])),
            "mean_sat_frac": float(np.mean([a["sat_frac"] for a in actuation])),
            "mean_p_target_final": float(np.mean([a["p_target_final"] for a in actuation])),
            "frac_p_target_final_ge_0p9": float(np.mean(
                [a["p_target_final"] >= 0.9 for a in actuation])),
            "basis_note": ("alpha_mean_full_run averages alpha(t) over ALL %d "
                           "denoising steps; alpha_mean_commit_window averages over "
                           "steps 0..commit_step of the answer-letter slot" % STEPS),
        }
    knobs = {k: getattr(args, k) for k in METHOD_KNOBS[args.method]}
    summary.update({
        "tag": tag,
        "sweep_method": args.method,
        "knobs": knobs,
        "config": cfg,
        "split_path": args.split_path,
        "n_instances_run": len(recs),
        "n_instances_available": len(target_instances(load_jsonl(args.split_path), TARGET)),
        "sweep_seed": args.seed,
        "hook_fire_count": int(n_fired),
        "hook_fire_per_item": float(n_fired) / max(1, len(items)),
        "elapsed_s": elapsed,
        "s_per_item": elapsed / max(1, len(items)),
        "gen_length": GEN_LENGTH, "steps": STEPS, "block_length": BLOCK_LENGTH,
        "temperature": 0.0, "remasking": "low_confidence",
        "delta_norms_per_layer": (None if delta_norms is None
                                  else [round(float(v), 5) for v in delta_norms]),
        "delta_norm_mean_over_layers": (None if delta_norms is None
                                        else float(delta_norms.mean())),
        "actuation": act_summary,
        "git_sha": git_sha(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    write_jsonl(raw_path, raw)
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=1)
    ts = summary["metric"]["target_stats"]
    print("[%s] DONE gap_raw=%+.4f gap_deb=%+.4f invalid=%.4f cov=%.3f "
          "mu=%.4f raw_skew=%.4f fire=%d %.0fs -> %s"
          % (tag, ts["pref_gap_raw"], ts["pref_gap_debiased"],
             summary["invalid_rate"], summary["coverage"],
             summary["metric"]["mu"], summary["metric"]["raw_skew_mu"],
             n_fired, elapsed, sum_path), flush=True)


# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print("[selftest] %-60s %s%s" % (name, "PASS" if cond else "FAIL",
                                         (" " + extra) if extra else ""))

    check("every method declares its swept knobs",
          set(METHOD_KNOBS) == {"base", "ours_PI", "ours_PID", "normal", "caa",
                                "actadd", "aura_vanilla", "aura_inject", "itic",
                                "linearact", "meanact_raw", "meanact_unit"})

    if os.path.exists(DEFAULT_BUILD):
        recs, items = load_items(DEFAULT_BUILD, TARGET)
        check("BUILD target instances = 450", len(recs) == 450, "(%d)" % len(recs))
        check("4 items per instance", len(items) == 4 * len(recs))
        check("every item has a target option (A or B)",
              all(it["target_present_at"] in LETTERS for it in items))
        members = FAMILY_MEMBERS[TARGET]
        check("target_present_at points at the black-family subject",
              all(it["subj0" if it["target_present_at"] == "A" else "subj1"]
                  in members for it in items))
        r1, _ = load_items(DEFAULT_BUILD, TARGET, n_instances=120, seed=42)
        r2, _ = load_items(DEFAULT_BUILD, TARGET, n_instances=120, seed=42)
        check("seeded subsample is deterministic",
              [r["instance_id"] for r in r1] == [r["instance_id"] for r in r2])
        check("seeded subsample has the requested size", len(r1) == 120)
        r3, _ = load_items(DEFAULT_BUILD, TARGET, n_instances=120, seed=43)
        check("a different seed gives a different subsample",
              [r["instance_id"] for r in r1] != [r["instance_id"] for r in r3])
        sub = {r["instance_id"] for r in r1}
        check("subsample spans many templates/attributes (not a file prefix)",
              len({r["template_id"] for r in recs if r["instance_id"] in sub}) >= 5
              and len({r["attribute_id"] for r in recs if r["instance_id"] in sub}) >= 10)
    else:
        print("[selftest] item checks: SKIP (run `python -m unqover_hf.splits`)")

    # ---- traced generation == the imported controller, on a stub model ----- #
    try:
        import denoise_pid as D

        class StubModel:
            """Deterministic fake LM: logits are a fixed function of position and
            of the current alpha, so any divergence in control or commit order
            shows up as a different output sequence."""
            device = "cpu"

            def __init__(self, steerer, vocab=64):
                self.steerer = steerer
                self.vocab = vocab

            def __call__(self, x):
                T = x.shape[1]
                pos = torch.arange(T, dtype=torch.float32).unsqueeze(1)
                vv = torch.arange(self.vocab, dtype=torch.float32).unsqueeze(0)
                base = torch.cos(0.37 * pos * (vv + 1.0)) + 0.11 * (x[0].float().unsqueeze(1) % 7)
                logits = (base + 0.5 * float(self.steerer.alpha)).unsqueeze(0)
                return type("O", (), {"logits": logits})()

        import bbq_eval as B
        old_mask = B.MASK_ID
        B.MASK_ID = 63
        prompt = torch.arange(5).unsqueeze(0)
        outs = []
        for fn in ("orig", "traced"):
            st = D.AllLayerSteerer(torch.zeros(4))
            m = StubModel(st)
            ctrl = D.PID(3.0, 0.1, 0.0, D.SETPOINT, 6.0, antiwindup=True)
            if fn == "orig":
                res = D.controlled_generate(m, st, ctrl, prompt, 1, 2, steer_on=True)
            else:
                res = controlled_generate_traced(m, st, ctrl, prompt, 1, 2,
                                                 steer_on=True)
            outs.append(res)
        B.MASK_ID = old_mask
        same = (torch.equal(outs[0][0], outs[1][0])
                and np.allclose(outs[0][1], outs[1][1])
                and np.allclose(outs[0][2], outs[1][2])
                and (outs[0][3] == outs[1][3]).all())
        check("traced generate == denoise_pid.controlled_generate (stub model)", same)
        check("traced generate reports a commit step for the answer slot",
              outs[1][4] is not None and 0 <= outs[1][4] < STEPS,
              str(outs[1][4]))
    except Exception as exc:                                   # noqa: BLE001
        check("traced-generate equivalence (%s: %s)" % (type(exc).__name__, exc), False)

    # ---- every method actually builds its intervention (CPU tensor math) --- #
    # This is the check that would have caught `ours_PI` falling through
    # build_intervention's dispatch to `raise ValueError("unknown method")`.
    if os.path.exists(ARROWS_OUT):
        from unqover_hf import sweep_grid as G
        parser = build_parser()
        built, failed = 0, []
        for m, knobs in G.configs():
            argv = ["--method", m]
            for k in sorted(knobs):
                argv += [G.FLAG[k], str(knobs[k])]
            a = parser.parse_args(argv)
            try:
                fn, norms, cfg = build_intervention(m, a)
            except Exception as exc:                           # noqa: BLE001
                failed.append((m, knobs, "%s: %s" % (type(exc).__name__, exc)))
                continue
            if m in ("base", "ours_PI", "ours_PID"):
                ok_one = fn is None
            else:
                ok_one = callable(fn)
            if not ok_one:
                failed.append((m, knobs, "attach_fn is %r" % (fn,)))
            else:
                built += 1
        check("every one of the %d pre-registered configs builds its "
              "intervention on CPU" % len(G.configs()),
              not failed, str(failed[:3]))
        check("all 61 configs accounted for", built + len(failed) == 61,
              "(built=%d failed=%d)" % (built, len(failed)))
    else:
        print("[selftest] intervention-build checks: SKIP (arrows not built yet)")

    # ---- tags are distinct per config -------------------------------------- #
    class A:
        pass
    tags = set()
    for m, knobs in METHOD_KNOBS.items():
        a = A()
        a.method = m
        a.amax, a.alpha, a.gamma, a.topk = 6.0, 4.0, 2.0, 48
        a.strength, a.variant, a.pair_index, a.layer = 2.0, "gaussian", 1, 14
        tags.add(make_tag(a))
    check("each method produces a distinct tag", len(tags) == len(METHOD_KNOBS),
          str(sorted(tags)))

    print("[selftest] run_race OVERALL: %s" % ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------- #
def sweep(args, base_argv):
    """Run a stride of the PRE-REGISTERED grid with ONE model load.

    Task j of a SLURM array of K runs configs j, j+K, j+2K, ... of
    unqover_hf.sweep_grid.arg_strings() -- so the array covers the grid exactly
    once, every task carries a mix of cheap and expensive methods, and the model
    is loaded once per task instead of once per config."""
    import common as C
    from unqover_hf import sweep_grid as G

    strings = G.arg_strings(args.n_instances or G.N_SWEEP, args.seed)
    mine = [(i, s) for i, s in enumerate(strings)
            if i % args.stride == args.offset]
    print("[sweep] task %d/%d -> %d of %d configs"
          % (args.offset, args.stride, len(mine), len(strings)), flush=True)
    model, tok = C.load_model()
    failures = []
    for i, s in mine:
        sub = build_parser().parse_args(base_argv + s.split())
        print("\n[sweep] === config %d/%d: %s ===" % (i + 1, len(strings), s),
              flush=True)
        try:
            run(sub, model=model, tok=tok)
        except Exception as exc:                               # noqa: BLE001
            failures.append((s, "%s: %s" % (type(exc).__name__, exc)))
            print("[sweep] *** CONFIG FAILED: %s -- %s: %s"
                  % (s, type(exc).__name__, exc), flush=True)
            import traceback
            traceback.print_exc()
    if failures:
        print("\n[sweep] %d CONFIG(S) FAILED:" % len(failures), flush=True)
        for s, e in failures:
            print("  %s  ->  %s" % (s, e), flush=True)
        sys.exit(2)
    print("[sweep] task %d DONE, %d configs" % (args.offset, len(mine)), flush=True)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="run a stride of the pre-registered grid, one model load")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--method", choices=sorted(METHOD_KNOBS))
    ap.add_argument("--split-path", default=DEFAULT_BUILD)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-instances", type=int, default=0,
                    help="0 = all target-family instances of the split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap on ITEMS (debug only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--arrows", default=ARROWS_OUT)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--amax", type=float, default=6.0)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--kd", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--variant", default="gaussian", choices=["gaussian", "empirical"])
    ap.add_argument("--pair-index", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--n-noise", type=int, default=0)
    return ap


def main():
    ap = build_parser()
    argv = sys.argv[1:]
    args = ap.parse_args(argv)

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.sweep:
        # strip the sweep-control flags; keep everything else (paths, arrows,
        # split, out-dir) as the shared base for every config in the stride
        keep, skip = [], {"--sweep", "--stride", "--offset", "--method"}
        it = iter(range(len(argv)))
        i = 0
        while i < len(argv):
            a = argv[i]
            if a in skip:
                i += 1 + (0 if a == "--sweep" else 1)
                continue
            keep.append(a)
            i += 1
        sweep(args, keep)
        return
    if not args.method:
        ap.error("--method required (or --selftest / --sweep)")
    run(args)


if __name__ == "__main__":
    main()
