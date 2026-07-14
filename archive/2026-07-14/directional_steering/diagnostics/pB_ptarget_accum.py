#!/usr/bin/env python
"""Formulation-B precondition test on LLaDA-8B-Instruct.

QUESTION: over the diffusion denoising steps (0..63), does the model's
probability mass on the TARGET answer letter (the Black option) accumulate as a
controllable trajectory, and is that trajectory moved by open-loop steering?

We copy the bbq_eval.generate() loop VERBATIM and INSTRUMENT it: at every
denoising step we take the FINAL-LAYER logits over the current partial sequence
x_t, softmax over the full vocab, and read the probabilities of the letter
tokens at every gen-region position. Two conditions, one model load:
  1. clean   (alpha=0, no hook)
  2. steered (BiasSteerer add mode, RAW direction, alpha=8 at block L14)

Rigor: every stored number is measured; we run mandatory validation
(prob ranges, committed-token==parsed-letter, ground-truth separation of
final-step p_target for target vs non-target outcomes) and report pass rates
before any verdict.
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = "/home/lukas/users/shashmi/dlm_bias/eval"
_DS_DIR = "/home/lukas/users/shashmi/dlm_bias/directional_steering"
sys.path.insert(0, _EVAL_DIR)
sys.path.insert(0, _DS_DIR)
import bbq_eval as B  # noqa: E402

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DIRECTION_PATH = "/home/lukas/users/shashmi/dlm_bias/directional_steering/race_black_anchored_text.pt"
EVAL_ITEMS = "/home/lukas/users/shashmi/dlm_bias/experiments/data/black_referent_ambig_eval.jsonl"
LAYER = 14
DEVICE = "cuda"

GEN_LENGTH = 32
STEPS = 64
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
REMASKING = "low_confidence"
CFG_SCALE = 0.0
N_ITEMS = 16
ALPHA = 8.0

# Black group tags -- SAME set as directional_steering/build_anchored.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}
LETTERS = ["A", "B", "C"]

# Candidate letter token ids (probed from tokenizer):
#   plain  A=32 B=33 C=34 ;  space-prefixed  " A"=355 " B"=413 " C"=348.
# We track BOTH variants per letter so the readout matches whatever the model
# actually commits at the answer position (decided per item during validation).
PLAIN = {"A": 32, "B": 33, "C": 34}
SPACE = {"A": 355, "B": 413, "C": 348}
# Column order for the stored (steps, gen_pos, K) probability tensor.
CAND = [("A", PLAIN["A"]), ("B", PLAIN["B"]), ("C", PLAIN["C"]),
        ("A_sp", SPACE["A"]), ("B_sp", SPACE["B"]), ("C_sp", SPACE["C"])]
CAND_IDS = [tid for _, tid in CAND]
COL = {name: i for i, (name, _) in enumerate(CAND)}


def black_options(row):
    """Indices of options whose group tag is a Black tag (same as build_anchored)."""
    out = []
    for k in range(3):
        pair = B.get_answer_info(row, k)
        if not pair:
            continue
        if str(pair[-1]).strip().lower() in BLACK_TAGS:
            out.append(k)
    return out


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
def instrumented_generate(model, prompt, cand_ids):
    """bbq_eval.generate() (verbatim logic) + per-step logit instrumentation.

    Returns:
      x          final sequence (1, prompt_len+GEN_LENGTH)
      probs      (STEPS, GEN_LENGTH, K) softmax prob of each candidate token id
                 at each gen-region position at each denoising step
      mask_count (STEPS,) number of still-masked gen positions AFTER step i
    """
    mask_id = B.MASK_ID
    x = torch.full((1, prompt.shape[1] + GEN_LENGTH), mask_id, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_len = prompt.shape[1]

    assert GEN_LENGTH % BLOCK_LENGTH == 0
    num_blocks = GEN_LENGTH // BLOCK_LENGTH
    assert STEPS % num_blocks == 0
    steps_per_block = STEPS // num_blocks

    K = len(cand_ids)
    cand = torch.tensor(cand_ids, device=model.device)
    probs = np.zeros((STEPS, GEN_LENGTH, K), dtype=np.float64)
    mask_count = np.zeros((STEPS,), dtype=np.int64)

    step_global = 0
    for nb in range(num_blocks):
        b0 = prompt_len + nb * BLOCK_LENGTH
        b1 = prompt_len + (nb + 1) * BLOCK_LENGTH
        block_mask_index = x[:, b0:b1] == mask_id
        num_transfer_tokens = B.get_num_transfer_tokens(block_mask_index, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x).logits  # cfg_scale = 0.0 branch

            # --- INSTRUMENT: full-vocab softmax at gen positions, read candidates ---
            gen_logits = logits[0, prompt_len:, :].float()          # (GEN_LENGTH, V)
            p_full = torch.softmax(gen_logits, dim=-1)              # (GEN_LENGTH, V)
            probs[step_global] = p_full.index_select(1, cand).cpu().numpy()

            logits_with_noise = B.add_gumbel_noise(logits, TEMPERATURE)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if REMASKING == "low_confidence":
                p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            else:
                raise NotImplementedError(REMASKING)
            x0_p[:, b1:] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float("-inf"))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

            mask_count[step_global] = int((x[0, prompt_len:] == mask_id).sum().item())
            step_global += 1

    return x, probs, mask_count


def build_input(tok, row):
    base_prompt = B.build_prompt(row)
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": base_prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return torch.tensor(tok(prompt_text)["input_ids"], device=DEVICE).unsqueeze(0)


@torch.no_grad()
def run_condition(model, tok, items, cand_ids):
    """Run all items; return list of per-item dicts (no steerer handling here)."""
    results = []
    for rec in items:
        row = rec["row"]
        input_ids = build_input(tok, row)
        x, probs, mask_count = instrumented_generate(model, input_ids, cand_ids)
        prompt_len = input_ids.shape[1]
        gen_ids = x[0, prompt_len:].tolist()
        gen_text = tok.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        results.append({
            "gen_ids": gen_ids,
            "gen_text": gen_text,
            "probs": probs,            # (STEPS, GEN_LENGTH, K)
            "mask_count": mask_count,  # (STEPS,)
        })
    return results


def analyze(items, cond_results, cond_name):
    """Post-hoc analysis + validation for one condition. Returns dict."""
    per_item = []
    usable_idx = []
    val_prob_range_ok = True
    val_prob_sum_ok = True
    val_committed_ok = []   # committed token == parsed letter
    final_ptarget_target_outcome = []
    final_ptarget_nontarget_outcome = []

    for it, res in zip(items, cond_results):
        row = it["row"]
        target_letter = it["target_letter"]
        probs = res["probs"]            # (STEPS, GEN_LENGTH, K)
        gen_ids = res["gen_ids"]
        gen_text = res["gen_text"]

        # Validation (a): prob ranges / restricted sum <= 1.
        if not (np.all(probs >= -1e-9) and np.all(probs <= 1 + 1e-6)):
            val_prob_range_ok = False
        # restricted 4-way sum (A,B,C,target) must be <= 1 at every pos/step.
        # (subset of a full softmax). Use the 3 plain + 3 space letters as the
        # superset; any 4-subset is <= this 6-sum, so check the 6-sum <= 1.
        if np.any(probs.sum(axis=-1) > 1 + 1e-6):
            val_prob_sum_ok = False

        letter = B.parse_letter(gen_text, row)
        item_out = {
            "example_id": int(row.get("example_id", -1)),
            "target_letter": target_letter,
            "gen_text": gen_text,
            "parsed_letter": letter,
        }

        if letter is None:
            item_out["usable"] = False
            item_out["exclude_reason"] = "no_answer (parse_letter returned None)"
            per_item.append(item_out)
            continue

        # letter_pos = first gen position whose committed token id is that letter
        # (plain or space variant).
        want = {PLAIN[letter], SPACE[letter]}
        letter_pos = next((p for p, tid in enumerate(gen_ids) if tid in want), None)
        if letter_pos is None:
            item_out["usable"] = False
            item_out["exclude_reason"] = (
                f"parsed letter {letter} but no gen position holds its token id "
                f"(committed via substring fallback, not a letter token)")
            per_item.append(item_out)
            continue

        committed_tid = gen_ids[letter_pos]
        # Validation (b): committed token decodes to the parsed letter.
        committed_ok = committed_tid in want
        val_committed_ok.append(committed_ok)

        # Which variant scheme did the model use at the answer position?
        variant = "plain" if committed_tid in PLAIN.values() else "space"
        tgt_tid = PLAIN[target_letter] if variant == "plain" else SPACE[target_letter]
        tgt_col = CAND_IDS.index(tgt_tid)

        p_target_traj = probs[:, letter_pos, tgt_col]  # (STEPS,)
        final_is_target = (letter == target_letter)

        item_out.update({
            "usable": True,
            "letter_pos": letter_pos,
            "committed_token_id": committed_tid,
            "committed_ok": bool(committed_ok),
            "variant": variant,
            "target_token_id": tgt_tid,
            "final_is_target": bool(final_is_target),
            "p_target_traj": p_target_traj.tolist(),
            "mask_count": res["mask_count"].tolist(),
        })
        per_item.append(item_out)
        usable_idx.append(len(per_item) - 1)

        final_pt = float(p_target_traj[-1])
        if final_is_target:
            final_ptarget_target_outcome.append(final_pt)
        else:
            final_ptarget_nontarget_outcome.append(final_pt)

    # Mean/std trajectory over USABLE items.
    usable = [per_item[i] for i in usable_idx]
    if usable:
        traj = np.array([u["p_target_traj"] for u in usable], dtype=np.float64)  # (nu, STEPS)
        mean_traj = traj.mean(axis=0)
        std_traj = traj.std(axis=0)
    else:
        mean_traj = np.zeros(STEPS)
        std_traj = np.zeros(STEPS)

    n_final_target = sum(1 for u in usable if u["final_is_target"])

    return {
        "condition": cond_name,
        "n_total": len(items),
        "n_usable": len(usable),
        "n_excluded": len(items) - len(usable),
        "n_final_answer_is_target": n_final_target,
        "mean_p_target_traj": mean_traj.tolist(),
        "std_p_target_traj": std_traj.tolist(),
        "validation": {
            "prob_range_ok": bool(val_prob_range_ok),
            "restricted_sum_le_1_ok": bool(val_prob_sum_ok),
            "committed_token_matches_parsed_pass_rate":
                (float(np.mean(val_committed_ok)) if val_committed_ok else None),
            "n_committed_checked": len(val_committed_ok),
            "gt_final_ptarget_target_outcome_mean":
                (float(np.mean(final_ptarget_target_outcome))
                 if final_ptarget_target_outcome else None),
            "gt_final_ptarget_target_outcome_n": len(final_ptarget_target_outcome),
            "gt_final_ptarget_nontarget_outcome_mean":
                (float(np.mean(final_ptarget_nontarget_outcome))
                 if final_ptarget_nontarget_outcome else None),
            "gt_final_ptarget_nontarget_outcome_n": len(final_ptarget_nontarget_outcome),
        },
        "per_item": per_item,
    }


def settling_step(mean_traj):
    """Step index by which the trajectory reaches ~90% of its final value,
    measured from its t=0 baseline. Returns (settling_step, t0, final)."""
    m = np.asarray(mean_traj)
    t0, final = float(m[0]), float(m[-1])
    span = final - t0
    if abs(span) < 1e-9:
        return 0, t0, final
    thresh = t0 + 0.9 * span
    for t in range(len(m)):
        if (span > 0 and m[t] >= thresh) or (span < 0 and m[t] <= thresh):
            return t, t0, final
    return len(m) - 1, t0, final


def main():
    print("Loading model/tokenizer (once) ...")
    model = B.AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tok = B.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    saved = torch.load(DIRECTION_PATH, map_location="cpu")
    direction = saved["direction"].float().to(DEVICE)
    raw_norm = float(direction.norm().item())
    print(f"direction layer={saved.get('layer')} raw_norm={raw_norm:.4f}; "
          f"add alpha={ALPHA:g} adds {ALPHA*raw_norm:.1f} to projection")

    rows = load_items(N_ITEMS)
    print(f"Loaded {len(rows)} eval items.")

    # Determine target letter per item via black_options (exactly one Black opt).
    items = []
    skipped = []
    for row in rows:
        b = black_options(row)
        if len(b) != 1:
            skipped.append({"example_id": int(row.get("example_id", -1)),
                            "n_black_options": len(b)})
            continue
        black_idx = b[0]
        items.append({
            "row": row,
            "black_idx": black_idx,
            "target_letter": chr(ord("A") + black_idx),
        })
    print(f"Items with exactly one Black option: {len(items)}; "
          f"skipped (not exactly one): {len(skipped)}")

    meta = {
        "model_path": MODEL_PATH, "direction_path": DIRECTION_PATH,
        "eval_items": EVAL_ITEMS, "layer": LAYER, "n_items_loaded": len(rows),
        "n_items_used": len(items), "gen_length": GEN_LENGTH, "steps": STEPS,
        "block_length": BLOCK_LENGTH, "temperature": TEMPERATURE,
        "remasking": REMASKING, "cfg_scale": CFG_SCALE, "alpha": ALPHA,
        "raw_norm": raw_norm,
        "black_tags": sorted(BLACK_TAGS),
        "letter_token_ids_plain": PLAIN, "letter_token_ids_space": SPACE,
        "candidate_column_order": [name for name, _ in CAND],
        "target_letter_rule": "chr(ord('A')+black_idx), black_idx = unique Black option index",
        "ptarget_readout": ("P(target token) at the gen position holding the model's "
                            "final parsed letter; target-token VARIANT (plain vs space) "
                            "chosen to match the variant actually committed at that position"),
        "skipped_items": skipped,
    }

    # ---- Condition 1: clean (no hook) ----
    print("\n=== condition: clean (alpha=0, no hook) ===")
    clean_res = run_condition(model, tok, items, CAND_IDS)

    # ---- Condition 2: steered (add, raw, alpha=8, block L14) ----
    print("=== condition: steered (add, raw dir, alpha=8, L14) ===")
    steerer = B.BiasSteerer(direction, alpha=ALPHA, mode="add")
    block = B.resolve_module(model, B.BLOCKS_PATH)[LAYER]
    steerer.attach(block)
    try:
        steer_res = run_condition(model, tok, items, CAND_IDS)
    finally:
        steerer.detach()

    clean_an = analyze(items, clean_res, "clean")
    steer_an = analyze(items, steer_res, "steered")

    for an in (clean_an, steer_an):
        ss, t0, final = settling_step(an["mean_p_target_traj"])
        an["settling_step_90pct"] = ss
        an["p_target_t0"] = t0
        an["p_target_final"] = final

    out = {"meta": meta, "conditions": {"clean": clean_an, "steered": steer_an}}
    json_path = os.path.join(_HERE, "pB_ptarget_accum.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")

    make_figure(out, os.path.join(_HERE, "pB_ptarget_accum.png"))

    # ---- console summary ----
    print("\n" + "=" * 70)
    for name in ("clean", "steered"):
        a = out["conditions"][name]
        v = a["validation"]
        print(f"[{name}] usable={a['n_usable']}/{a['n_total']} "
              f"excluded={a['n_excluded']} final==target={a['n_final_answer_is_target']}")
        print(f"   validation: prob_range_ok={v['prob_range_ok']} "
              f"sum<=1={v['restricted_sum_le_1_ok']} "
              f"committed_match={v['committed_token_matches_parsed_pass_rate']}")
        print(f"   GT sep: target-outcome final p_target mean="
              f"{v['gt_final_ptarget_target_outcome_mean']} "
              f"(n={v['gt_final_ptarget_target_outcome_n']}) | "
              f"non-target-outcome mean="
              f"{v['gt_final_ptarget_nontarget_outcome_mean']} "
              f"(n={v['gt_final_ptarget_nontarget_outcome_n']})")
        print(f"   p_target t0={a['p_target_t0']:.4f} final={a['p_target_final']:.4f} "
              f"settling(90%)=step {a['settling_step_90pct']}")
    print("=" * 70)
    print("Done.")


def make_figure(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(out["meta"]["steps"])
    C = out["conditions"]
    colors = {"clean": "#4C72B0", "steered": "#DD8452"}
    labels = {"clean": "clean (alpha=0)", "steered": "steered (add raw, alpha=8, L14)"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ("clean", "steered"):
        a = C[name]
        # faint per-item usable lines
        for u in a["per_item"]:
            if u.get("usable"):
                ax.plot(steps, u["p_target_traj"], color=colors[name], alpha=0.12, lw=0.8)
        m = np.array(a["mean_p_target_traj"])
        ax.plot(steps, m, color=colors[name], lw=2.5,
                label=f"{labels[name]}  (mean, n_usable={a['n_usable']})")

    # mark generation-complete step (mask_count hits 0) using a usable clean item.
    complete_step = None
    for u in C["clean"]["per_item"]:
        if u.get("usable"):
            mc = u["mask_count"]
            for t, c in enumerate(mc):
                if c == 0:
                    complete_step = t
                    break
            break
    if complete_step is not None:
        ax.axvline(complete_step, color="gray", ls="--", lw=1,
                   label=f"gen complete (all committed) @ step {complete_step}")

    baseline = 1.0 / 3.0
    ax.axhline(baseline, color="black", ls=":", lw=1, alpha=0.6,
               label="uniform 3-way baseline (1/3)")
    ax.set_xlabel("denoising step t (0..63)")
    ax.set_ylabel("P(target letter token) at answer position")
    ax.set_title("Formulation-B precondition: does p_target accumulate & move with steering?")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"Saved PNG  -> {png_path}")


if __name__ == "__main__":
    main()
