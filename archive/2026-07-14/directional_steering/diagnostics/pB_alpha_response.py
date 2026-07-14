#!/usr/bin/env python
"""Formulation-B actuation->output response curve on LLaDA-8B-Instruct.

Extends the VALIDATED pB_ptarget_accum prototype (agent-a44467e25b9713521):
same model/item loading, same target-letter derivation (chr('A')+black_idx,
exactly one Black option), same instrumented generate loop (gen=32/steps=64/
block=32, temp=0, low_confidence), same readout of the target-letter token at
the committed answer position, same mandatory validation.

NEW: instead of {clean, alpha=8} we sweep OPEN-LOOP add steering at L14 (raw
direction) over alpha in {0,2,4,8,12,16,24,32} at n=64, ONE model load, and
trace how the controlled variable p_target responds to the gain alpha.

Rigor: every stored number is measured. Per alpha we run ground-truth
separation (final p_target for target-outcome vs non-target-outcome items),
committed-token==parsed-letter pass rate, and usable/excluded counts. Ambiguous
/ unparseable items are counted, never smoothed away.
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
N_ITEMS = 64
ALPHAS = [0.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
PRECOMMIT_STEPS = 32  # controllable window = steps 0..31 (first half of denoising)

# Black group tags -- SAME set as directional_steering/build_anchored.py.
BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}
LETTERS = ["A", "B", "C"]

PLAIN = {"A": 32, "B": 33, "C": 34}
SPACE = {"A": 355, "B": 413, "C": 348}
CAND = [("A", PLAIN["A"]), ("B", PLAIN["B"]), ("C", PLAIN["C"]),
        ("A_sp", SPACE["A"]), ("B_sp", SPACE["B"]), ("C_sp", SPACE["C"])]
CAND_IDS = [tid for _, tid in CAND]
COL = {name: i for i, (name, _) in enumerate(CAND)}


def black_options(row):
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
    """bbq_eval.generate() (verbatim logic) + per-step logit instrumentation."""
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

            gen_logits = logits[0, prompt_len:, :].float()
            p_full = torch.softmax(gen_logits, dim=-1)
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
            "probs": probs,
            "mask_count": mask_count,
        })
    return results


def pred_class(letter, target_letter, unknown_letter):
    """Classify the parsed answer into target / abstain / non_target."""
    if letter is None:
        return "unparseable"
    if letter == target_letter:
        return "target"
    if unknown_letter is not None and letter == unknown_letter:
        return "abstain"
    return "non_target"


def analyze(items, cond_results, alpha):
    """Post-hoc analysis + validation for one alpha condition."""
    per_item = []
    usable_idx = []
    val_prob_range_ok = True
    val_prob_sum_ok = True
    val_committed_ok = []
    final_ptarget_target_outcome = []
    final_ptarget_nontarget_outcome = []

    # pick-rate counts over all USED items (exactly-one-black).
    cls_counts = {"target": 0, "non_target": 0, "abstain": 0, "unparseable": 0}
    precommit_per_item = []
    t0_per_item = []

    for it, res in zip(items, cond_results):
        row = it["row"]
        target_letter = it["target_letter"]
        unknown_letter = it["unknown_letter"]
        probs = res["probs"]
        gen_ids = res["gen_ids"]
        gen_text = res["gen_text"]

        if not (np.all(probs >= -1e-9) and np.all(probs <= 1 + 1e-6)):
            val_prob_range_ok = False
        if np.any(probs.sum(axis=-1) > 1 + 1e-6):
            val_prob_sum_ok = False

        letter = B.parse_letter(gen_text, row)
        cls = pred_class(letter, target_letter, unknown_letter)
        cls_counts[cls] += 1

        item_out = {
            "example_id": int(row.get("example_id", -1)),
            "target_letter": target_letter,
            "unknown_letter": unknown_letter,
            "gen_text": gen_text,
            "parsed_letter": letter,
            "pred_class": cls,
        }

        if letter is None:
            item_out["usable"] = False
            item_out["exclude_reason"] = "no_answer (parse_letter returned None)"
            per_item.append(item_out)
            continue

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
        committed_ok = committed_tid in want
        val_committed_ok.append(committed_ok)

        variant = "plain" if committed_tid in PLAIN.values() else "space"
        tgt_tid = PLAIN[target_letter] if variant == "plain" else SPACE[target_letter]
        tgt_col = CAND_IDS.index(tgt_tid)

        p_target_traj = probs[:, letter_pos, tgt_col]
        final_is_target = (letter == target_letter)

        precommit = float(p_target_traj[:PRECOMMIT_STEPS].mean())
        t0_val = float(p_target_traj[0])
        precommit_per_item.append(precommit)
        t0_per_item.append(t0_val)

        item_out.update({
            "usable": True,
            "letter_pos": letter_pos,
            "committed_token_id": committed_tid,
            "committed_ok": bool(committed_ok),
            "variant": variant,
            "target_token_id": tgt_tid,
            "final_is_target": bool(final_is_target),
            "final_p_target": float(p_target_traj[-1]),
            "precommit_p_target": precommit,
            "t0_p_target": t0_val,
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

    usable = [per_item[i] for i in usable_idx]
    if usable:
        traj = np.array([u["p_target_traj"] for u in usable], dtype=np.float64)
        mean_traj = traj.mean(axis=0)
        std_traj = traj.std(axis=0)
        final_vals = traj[:, -1]
        final_mean = float(final_vals.mean())
        final_std = float(final_vals.std())
        precommit_mean = float(np.mean(precommit_per_item))
        precommit_std = float(np.std(precommit_per_item))
        t0_mean = float(np.mean(t0_per_item))
    else:
        mean_traj = np.zeros(STEPS)
        std_traj = np.zeros(STEPS)
        final_mean = final_std = precommit_mean = precommit_std = t0_mean = None

    n_used = len(items)
    n_final_target = cls_counts["target"]

    return {
        "alpha": alpha,
        "n_used": n_used,
        "n_usable": len(usable),
        "n_excluded": n_used - len(usable),
        "final_p_target_mean": final_mean,
        "final_p_target_std": final_std,
        "precommit_p_target_mean": precommit_mean,
        "precommit_p_target_std": precommit_std,
        "t0_p_target_mean": t0_mean,
        # pick-rate classification over ALL used items:
        "class_counts": cls_counts,
        "target_pick_rate": cls_counts["target"] / n_used if n_used else None,
        "non_target_pick_rate": cls_counts["non_target"] / n_used if n_used else None,
        "abstain_rate": cls_counts["abstain"] / n_used if n_used else None,
        "unparseable_rate": cls_counts["unparseable"] / n_used if n_used else None,
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


def main():
    print("Loading model/tokenizer (once) ...", flush=True)
    model = B.AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tok = B.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    saved = torch.load(DIRECTION_PATH, map_location="cpu")
    direction = saved["direction"].float().to(DEVICE)
    raw_norm = float(direction.norm().item())
    print(f"direction layer={saved.get('layer')} raw_norm={raw_norm:.4f}", flush=True)

    rows = load_items(N_ITEMS)
    print(f"Loaded {len(rows)} eval items.", flush=True)

    items = []
    skipped = []
    for row in rows:
        b = black_options(row)
        if len(b) != 1:
            skipped.append({"example_id": int(row.get("example_id", -1)),
                            "n_black_options": len(b)})
            continue
        black_idx = b[0]
        uk = B.unknown_index(row)
        items.append({
            "row": row,
            "black_idx": black_idx,
            "target_letter": chr(ord("A") + black_idx),
            "unknown_letter": (chr(ord("A") + uk) if uk is not None else None),
        })
    print(f"Items with exactly one Black option: {len(items)}; "
          f"skipped (not exactly one): {len(skipped)}", flush=True)

    block = B.resolve_module(model, B.BLOCKS_PATH)[LAYER]

    meta = {
        "model_path": MODEL_PATH, "direction_path": DIRECTION_PATH,
        "eval_items": EVAL_ITEMS, "layer": LAYER, "n_items_loaded": len(rows),
        "n_items_used": len(items), "gen_length": GEN_LENGTH, "steps": STEPS,
        "block_length": BLOCK_LENGTH, "temperature": TEMPERATURE,
        "remasking": REMASKING, "cfg_scale": CFG_SCALE, "alphas": ALPHAS,
        "steer_mode": "add", "steer_direction": "raw", "raw_norm": raw_norm,
        "precommit_window_steps": PRECOMMIT_STEPS,
        "black_tags": sorted(BLACK_TAGS),
        "letter_token_ids_plain": PLAIN, "letter_token_ids_space": SPACE,
        "candidate_column_order": [name for name, _ in CAND],
        "target_letter_rule": "chr(ord('A')+black_idx), black_idx = unique Black option index",
        "abstain_letter_rule": "chr(ord('A')+unknown_index(row)); abstain = pick the 'unknown' option",
        "ptarget_readout": ("P(target token) at the gen position holding the model's "
                            "final parsed letter; target-token variant (plain vs space) "
                            "chosen to match the committed variant at that position"),
        "pick_rate_denominator": "all USED items (exactly-one-Black), n_items_used",
        "skipped_items": skipped,
    }

    conditions = {}
    for alpha in ALPHAS:
        tag = f"alpha={alpha:g}"
        print(f"\n=== condition: add raw dir, {tag}, L{LAYER} ===", flush=True)
        steerer = None
        if alpha != 0.0:
            steerer = B.BiasSteerer(direction, alpha=alpha, mode="add")
            steerer.attach(block)
        try:
            res = run_condition(model, tok, items, CAND_IDS)
        finally:
            if steerer is not None:
                steerer.detach()
        an = analyze(items, res, alpha)
        conditions[f"{alpha:g}"] = an

        v = an["validation"]
        print(f"  usable={an['n_usable']}/{an['n_used']} excluded={an['n_excluded']} "
              f"| committed_match={v['committed_token_matches_parsed_pass_rate']}", flush=True)
        print(f"  final p_target mean={an['final_p_target_mean']} "
              f"precommit={an['precommit_p_target_mean']} t0={an['t0_p_target_mean']}", flush=True)
        print(f"  picks: target={an['class_counts']['target']} "
              f"non_target={an['class_counts']['non_target']} "
              f"abstain={an['class_counts']['abstain']} "
              f"unparseable={an['class_counts']['unparseable']} "
              f"| target_pick_rate={an['target_pick_rate']}", flush=True)
        print(f"  GT sep: target-outcome mean={v['gt_final_ptarget_target_outcome_mean']} "
              f"(n={v['gt_final_ptarget_target_outcome_n']}) | non-target-outcome "
              f"mean={v['gt_final_ptarget_nontarget_outcome_mean']} "
              f"(n={v['gt_final_ptarget_nontarget_outcome_n']})", flush=True)

    out = {"meta": meta, "conditions": conditions}
    json_path = os.path.join(_HERE, "pB_alpha_response.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {json_path}", flush=True)

    make_figure(out, os.path.join(_HERE, "pB_alpha_response.png"))
    print("Done.", flush=True)


def make_figure(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = out["conditions"]
    alphas = out["meta"]["alphas"]
    keys = [f"{a:g}" for a in alphas]
    n_used = out["meta"]["n_items_used"]

    def arr(field):
        return np.array([np.nan if C[k][field] is None else C[k][field]
                         for k in keys], dtype=float)

    final_mean = arr("final_p_target_mean")
    final_std = arr("final_p_target_std")
    precommit = arr("precommit_p_target_mean")
    pickrate = arr("target_pick_rate")  # always measurable from parsed text
    n_usable = np.array([C[k]["n_usable"] for k in keys])
    # a p_target point is trustworthy only if ALL used items are usable AND the
    # ground-truth target/non-target separation held (validated readout).
    trustworthy = np.array([
        (C[k]["n_usable"] == n_used)
        and (C[k]["validation"]["gt_final_ptarget_target_outcome_mean"] is not None)
        and (C[k]["validation"]["gt_final_ptarget_nontarget_outcome_mean"] is not None)
        and (C[k]["validation"]["gt_final_ptarget_target_outcome_mean"]
             > C[k]["validation"]["gt_final_ptarget_nontarget_outcome_mean"])
        for k in keys])
    fm_t = np.where(trustworthy, final_mean, np.nan)
    fs_t = np.where(trustworthy, final_std, np.nan)
    pc_t = np.where(trustworthy, precommit, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ---- Panel 1: response curve (actuation alpha -> output) ----
    ax = axes[0]
    # shade the region where the p_target readout is NOT trustworthy (usable<n).
    invalid_alphas = [a for a, t in zip(alphas, trustworthy) if not t]
    if invalid_alphas:
        x0 = min(invalid_alphas) - 1.0
        ax.axvspan(x0, max(alphas) + 1.0, color="#D62728", alpha=0.07)
        ax.annotate("p_target readout INVALID\n(usable < n, GT separation lost)",
                    xy=(x0 + 0.5, 0.93), fontsize=8, color="#D62728",
                    ha="left", va="top")
    ax.errorbar(alphas, fm_t, yerr=fs_t, marker="o", lw=2.2, capsize=3,
                color="#C44E52", label="final p_target (mean +/- std) [validated only]")
    ax.plot(alphas, pc_t, marker="s", lw=2.2, color="#4C72B0",
            label="pre-commit p_target (steps 0..31 mean) [validated only]")
    ax.plot(alphas, pickrate, marker="^", lw=2.2, color="#55A868",
            label="target-pick-rate (all alphas, from parsed text)")
    # annotate usable count at each pick-rate point
    for a, pr, nu in zip(alphas, pickrate, n_usable):
        ax.annotate(f"u={nu}", xy=(a, pr), fontsize=6.5, color="#2f6b3f",
                    ha="center", va="bottom")
    ax.axhline(1.0 / 3.0, color="black", ls=":", lw=1, alpha=0.6,
               label="uniform 3-way baseline (1/3)")
    ax.axvline(0, color="gray", ls="--", lw=1, alpha=0.5)
    ax.annotate("clean\n(alpha=0)", xy=(0, 0.02), fontsize=8, color="gray",
                ha="left", va="bottom")
    ax.set_xlabel("open-loop steering gain  alpha  (add, raw dir, L14)")
    ax.set_ylabel("controlled variable  /  pick-rate")
    ax.set_title(f"Actuation -> output response curve  (n_used={n_used})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(alphas)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # ---- Panel 2: per-step mean trajectory for representative alphas ----
    ax = axes[1]
    steps = np.arange(out["meta"]["steps"])
    rep = [a for a in [0.0, 4.0, 8.0, 16.0, 32.0] if f"{a:g}" in C]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rep)))
    for a, col in zip(rep, cmap):
        k = f"{a:g}"
        ax.plot(steps, C[k]["mean_p_target_traj"], lw=2, color=col,
                label=f"alpha={a:g} (n={C[k]['n_usable']})")
    ax.axhline(1.0 / 3.0, color="black", ls=":", lw=1, alpha=0.6,
               label="uniform 3-way (1/3)")
    ax.set_xlabel("denoising step t (0..63)")
    ax.set_ylabel("mean P(target letter token) at answer position")
    ax.set_title("Per-step trajectory by alpha")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Formulation-B: p_target response to open-loop steering gain", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=130)
    print(f"Saved PNG  -> {png_path}", flush=True)


if __name__ == "__main__":
    main()
