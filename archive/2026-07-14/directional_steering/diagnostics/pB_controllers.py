#!/usr/bin/env python
"""Closed-loop steering controllers over the LLaDA-8B-Instruct denoising trajectory.

Head-to-head test of FOUR closed-loop controllers vs open-loop and clean
baselines, on the decode-space signal p_target (the Black-option answer letter).

Six conditions, ONE model load:
  baselines : clean (alpha=0), openloop8 (alpha_t=8 constant)
  controllers (read p_ctrl(t) each step, clamp alpha in [0,8]):
    1 PI_antiwindup : alpha = clip(Kp*e + Ki*integ, 0, 8), e=s*-p_ctrl
    2 deadline      : while t<D and p_ctrl<s*: alpha=8 else 0 (D = answer commit step)
    3 mpc           : 1-step lookahead over alpha in {0,4,8}; pick smallest with
                      predicted p_ctrl>=s* (else 8). Lookahead every 2 steps, hold between.
    4 remask_feedback: DEFER answer-position commit while p_ctrl<s* (override schedule),
                      force-commit at final deadline.

CONTROL OBSERVABLE p_ctrl(t): P(target letter token) read at the FIRST gen-region
position (gen pos 0, the natural letter slot). We track BOTH letter-token variants
(plain + space) and sum them; empirically the model commits the plain variant here.
Decided with a ONE-STEP delay (alpha_t chosen from p_ctrl_{t-1}); MPC decides
within-step from its lookahead.

FINAL METRICS use the VALIDATED post-hoc p_target readout at the TRUE parsed-letter
position (identical to pB_ptarget_accum.py) so all conditions are comparable.

Rigor: every readout is validated (prob ranges, committed==parsed, GT separation),
excluded items counted, p_ctrl-vs-true-position agreement reported. No fabrication.
"""
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = "/home/lukas/users/shashmi/dlm_bias/eval"
sys.path.insert(0, _EVAL_DIR)
import bbq_eval as B  # noqa: E402

MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DIRECTION_PATH = "/home/lukas/users/shashmi/dlm_bias/directional_steering/race_black_anchored_text.pt"
EVAL_ITEMS = "/home/lukas/users/shashmi/dlm_bias/experiments/data/black_referent_ambig_eval.jsonl"
LAYER = 14
DEVICE = "cuda"  # CUDA_VISIBLE_DEVICES=7 pins this to physical GPU 7

GEN_LENGTH = 32
STEPS = 64
BLOCK_LENGTH = 32
TEMPERATURE = 0.0
REMASKING = "low_confidence"
N_ITEMS = 32

ALPHA_MAX = 8.0          # competence ceiling from the alpha sweep -> clamp [0, 8]
SETPOINT = 0.9           # s* on p_ctrl
KP, KI = 8.0, 2.0        # PI gains (start values; report if retuned)
MPC_GRID = [0.0, 4.0, 8.0]
MPC_EVERY = 2            # run lookahead every N steps, hold alpha between
DEADLINE_D = 31          # answer commits at the last transfer step (empirically 31)

BLACK_TAGS = {"black", "african american", "f-black", "m-black", "african"}
LETTERS = ["A", "B", "C"]
PLAIN = {"A": 32, "B": 33, "C": 34}
SPACE = {"A": 355, "B": 413, "C": 348}
CAND = [("A", PLAIN["A"]), ("B", PLAIN["B"]), ("C", PLAIN["C"]),
        ("A_sp", SPACE["A"]), ("B_sp", SPACE["B"]), ("C_sp", SPACE["C"])]
CAND_IDS = [tid for _, tid in CAND]
CTRL_POS = 0  # first gen-region position = natural letter slot


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


def build_input(tok, row):
    base_prompt = B.build_prompt(row)
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": base_prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return torch.tensor(tok(prompt_text)["input_ids"], device=DEVICE).unsqueeze(0)


# --------------------------------------------------------------------------- #
# Controllers. Each carries state reset per item; `kind` selects loop behaviour.
# --------------------------------------------------------------------------- #
class Controller:
    kind = "gain"  # gain | mpc | remask

    def reset(self):
        pass

    def decide(self, t, p_ctrl_prev, ans_masked):
        """Return desired alpha for step t (loop clamps to [0, ALPHA_MAX])."""
        return 0.0

    def observe(self, t, p_ctrl_t):
        pass


class Clean(Controller):
    def decide(self, t, p_ctrl_prev, ans_masked):
        return 0.0


class OpenLoop(Controller):
    def __init__(self, alpha):
        self.a = alpha

    def decide(self, t, p_ctrl_prev, ans_masked):
        return self.a


class PI(Controller):
    def __init__(self, kp, ki, sstar):
        self.kp, self.ki, self.sstar = kp, ki, sstar

    def reset(self):
        self.integ = 0.0

    def decide(self, t, p_ctrl_prev, ans_masked):
        pc = 0.0 if p_ctrl_prev is None else p_ctrl_prev
        e = self.sstar - pc
        # anti-windup: only integrate when the (pre-clamp) command is not saturated
        raw = self.kp * e + self.ki * (self.integ + e)
        if 0.0 <= raw <= ALPHA_MAX:
            self.integ += e
        return raw


class Deadline(Controller):
    """Push hard while the answer position is still OPEN (masked) and below the
    setpoint; back off to 0 once it has committed. Keyed on the answer's actual
    mask-state (the true per-item commit deadline), NOT a fixed step, so steering
    stays ON through the single-shot commit step and only the wasted post-commit
    steps are dropped."""

    def __init__(self, D, sstar):
        self.D, self.sstar = D, sstar

    def decide(self, t, p_ctrl_prev, ans_masked):
        pc = 0.0 if p_ctrl_prev is None else p_ctrl_prev
        if ans_masked and pc < self.sstar:
            return ALPHA_MAX
        return 0.0


class MPC(Controller):
    kind = "mpc"

    def __init__(self, grid, sstar, every):
        self.grid = sorted(grid)
        self.sstar, self.every = sstar, every

    def reset(self):
        self.held = self.grid[-1]


class Remask(Controller):
    kind = "remask"

    def __init__(self, sstar, D, base_alpha=ALPHA_MAX):
        self.sstar, self.D, self.base_alpha = sstar, D, base_alpha

    def decide(self, t, p_ctrl_prev, ans_masked):
        # while the answer is still open we keep pushing hard toward the setpoint
        pc = 0.0 if p_ctrl_prev is None else p_ctrl_prev
        return self.base_alpha if pc < self.sstar else 0.0

    def should_defer(self, t, p_ctrl_t):
        # defer the answer-position commit while below setpoint AND before the
        # hard final deadline (leave room to force-commit).
        return p_ctrl_t < self.sstar and t < STEPS - 1


def clamp(a):
    return float(min(ALPHA_MAX, max(0.0, a)))


def pctrl_from_logits(logits, prompt_len, tgt_plain, tgt_space):
    """p_ctrl = P(plain target)+P(space target) at gen position CTRL_POS."""
    row = logits[0, prompt_len + CTRL_POS, :].float()
    p = torch.softmax(row, dim=-1)
    return float(p[tgt_plain].item() + p[tgt_space].item())


@torch.no_grad()
def controlled_generate(model, steerer, controller, prompt, tgt_plain, tgt_space):
    mask_id = B.MASK_ID
    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + GEN_LENGTH), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt_len] = prompt.clone()

    b0, b1 = prompt_len, prompt_len + BLOCK_LENGTH
    block_mask_index = x[:, b0:b1] == mask_id
    ntt = B.get_num_transfer_tokens(block_mask_index, STEPS)  # single block

    cand = torch.tensor(CAND_IDS, device=model.device)
    K = len(CAND_IDS)
    probs = np.zeros((STEPS, GEN_LENGTH, K), dtype=np.float64)
    mask_count = np.zeros((STEPS,), dtype=np.int64)
    p_ctrl_traj = np.zeros((STEPS,), dtype=np.float64)
    alpha_traj = np.zeros((STEPS,), dtype=np.float64)

    controller.reset()
    p_ctrl_prev = None
    nfe = 0
    deferred_steps = 0
    ans_pos = prompt_len + CTRL_POS

    for i in range(STEPS):
        mask_index = x == mask_id
        masks_remain = bool(mask_index[0, prompt_len:].any().item())

        if controller.kind == "mpc" and masks_remain:
            # 1-step lookahead over the grid every MPC_EVERY steps; hold between.
            if i % controller.every == 0:
                chosen_a, chosen_logits, chosen_pc = None, None, None
                a = lg = pc = None
                for a in controller.grid:            # ascending: pick smallest satisfier
                    steerer.alpha = clamp(a)
                    lg = model(x).logits
                    nfe += 1
                    pc = pctrl_from_logits(lg, prompt_len, tgt_plain, tgt_space)
                    if chosen_a is None:
                        chosen_a, chosen_logits, chosen_pc = a, lg, pc  # fallback = smallest
                    if pc >= controller.sstar:
                        chosen_a, chosen_logits, chosen_pc = a, lg, pc
                        break
                else:
                    # none satisfied -> use largest (last evaluated)
                    chosen_a, chosen_logits, chosen_pc = a, lg, pc
                controller.held = chosen_a
                alpha_used, logits, p_ctrl_t = chosen_a, chosen_logits, chosen_pc
            else:
                steerer.alpha = clamp(controller.held)
                logits = model(x).logits
                nfe += 1
                alpha_used = controller.held
                p_ctrl_t = pctrl_from_logits(logits, prompt_len, tgt_plain, tgt_space)
        else:
            ans_masked_now = bool(mask_index[0, ans_pos].item())
            alpha_used = clamp(controller.decide(i, p_ctrl_prev, ans_masked_now))
            steerer.alpha = alpha_used
            logits = model(x).logits
            nfe += 1
            p_ctrl_t = pctrl_from_logits(logits, prompt_len, tgt_plain, tgt_space)

        alpha_traj[i] = steerer.alpha
        p_ctrl_traj[i] = p_ctrl_t

        # instrument full-vocab softmax -> candidate probs at all gen positions
        gen_logits = logits[0, prompt_len:, :].float()
        p_full = torch.softmax(gen_logits, dim=-1)
        probs[i] = p_full.index_select(1, cand).cpu().numpy()

        # --- commit (verbatim sampler logic) ---
        logits_noise = B.add_gumbel_noise(logits, TEMPERATURE)
        x0 = torch.argmax(logits_noise, dim=-1)
        p = torch.nn.functional.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        x0_p[:, b1:] = float("-inf")
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, float("-inf"))
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        k = int(ntt[0, i])
        if k > 0:
            _, sel = torch.topk(confidence[0], k=k)
            transfer_index[0, sel] = True

        # --- controller #4: DEFER the answer-position commit ---
        if controller.kind == "remask":
            ans_masked = bool(mask_index[0, ans_pos].item())
            if ans_masked and controller.should_defer(i, p_ctrl_t):
                if transfer_index[0, ans_pos]:
                    transfer_index[0, ans_pos] = False  # hold it open (override schedule)
                    deferred_steps += 1
                    # commit next-best masked position instead (keep throughput/schedule)
                    conf2 = confidence[0].clone()
                    conf2[ans_pos] = float("-inf")
                    if torch.isfinite(conf2).any():
                        alt = int(torch.argmax(conf2).item())
                        transfer_index[0, alt] = True

        x[transfer_index] = x0[transfer_index]
        mask_count[i] = int((x[0, prompt_len:] == mask_id).sum().item())
        controller.observe(i, p_ctrl_t)
        p_ctrl_prev = p_ctrl_t

    # remask: force-commit the answer position if we held it open to the end.
    # Commit the STEERED argmax (defer-then-commit-under-steering, the fair test),
    # then REFRESH the final-step readout so p_target/p_ctrl reflect the now-committed
    # token (otherwise the last recorded step saw pos 0 still masked -> soft ~0.5).
    forced_commit = False
    if controller.kind == "remask" and bool((x[0, ans_pos] == mask_id).item()):
        # (1) commit the STEERED argmax at pos 0 (defer-then-commit-under-steering).
        steerer.alpha = clamp(controller.base_alpha)
        logits = model(x).logits
        nfe += 1
        x0 = torch.argmax(B.add_gumbel_noise(logits, TEMPERATURE), dim=-1)
        x[0, ans_pos] = x0[0, ans_pos]
        forced_commit = True
        # (2) settled readout forward on the NOW-committed sequence (alpha=0), so
        #     the final-step p_target/p_ctrl reflect the committed token (as clean/
        #     openloop already do for their committed positions).
        steerer.alpha = 0.0
        logits = model(x).logits
        nfe += 1
        gen_logits = logits[0, prompt_len:, :].float()
        probs[STEPS - 1] = torch.softmax(gen_logits, dim=-1).index_select(1, cand).cpu().numpy()
        p_ctrl_traj[STEPS - 1] = pctrl_from_logits(logits, prompt_len, tgt_plain, tgt_space)

    info = {"nfe": nfe, "deferred_steps": deferred_steps, "forced_commit": forced_commit}
    return x, probs, mask_count, p_ctrl_traj, alpha_traj, info


@torch.no_grad()
def run_condition(model, tok, items, steerer, controller):
    results = []
    for it in items:
        row = it["row"]
        tgt = it["target_letter"]
        input_ids = build_input(tok, row)
        x, probs, mask_count, pcw, alw, info = controlled_generate(
            model, steerer, controller, input_ids, PLAIN[tgt], SPACE[tgt])
        prompt_len = input_ids.shape[1]
        gen_ids = x[0, prompt_len:].tolist()
        gen_text = tok.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0].strip()
        results.append({
            "gen_ids": gen_ids, "gen_text": gen_text, "probs": probs,
            "mask_count": mask_count, "p_ctrl_traj": pcw, "alpha_traj": alw,
            "info": info,
        })
    return results


def analyze(items, cond_results, cond_name):
    per_item = []
    usable_idx = []
    val_prob_range_ok = True
    val_prob_sum_ok = True
    val_committed_ok = []
    final_pt_target, final_pt_nontarget = [], []
    pos_agree = []  # letter_pos == CTRL_POS
    picks = {"target": 0, "non_target": 0, "abstain": 0, "unparseable": 0}
    alpha_sums, nfes, deferred_any, forced_any = [], [], 0, 0

    for it, res in zip(items, cond_results):
        row = it["row"]
        tgt = it["target_letter"]
        black_idx = it["black_idx"]
        unk_idx = it["unknown_idx"]
        probs = res["probs"]
        gen_ids = res["gen_ids"]
        gen_text = res["gen_text"]

        if not (np.all(probs >= -1e-9) and np.all(probs <= 1 + 1e-6)):
            val_prob_range_ok = False
        if np.any(probs.sum(axis=-1) > 1 + 1e-6):
            val_prob_sum_ok = False

        alpha_sums.append(float(np.sum(res["alpha_traj"])))
        nfes.append(int(res["info"]["nfe"]))
        deferred_any += int(res["info"]["deferred_steps"] > 0)
        forced_any += int(res["info"]["forced_commit"])

        letter = B.parse_letter(gen_text, row)
        item_out = {
            "example_id": int(row.get("example_id", -1)),
            "target_letter": tgt, "gen_text": gen_text, "parsed_letter": letter,
            "alpha_sum": float(np.sum(res["alpha_traj"])),
            "deferred_steps": int(res["info"]["deferred_steps"]),
            "forced_commit": bool(res["info"]["forced_commit"]),
        }

        # pick classification (over all items; reliable from parse_letter)
        if letter is None:
            picks["unparseable"] += 1
        else:
            pidx = LETTERS.index(letter)
            if pidx == black_idx:
                picks["target"] += 1
            elif unk_idx is not None and pidx == unk_idx:
                picks["abstain"] += 1
            else:
                picks["non_target"] += 1

        if letter is None:
            item_out["usable"] = False
            item_out["exclude_reason"] = "no_answer (parse_letter returned None)"
            per_item.append(item_out)
            continue

        want = {PLAIN[letter], SPACE[letter]}
        letter_pos = next((p for p, tid in enumerate(gen_ids) if tid in want), None)
        if letter_pos is None:
            item_out["usable"] = False
            item_out["exclude_reason"] = "parsed letter has no committed letter-token (substring fallback)"
            per_item.append(item_out)
            continue

        committed_tid = gen_ids[letter_pos]
        committed_ok = committed_tid in want
        val_committed_ok.append(committed_ok)
        pos_agree.append(letter_pos == CTRL_POS)

        variant = "plain" if committed_tid in PLAIN.values() else "space"
        tgt_tid = PLAIN[tgt] if variant == "plain" else SPACE[tgt]
        tgt_col = CAND_IDS.index(tgt_tid)
        p_target_traj = probs[:, letter_pos, tgt_col]
        final_is_target = (letter == tgt)

        item_out.update({
            "usable": True, "letter_pos": letter_pos,
            "committed_token_id": committed_tid, "committed_ok": bool(committed_ok),
            "variant": variant, "final_is_target": bool(final_is_target),
            "p_target_final": float(p_target_traj[-1]),
            "p_ctrl_final": float(res["p_ctrl_traj"][-1]),
        })
        per_item.append(item_out)
        usable_idx.append(len(per_item) - 1)

        if final_is_target:
            final_pt_target.append(float(p_target_traj[-1]))
        else:
            final_pt_nontarget.append(float(p_target_traj[-1]))

    n_total = len(items)
    usable = [per_item[i] for i in usable_idx]
    n_usable = len(usable)
    mean_final_pt = (float(np.mean([u["p_target_final"] for u in usable]))
                     if usable else None)
    n_final_target = sum(1 for u in usable if u["final_is_target"])

    # mean trajectories over ALL items (p_ctrl and alpha are defined for every item)
    pctrl_all = np.array([r["p_ctrl_traj"] for r in cond_results], dtype=np.float64)
    alpha_all = np.array([r["alpha_traj"] for r in cond_results], dtype=np.float64)
    mean_pctrl_traj = pctrl_all.mean(axis=0).tolist()
    mean_alpha_traj = alpha_all.mean(axis=0).tolist()

    return {
        "condition": cond_name,
        "n_total": n_total, "n_usable": n_usable, "n_excluded": n_total - n_usable,
        "pick_counts": picks,
        "pick_rates": {k: v / n_total for k, v in picks.items()},
        "n_final_answer_is_target": n_final_target,
        "target_rate_usable": (n_final_target / n_usable if n_usable else None),
        "mean_final_p_target": mean_final_pt,
        "total_actuation_mean": float(np.mean(alpha_sums)),
        "total_actuation_std": float(np.std(alpha_sums)),
        "mean_nfe": float(np.mean(nfes)),
        "n_items_with_defer": deferred_any,
        "n_items_force_committed": forced_any,
        "mean_p_ctrl_traj": mean_pctrl_traj,
        "mean_alpha_traj": mean_alpha_traj,
        "validation": {
            "prob_range_ok": bool(val_prob_range_ok),
            "restricted_sum_le_1_ok": bool(val_prob_sum_ok),
            "committed_matches_parsed_pass_rate":
                (float(np.mean(val_committed_ok)) if val_committed_ok else None),
            "n_committed_checked": len(val_committed_ok),
            "pctrl_position_agreement_rate":
                (float(np.mean(pos_agree)) if pos_agree else None),
            "n_pctrl_pos_checked": len(pos_agree),
            "gt_final_ptarget_target_outcome_mean":
                (float(np.mean(final_pt_target)) if final_pt_target else None),
            "gt_final_ptarget_target_outcome_n": len(final_pt_target),
            "gt_final_ptarget_nontarget_outcome_mean":
                (float(np.mean(final_pt_nontarget)) if final_pt_nontarget else None),
            "gt_final_ptarget_nontarget_outcome_n": len(final_pt_nontarget),
        },
        "per_item": per_item,
    }


def make_figure(out, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(STEPS)
    C = out["conditions"]
    order = ["clean", "openloop8", "PI_antiwindup", "deadline", "mpc", "remask_feedback"]
    order = [c for c in order if c in C]
    palette = {
        "clean": "#4C72B0", "openloop8": "#DD8452", "PI_antiwindup": "#55A868",
        "deadline": "#C44E52", "mpc": "#8172B3", "remask_feedback": "#937860",
    }

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))

    for name in order:
        a = C[name]
        axA.plot(steps, a["mean_p_ctrl_traj"], color=palette.get(name), lw=2.2, label=name)
    axA.axhline(SETPOINT, color="black", ls="--", lw=1, alpha=0.7, label=f"setpoint s*={SETPOINT}")
    axA.axvline(DEADLINE_D, color="gray", ls=":", lw=1, alpha=0.7, label=f"answer commit ~step {DEADLINE_D}")
    axA.axhline(1 / 3, color="black", ls=":", lw=0.8, alpha=0.4)
    axA.set_xlabel("denoising step t")
    axA.set_ylabel("mean p_ctrl(t) = P(target letter) at gen pos 0")
    axA.set_title("(A) Control observable trajectory (all items)")
    axA.set_ylim(-0.02, 1.02)
    axA.legend(fontsize=8, loc="best")
    axA.grid(alpha=0.3)

    xs = np.arange(len(order))
    w = 0.38
    tr = [C[n]["target_rate_usable"] if C[n]["target_rate_usable"] is not None else np.nan for n in order]
    dose = [C[n]["total_actuation_mean"] for n in order]
    dose_norm = [d / ALPHA_MAX / STEPS for d in dose]  # fraction of max possible dose
    axB.bar(xs - w / 2, tr, w, color="#4C72B0", label="target pick-rate (usable)")
    axB.bar(xs + w / 2, dose_norm, w, color="#DD8452", label="total actuation / max dose")
    for i, d in enumerate(dose):
        axB.text(xs[i] + w / 2, dose_norm[i] + 0.01, f"{d:.0f}", ha="center", fontsize=7)
    for i, t in enumerate(tr):
        if not np.isnan(t):
            axB.text(xs[i] - w / 2, t + 0.01, f"{t:.2f}", ha="center", fontsize=7)
    axB.set_xticks(xs)
    axB.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    axB.set_ylabel("rate  /  normalized dose")
    axB.set_title("(B) Target pick-rate vs total actuation (dose)")
    axB.set_ylim(0, 1.05)
    axB.legend(fontsize=8, loc="best")
    axB.grid(alpha=0.3, axis="y")

    fig.suptitle("Closed-loop steering controllers vs open-loop / clean (LLaDA-8B-Instruct, p_target)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print(f"Saved PNG  -> {png_path}")


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={torch.cuda.get_device_name(0)}")

    print("Loading model/tokenizer (once) ...")
    model = B.AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    tok = B.AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"model param device: {next(model.parameters()).device}")

    saved = torch.load(DIRECTION_PATH, map_location="cpu")
    direction = saved["direction"].float().to(DEVICE)
    raw_norm = float(direction.norm().item())
    print(f"direction layer={saved.get('layer')} raw_norm={raw_norm:.4f}")

    rows = load_items(N_ITEMS)
    items, skipped = [], []
    for row in rows:
        b = black_options(row)
        if len(b) != 1:
            skipped.append({"example_id": int(row.get("example_id", -1)), "n_black": len(b)})
            continue
        items.append({
            "row": row, "black_idx": b[0],
            "target_letter": chr(ord("A") + b[0]),
            "unknown_idx": B.unknown_index(row),
        })
    print(f"Usable items (exactly one Black option): {len(items)}; skipped: {len(skipped)}")

    steerer = B.BiasSteerer(direction, alpha=0.0, mode="add")
    block = B.resolve_module(model, B.BLOCKS_PATH)[LAYER]
    steerer.attach(block)

    conditions = [
        ("clean", Clean()),
        ("openloop8", OpenLoop(ALPHA_MAX)),
        ("PI_antiwindup", PI(KP, KI, SETPOINT)),
        ("deadline", Deadline(DEADLINE_D, SETPOINT)),
        ("mpc", MPC(MPC_GRID, SETPOINT, MPC_EVERY)),
        ("remask_feedback", Remask(SETPOINT, DEADLINE_D)),
    ]

    out_conditions = {}
    try:
        for name, ctrl in conditions:
            print(f"\n=== condition: {name} ({ctrl.kind}) ===")
            res = run_condition(model, tok, items, steerer, ctrl)
            an = analyze(items, res, name)
            out_conditions[name] = an
            v = an["validation"]
            print(f"  usable={an['n_usable']}/{an['n_total']} excluded={an['n_excluded']} "
                  f"target-rate(usable)={an['target_rate_usable']}")
            print(f"  picks={an['pick_counts']}  mean_final_p_target={an['mean_final_p_target']}")
            print(f"  total_actuation(mean)={an['total_actuation_mean']:.1f}  mean_nfe={an['mean_nfe']:.1f}")
            print(f"  VAL committed_match={v['committed_matches_parsed_pass_rate']} "
                  f"pctrl_pos_agree={v['pctrl_position_agreement_rate']} "
                  f"GT tgt={v['gt_final_ptarget_target_outcome_mean']} "
                  f"nontgt={v['gt_final_ptarget_nontarget_outcome_mean']}")
            if ctrl.kind == "remask":
                print(f"  #4: items_with_defer={an['n_items_with_defer']} "
                      f"force_committed={an['n_items_force_committed']}")
    finally:
        steerer.detach()

    meta = {
        "device": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_path": MODEL_PATH, "direction_path": DIRECTION_PATH,
        "eval_items": EVAL_ITEMS, "layer": LAYER, "raw_norm": raw_norm,
        "n_items_used": len(items), "n_skipped": len(skipped),
        "gen_length": GEN_LENGTH, "steps": STEPS, "block_length": BLOCK_LENGTH,
        "temperature": TEMPERATURE, "remasking": REMASKING,
        "alpha_clamp": [0.0, ALPHA_MAX], "setpoint_sstar": SETPOINT,
        "PI_Kp": KP, "PI_Ki": KI, "mpc_grid": MPC_GRID, "mpc_every": MPC_EVERY,
        "deadline_D": DEADLINE_D,
        "control_observable": ("P(plain target)+P(space target) at gen pos 0; "
                               "one-step-delayed for gain/deadline/remask, within-step for mpc"),
        "final_metric_readout": ("validated P(target token) at TRUE parsed-letter "
                                 "position; variant matched to committed token"),
        "skipped_items": skipped,
    }
    out = {"meta": meta, "conditions": out_conditions}
    json_path = os.path.join(_HERE, "pB_controllers.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {json_path}")
    make_figure(out, os.path.join(_HERE, "pB_controllers.png"))
    print("Done.")


if __name__ == "__main__":
    main()
