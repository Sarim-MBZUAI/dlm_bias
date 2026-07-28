#!/usr/bin/env python
"""dream/common_dream.py -- SHARED INFRA for the Dream-v0-Instruct-7B port.

Dream analogue of baselines/common.py: model load, hook tuple-contract factories,
a BBQ generation wrapper, and the per-item eval loop -- so every Dream steering /
baseline script imports IDENTICAL infra and writes result files in the SAME schema
as the LLaDA stack (steering/pid_steer.py, baselines/common.py).

Harness-import convention (SAME as steering/denoise_pid.py:58-60): ROOT points at
the MAIN tree; we sys.path.insert eval/ + steering/ and reuse the pure-python BBQ
helpers (build_prompt / parse_letter / get_answer_info / unknown_index / LETTERS /
resolve_module / hidden_from_output / output_with_hidden) plus pid_steer's
black_idx_of / unk_idx_of / BLACK_TAGS verbatim.

DEVIATIONS from the LLaDA common.py (see README):
  * Model load is AutoModel(trust_remote_code) as before, but the sampler is
    Dream's OWN model.diffusion_generate (generation_utils.py) -- NOT the copied
    LLaDA block-diffusion generate(). Full bidirectional attention, alg="entropy".
  * Block list path is model.model.layers (28 DreamDecoderLayer), each returning a
    TUPLE (hidden,)+... -- handled by the shared hidden_from_output/output_with_hidden.
  * The Dream analogues of LLaDA attn_out / ff_out (pre-residual deltas) are the
    Linear INPUTS o_proj (28x128 concat heads) and down_proj (18944-d gated MLP);
    steer them with the *_pre_hook factories.
  * Any custom per-position LOGITS read must apply Dream's next-token shift
    (generation_utils.py:412); this module only reads hidden states + uses
    diffusion_generate, so no shift is needed here.

CLI:
    python -m dream.common_dream --selftest        # offline hook-math check, no GPU
    python dream/common_dream.py --smoke [--smoke-items N]   # 2-item GPU smoke
"""
import argparse
import json
import os
import sys
import time

import torch

# --------------------------------------------------------------------------- #
# Harness import (ROOT = MAIN tree on purpose -- model/data/arrows live there). #
# --------------------------------------------------------------------------- #
ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

from bbq_eval import (  # noqa: E402,F401
    build_prompt,
    parse_letter,
    get_answer_info,
    unknown_index,
    resolve_module,
    hidden_from_output,
    output_with_hidden,
    LETTERS,
)
from pid_steer import (  # noqa: E402,F401
    black_idx_of, unk_idx_of, BLACK_TAGS, unit_rows, AddVec,
)

# --------------------------------------------------------------------------- #
# Dream-v0-Instruct-7B facts (verified from local model code).                 #
# --------------------------------------------------------------------------- #
MODEL_DIR = os.path.join(ROOT, "Dream-v0-Instruct-7B")
SWEEP400 = os.path.join(ROOT, "data", "bbq_items", "_sweep400.jsonl")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

N_LAYERS = 28          # config num_hidden_layers
D_MODEL = 3584         # hidden_size
D_MLP = 18944          # intermediate_size (gated MLP width; down_proj input)
N_HEADS = 28           # attention heads
HEAD_DIM = 128         # head_dim (N_HEADS*HEAD_DIM = 3584 = o_proj input)
MASK_ID = 151666

BLOCKS_PATH = "model.layers"   # AutoModel returns DreamModel; DreamModel.model.layers
                               # is the 28-DreamDecoderLayer ModuleList (verified via
                               # named_modules on GPU; NOT model.model.layers).

# Generation defaults for BBQ eval (short letter answer, greedy).
GEN_DEFAULTS = dict(max_new_tokens=32, steps=64, temperature=0.0,
                    alg="entropy", alg_temp=0.0)


# --------------------------------------------------------------------------- #
# Module-path helpers.                                                         #
# --------------------------------------------------------------------------- #
def block_path(k):
    """DreamDecoderLayer k: output is the block residual (TUPLE contract)."""
    return f"{BLOCKS_PATH}.{int(k)}"


def o_proj_path(k):
    """attn out-proj INPUT = 3584-d concat-heads activation (LLaDA attn_out analogue)."""
    return f"{BLOCKS_PATH}.{int(k)}.self_attn.o_proj"


def down_proj_path(k):
    """MLP down-proj INPUT = 18944-d gated activation (LLaDA ff_out analogue)."""
    return f"{BLOCKS_PATH}.{int(k)}.mlp.down_proj"


# --------------------------------------------------------------------------- #
# Fire counter (shared by the factory hooks).  run_items resets it before the  #
# loop and asserts >0 after so a hook that never attaches is caught.           #
# --------------------------------------------------------------------------- #
_FIRE = {"n": 0}


def reset_fire_count():
    _FIRE["n"] = 0


def get_fire_count():
    return _FIRE["n"]


def _bump():
    _FIRE["n"] += 1


# --------------------------------------------------------------------------- #
# Tuple-contract hook factories (mirror baselines/common.py:116-215).          #
#                                                                              #
# A DreamDecoderLayer.forward returns (hidden,)+...; the Linear submodules      #
# (o_proj / down_proj) return a bare tensor. hidden_from_output /              #
# output_with_hidden make the edit type-agnostic so the SAME factory works at   #
# block level (forward hook) and at Linear-input level (forward_pre hook).      #
# --------------------------------------------------------------------------- #
def make_edit_hook(edit):
    """Wrap edit(hidden)->hidden into a forward hook that edits the module OUTPUT."""
    def hook(module, inp, out):
        _bump()
        h = hidden_from_output(out)
        return output_with_hidden(out, edit(h))
    return hook


def _to(t, ref):
    return t.to(ref.dtype).to(ref.device)


def add_vec_hook(vec):
    """Additive residual edit on a block OUTPUT: h <- h + vec (broadcasts B,seq)."""
    return make_edit_hook(lambda h: h + _to(vec, h))


def mul_vec_hook(gate):
    """Multiplicative per-neuron edit on a block OUTPUT: h <- h * gate."""
    return make_edit_hook(lambda h: h * _to(gate, h))


def affine_hook(beta, bias):
    """Affine per-neuron edit on a block OUTPUT: h <- beta*h + bias."""
    return make_edit_hook(lambda h: _to(beta, h) * h + _to(bias, h))


# ---- forward_PRE_hook variants (edit a Linear's INPUT) --------------------- #
# The 18944-d gated MLP activation and the 3584-d concat-heads attention
# activation are computed inline; the module whose OUTPUT is the residual delta
# takes them as INPUT (down_proj / o_proj). So steer the INPUT via a pre-hook.
def make_pre_edit_hook(edit):
    """Wrap edit(x)->x into a forward_pre_hook (edits positional input[0])."""
    def pre_hook(module, args):
        _bump()
        x = args[0]
        return (edit(x),) + tuple(args[1:])
    return pre_hook


def add_vec_pre_hook(vec):
    return make_pre_edit_hook(lambda x: x + _to(vec, x))


def mul_vec_pre_hook(gate):
    return make_pre_edit_hook(lambda x: x * _to(gate, x))


def affine_pre_hook(beta, bias):
    return make_pre_edit_hook(lambda x: _to(beta, x) * x + _to(bias, x))


def attach(model, module_paths, hook_fn, pre=False):
    """Register hook_fn on every module in module_paths; return the handles.
    Items may be dotted strings (resolve_module) OR already-resolved nn.Modules."""
    handles = []
    for p in module_paths:
        mod = resolve_module(model, p) if isinstance(p, str) else p
        if pre:
            handles.append(mod.register_forward_pre_hook(hook_fn))
        else:
            handles.append(mod.register_forward_hook(hook_fn))
    return handles


# --------------------------------------------------------------------------- #
# Model load -- AutoModel(trust_remote_code), NOT AutoModelForCausalLM.        #
# --------------------------------------------------------------------------- #
def load_model_tok(model_dir=MODEL_DIR, device="cuda"):
    """Load Dream-v0-Instruct-7B + tokenizer (bf16, cuda, eval). NEEDS A GPU."""
    from transformers import AutoModel, AutoTokenizer
    model = (AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                       torch_dtype=torch.bfloat16).to(device).eval())
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return model, tok


# --------------------------------------------------------------------------- #
# BBQ generation wrapper -- Dream's OWN diffusion sampler.                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, input_ids, attention_mask=None, **overrides):
    """Run model.diffusion_generate with GEN_DEFAULTS and return ONLY generated ids.

    The returned sequence from diffusion_generate INCLUDES the prompt (see
    chat.py:107-110); we slice it off so callers get the completion only."""
    cfg = {**GEN_DEFAULTS, **overrides}
    out = model.diffusion_generate(
        input_ids, attention_mask=attention_mask,
        output_history=False, return_dict_in_generate=True, **cfg)
    return out.sequences[:, input_ids.shape[1]:]


def _chat_ids(tok, prompt, device):
    """BBQ prompt -> chat-templated ids (ChatML, add_generation_prompt)."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", add_generation_prompt=True)
    return ids.to(device)


# --------------------------------------------------------------------------- #
# The item loop.  NEEDS A GPU.  Mirrors baselines/common.py run_baseline +      #
# pid_steer.py:294-307 classification, writing the IDENTICAL result schema.     #
# --------------------------------------------------------------------------- #
def run_items(
    attach_fn=None,
    items_path=SWEEP400,
    out_dir=None,
    tag="baseline",
    limit=0,
    model=None,
    tok=None,
    baseline_black_rate=None,
    config_extra=None,
    gen_overrides=None,
):
    """Run one BBQ condition end-to-end and write cond_<tag>.json + _samples.jsonl.

    attach_fn(model) -> list of hook handles (build hooks with the factories above
    so the shared fire counter increments); run_items resets it, asserts >0 when an
    attach_fn is given, and ALWAYS detaches in a finally block. attach_fn=None runs
    the clean (unsteered) baseline.
    """
    if out_dir is None:
        out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    if model is None or tok is None:
        model, tok = load_model_tok()

    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    use_rows = rows[:limit] if limit else rows

    reset_fire_count()
    handles = attach_fn(model) if attach_fn is not None else []
    if not isinstance(handles, (list, tuple)):
        handles = [handles]

    per_item = []
    counts = {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    dis_n = dis_correct = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(use_rows):
            prompt = build_prompt(row)
            ids = _chat_ids(tok, prompt, model.device)
            out = generate(model, ids, **(gen_overrides or {}))
            gen = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
            letter = parse_letter(gen, row)
            pred = LETTERS.index(letter) if letter else None
            bidx = black_idx_of(row)
            uidx = unk_idx_of(row)
            if pred is None:
                cls = "unparseable"
            elif pred == bidx:
                cls = "black"
            elif pred == uidx:
                cls = "abstain"
            else:
                cls = "nonblack"
            counts[cls] += 1
            if row.get("context_condition") == "disambig" and pred is not None:
                dis_n += 1
                dis_correct += int(pred == int(row["label"]))
            per_item.append({
                "example_id": int(row.get("example_id", -1)),
                "question_index": str(row.get("question_index", "")),
                "category": row.get("category"),
                "question_polarity": row.get("question_polarity"),
                "context_condition": row.get("context_condition"),
                "context": row.get("context"),
                "question": row.get("question"),
                "options": {"A": row.get("ans0"), "B": row.get("ans1"), "C": row.get("ans2")},
                "ground_truth": row.get("label"),
                "ground_truth_text": ([row.get("ans0"), row.get("ans1"), row.get("ans2")][row["label"]]
                                      if isinstance(row.get("label"), int) and 0 <= row["label"] < 3 else None),
                "prompt": prompt,
                "black_idx": bidx, "unk_idx": uidx, "pred_index": pred,
                "pred_letter": letter, "model_output": gen, "pred_class": cls,
            })
            if (idx + 1) % 50 == 0:
                print(f"[{tag}] {idx+1}/{len(use_rows)} b={counts['black']} "
                      f"nb={counts['nonblack']} ab={counts['abstain']} "
                      f"un={counts['unparseable']} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for h in handles:
            h.remove()

    n_fired = get_fire_count()
    if attach_fn is not None:
        assert n_fired > 0, (
            f"[{tag}] hooks never fired (fire_count=0): attach_fn attached nothing "
            f"or its hooks were not built with the common_dream.py factories.")

    n = len(use_rows)
    rates = {
        "black_pick_rate": counts["black"] / n,
        "nonblack_pick_rate": counts["nonblack"] / n,
        "abstain_rate": counts["abstain"] / n,
        "unparseable_rate": counts["unparseable"] / n,
    }
    d_gap = (rates["black_pick_rate"] - baseline_black_rate
             if baseline_black_rate is not None else None)
    acc_disambig = (dis_correct / dis_n) if dis_n else None
    result = {
        "condition": tag, "model": "Dream-v0-Instruct-7B", "n": n,
        "counts": counts, "rates": rates, "d_gap": d_gap,
        "baseline_black_rate": baseline_black_rate,
        "acc_disambig": acc_disambig, "n_disambig": dis_n,
        "gen_defaults": {**GEN_DEFAULTS, **(gen_overrides or {})},
        "items_path": items_path, "hook_fire_count": n_fired,
        "elapsed_s": time.time() - t0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config": config_extra or {},
    }
    outp = os.path.join(out_dir, f"cond_{tag}.json")
    with open(outp, "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, f"cond_{tag}_samples.jsonl"), "w") as f:
        for it in per_item:
            f.write(json.dumps(it) + "\n")
    print(f"[{tag}] DONE rates={rates} d_gap={d_gap} acc_dis={acc_disambig} "
          f"fire_count={n_fired} -> {outp}", flush=True)
    return result


# --------------------------------------------------------------------------- #
# Offline self-test: hook math + Dream tuple contract on synthetic tensors.    #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    B, S, H = 2, 4, 6
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-common] {name:46s} : {'PASS' if cond else 'FAIL'}")

    h = torch.randn(B, S, H)
    vec = torch.randn(H)
    gate = torch.rand(H) + 0.1
    beta = torch.randn(H)
    bias = torch.randn(H)

    # --- forward-hook (output edit) on a DreamDecoderLayer TUPLE output ---
    reset_fire_count()
    tup_out = (h.clone(), "kv_cache_sentinel")
    r = add_vec_hook(vec)(None, None, tup_out)
    check("add_vec_hook tuple: structure preserved",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "kv_cache_sentinel")
    check("add_vec_hook tuple: h + vec", torch.allclose(r[0], h + vec))
    r = mul_vec_hook(gate)(None, None, tup_out)
    check("mul_vec_hook tuple: h * gate", torch.allclose(r[0], h * gate))
    r = affine_hook(beta, bias)(None, None, tup_out)
    check("affine_hook tuple: beta*h + bias", torch.allclose(r[0], beta * h + bias))

    # --- forward-hook, BARE tensor (o_proj/down_proj output type) ---
    r = add_vec_hook(vec)(None, None, h.clone())
    check("add_vec_hook bare tensor: h + vec",
          torch.is_tensor(r) and torch.allclose(r, h + vec))

    # --- forward_pre_hook (Linear input edit), extras preserved ---
    args = (h.clone(), torch.zeros(1))
    r = add_vec_pre_hook(vec)(None, args)
    check("add_vec_pre_hook: input[0]+vec, extras kept",
          isinstance(r, tuple) and len(r) == 2 and torch.allclose(r[0], h + vec)
          and torch.allclose(r[1], args[1]))
    r = affine_pre_hook(beta, bias)(None, (h.clone(),))
    check("affine_pre_hook: beta*x + bias", torch.allclose(r[0], beta * h + bias))
    r = mul_vec_pre_hook(gate)(None, (h.clone(),))
    check("mul_vec_pre_hook: x * gate", torch.allclose(r[0], h * gate))

    # --- fire counter increments once per hook invocation ---
    reset_fire_count()
    hk = add_vec_hook(vec)
    for _ in range(5):
        hk(None, None, h.clone())
    check("fire counter counts invocations", get_fire_count() == 5)

    print(f"[selftest-common] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def _smoke(n_items):
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"[smoke] CVD={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"dev={torch.cuda.get_device_name(0)}", flush=True)
    res = run_items(attach_fn=None, tag="smoke", limit=n_items)
    print(f"[smoke] DONE counts={res['counts']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline hook-math check on synthetic tensors (no GPU)")
    ap.add_argument("--smoke", action="store_true",
                    help="GPU smoke: run_items on a few clean items")
    ap.add_argument("--smoke-items", type=int, default=2)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if args.smoke:
        _smoke(args.smoke_items)
        return
    ap.error("nothing to do: pass --selftest (offline) or --smoke (GPU)")


if __name__ == "__main__":
    main()
