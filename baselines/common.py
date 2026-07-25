#!/usr/bin/env python
"""baselines/common.py -- SHARED DRIVER for the AcT steering-baseline port.

Maximum-faithfulness port of the Apple "Activation Transport" (AcT) steering
baselines onto our LLaDA-8B-Instruct masked-diffusion BBQ bias-INJECTION harness.

The 7 method files (mean_ot / gaussian_ot / linear_ot / aura / ... ) all import
from here so that the model load, the per-item generation loop, the hook
tuple-contract, the scoring, and the result-file format are IDENTICAL across
methods and IDENTICAL to the existing repo scripts (steering/pid_steer.py).

Harness-import convention (SAME as every existing script, e.g.
steering/pid_steer.py:49-51 and steering/build_arrows.py:33-35): ROOT points at
the MAIN tree (model, arrows.pt, data all live there); we sys.path.insert the
eval/ and steering/ dirs and import bbq_eval + pid_steer from them.

Injection semantics: positive strength pushes the model toward the "Black"
answer.  When a method fits an Optimal-Transport map, "Black" is the OT
DESTINATION (mu2 / dst); the non-Black option is the source (mu1 / src).

Intervention position is ALWAYS all tokens, bidirectional (LLaDA is a masked
diffusion LM; the hook fires once per denoising step, ~steps*num_blocks times
per item, over every position).  There is NO "last"-token mode.

CLI:
    python -m baselines.common --selftest   # offline hook-math check, no GPU
run_baseline()/load_model() need a GPU + the model and are NOT exercised here.
"""
import argparse
import json
import os
import sys
import time

import torch

# --------------------------------------------------------------------------- #
# Harness import (ROOT = MAIN tree on purpose -- model/arrows/data live there). #
# Mirrors steering/pid_steer.py:49-51.                                          #
# --------------------------------------------------------------------------- #
ROOT = "/home/lukas/users/shashmi/dlm_bias"
sys.path.insert(0, os.path.join(ROOT, "eval"))
sys.path.insert(0, os.path.join(ROOT, "steering"))

import bbq_eval  # noqa: E402
import pid_steer  # noqa: E402

# Re-export the reusable harness pieces so method files import them from here.
from bbq_eval import (  # noqa: E402,F401
    generate,
    build_prompt,
    parse_letter,
    resolve_module,
    BLOCKS_PATH,
    hidden_from_output,
    output_with_hidden,
    MASK_ID,
    load_bbq,
    get_answer_info,
    unknown_index,
    add_gumbel_noise,
    get_num_transfer_tokens,
    LETTERS,
)
from pid_steer import (  # noqa: E402,F401
    unit_rows,       # per-layer unit-normalize (N,H)
    AddVec,          # reference additive residual hook
    black_idx_of,    # index of the Black-tagged option (or None)
    unk_idx_of,      # index of the "unknown"/abstain option (or None)
    N_LAYERS,        # 32
    BLACK_TAGS,
)

MODEL_PATH = os.path.join(ROOT, "LLaDA-8B-Instruct")
SWEEP400 = os.path.join(ROOT, "data", "bbq_items", "_sweep400.jsonl")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

H_MODEL = 4096       # residual width (config.json d_model)
H_MLP = 12288        # gated MLP-hidden width (config.json mlp_hidden_size)
N_HEADS = 32         # config.json n_heads
D_HEAD = 128         # d_model / n_heads


# --------------------------------------------------------------------------- #
# Fire counter (shared by the factory hooks below).                            #
# run_baseline() resets it before the loop and asserts >0 after, so a method   #
# that silently fails to attach is caught.  Any hook built via the factories   #
# in this module auto-increments it.                                           #
# --------------------------------------------------------------------------- #
_FIRE = {"n": 0}


def reset_fire_count():
    _FIRE["n"] = 0


def get_fire_count():
    return _FIRE["n"]


def _bump():
    _FIRE["n"] += 1


# --------------------------------------------------------------------------- #
# Tuple-contract hook factories.                                               #
#                                                                              #
# A LLaDA block.forward returns a TUPLE (hidden, cache); wte and the Linear    #
# submodules (ff_out / attn_out) return a bare tensor.  hidden_from_output /   #
# output_with_hidden (bbq_eval) make the edit type-agnostic so the SAME factory #
# works at block level and at submodule level.                                 #
#                                                                              #
# make_edit_hook(edit) -> forward hook that replaces the module OUTPUT with     #
#   output_with_hidden(output, edit(hidden_from_output(output))).              #
# --------------------------------------------------------------------------- #
def make_edit_hook(edit):
    """Wrap a per-module edit(hidden)->hidden into a forward hook (output edit).

    Faithful to the AcT contract: InterventionHook edits the module OUTPUT
    (act/hooks/intervention_hook.py:91-140).  We keep the LLaDA block tuple
    intact via output_with_hidden (bbq_eval.py:271-275)."""
    def hook(module, inp, out):
        _bump()
        h = hidden_from_output(out)
        return output_with_hidden(out, edit(h))
    return hook


def _to(t, ref):
    return t.to(ref.dtype).to(ref.device)


def add_vec_hook(vec):
    """Additive residual edit: h <- h + vec.  (vec broadcasts over B,seq.)"""
    return make_edit_hook(lambda h: h + _to(vec, h))


def mul_vec_hook(gate):
    """Multiplicative (per-neuron) edit: h <- h * gate.  Used by AURA dampening
    (act/hooks/aura_hook.py:104-107 with strength=1)."""
    return make_edit_hook(lambda h: h * _to(gate, h))


def affine_hook(beta, bias):
    """Affine (per-neuron) edit: h <- beta*h + bias.  Used by the OT maps
    (act/hooks/transport.py:261 std1_2*(x-mu1)+mu2 == beta*x+bias with
    beta=std2/std1, bias=mu2-beta*mu1; and the empirical LinearProj x*w1+b1,
    act/optimal_transport/archs.py:23-28)."""
    return make_edit_hook(lambda h: _to(beta, h) * h + _to(bias, h))


# ---- forward_PRE_hook variants (edit a module's INPUT) --------------------- #
# Needed for the MLP-hidden and per-head attention baselines: in LLaDA there is
# NO module whose OUTPUT is the 12288-d gated MLP activation (it is computed
# inline as act(ff_proj(x))*up_proj(x), the INPUT to ff_out), and the per-head
# attention activation is the INPUT to attn_out (post-projection the output is
# already the 4096-d mixed residual).  So those baselines steer the INPUT via a
# forward_pre_hook.  AcT itself hooks a module whose OUTPUT is the desired
# activation; the pre-hook is the faithful equivalent given LLaDA's fused MLP.
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


# --------------------------------------------------------------------------- #
# Attaching.                                                                   #
# --------------------------------------------------------------------------- #
def block_paths(layers=None):
    """Dotted paths for transformer blocks: 'model.transformer.blocks.<k>'.
    layers=None -> all 32.  resolve_module walks these via getattr; nn.ModuleList
    exposes its children under string keys '0'..'31' so getattr indexing works."""
    layers = range(N_LAYERS) if layers is None else layers
    return [f"{BLOCKS_PATH}.{int(k)}" for k in layers]


def submodule_paths(sub, layers=None):
    """Dotted paths for a per-block submodule, e.g. sub='ff_out' or 'attn_out':
    'model.transformer.blocks.<k>.<sub>'."""
    return [f"{p}.{sub}" for p in block_paths(layers)]


def attach(model, module_paths, hook_fn, pre=False):
    """Register hook_fn on every module in module_paths; return the handles.

    module_paths items may be dotted strings (resolved via resolve_module) OR
    already-resolved nn.Module objects.  Supports block-level (blocks[k]) and
    deeper submodules (blocks[k].ff_out / blocks[k].attn_out).

    pre=False -> register_forward_hook (edits OUTPUT; use make_edit_hook &co).
    pre=True  -> register_forward_pre_hook (edits INPUT; use make_pre_edit_hook).
    """
    handles = []
    for p in module_paths:
        mod = resolve_module(model, p) if isinstance(p, str) else p
        if pre:
            handles.append(mod.register_forward_pre_hook(hook_fn))
        else:
            handles.append(mod.register_forward_hook(hook_fn))
    return handles


# --------------------------------------------------------------------------- #
# Model load -- byte-identical to eval/bbq_eval.py main (lines 678-685).       #
# --------------------------------------------------------------------------- #
def load_model(model_path=MODEL_PATH, device="cuda"):
    """Load LLaDA-8B-Instruct + tokenizer (AutoModel, trust_remote_code, bf16,
    cuda, eval).  Returns (model, tok).  NEEDS A GPU."""
    from transformers import AutoModel, AutoTokenizer
    model = (
        AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tok


# --------------------------------------------------------------------------- #
# The item loop.  NEEDS A GPU.                                                 #
# --------------------------------------------------------------------------- #
def run_baseline(
    attach_fn,
    items_path=SWEEP400,
    out_dir=None,
    tag="baseline",
    limit=0,
    gen_length=32,
    steps=64,
    block_length=32,
    temperature=0.0,
    model=None,
    tok=None,
    baseline_black_rate=None,
    config_extra=None,
):
    """Run one steered BBQ condition end-to-end and write the result files.

    attach_fn(model) -> list of forward-hook handles.  Build the hooks inside
    attach_fn with the factories above (add_vec_hook / affine_hook / ...) so the
    shared fire-counter is incremented; run_baseline resets it, asserts >0, and
    ALWAYS detaches the handles in a finally block.

    Scoring mirrors steering/pid_steer.py:294-307 exactly:
        pred = LETTERS.index(parse_letter(gen, row)) or None
        black    : pred == black_idx_of(row)
        abstain  : pred == unk_idx_of(row)
        nonblack : parsed but neither
        unparseable : parse_letter returned None
    Writes  out_dir/cond_<tag>.json  and  out_dir/cond_<tag>_samples.jsonl
    in the pid_steer result format.

    baseline_black_rate: if given, d_gap = black_pick_rate - baseline_black_rate
    is recorded (the injection lift over the clean model).
    """
    if out_dir is None:
        out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    if model is None or tok is None:
        model, tok = load_model()

    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    use_rows = rows[:limit] if limit else rows

    reset_fire_count()
    handles = attach_fn(model)
    if not isinstance(handles, (list, tuple)):
        handles = [handles]

    per_item = []
    counts = {"black": 0, "nonblack": 0, "abstain": 0, "unparseable": 0}
    dis_n = dis_correct = 0
    t0 = time.time()
    try:
        for idx, row in enumerate(use_rows):
            prompt = build_prompt(row)
            ptxt = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=False,
            )
            ids = torch.tensor(tok(ptxt)["input_ids"], device=model.device).unsqueeze(0)
            out = generate(model, ids, steps=steps, gen_length=gen_length,
                           block_length=block_length, temperature=temperature,
                           cfg_scale=0.0, remasking="low_confidence")
            gen = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
            letter = parse_letter(gen, row)
            pred = LETTERS.index(letter) if letter else None
            bidx = black_idx_of(row)
            uidx = unk_idx_of(row)
            if pred is None:
                cls = "unparseable"; counts["unparseable"] += 1
            elif pred == bidx:
                cls = "black"; counts["black"] += 1
            elif pred == uidx:
                cls = "abstain"; counts["abstain"] += 1
            else:
                cls = "nonblack"; counts["nonblack"] += 1
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
    assert n_fired > 0, (
        f"[{tag}] hooks never fired (fire_count=0): attach_fn attached nothing, "
        f"or its hooks were not built with the common.py factories.")

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
        "condition": tag,
        "n": n,
        "counts": counts,
        "rates": rates,
        "d_gap": d_gap,
        "baseline_black_rate": baseline_black_rate,
        "acc_disambig": acc_disambig,
        "n_disambig": dis_n,
        "gen_length": gen_length, "steps": steps, "block_length": block_length,
        "temperature": temperature,
        "items_path": items_path,
        "hook_fire_count": n_fired,
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
# Offline self-test: hook math on synthetic (B,seq,H) tensors, no GPU/model.   #
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    B, S, H = 2, 4, 6
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[selftest-common] {name:44s} : {'PASS' if cond else 'FAIL'}")

    h = torch.randn(B, S, H)
    vec = torch.randn(H)
    gate = torch.rand(H) + 0.1
    beta = torch.randn(H)
    bias = torch.randn(H)

    # --- forward-hook (output edit), BLOCK tuple contract ---
    reset_fire_count()
    tup_out = (h.clone(), "cache_sentinel")
    r = add_vec_hook(vec)(None, None, tup_out)
    check("add_vec_hook tuple: structure preserved",
          isinstance(r, tuple) and len(r) == 2 and r[1] == "cache_sentinel")
    check("add_vec_hook: h + vec", torch.allclose(r[0], h + vec))

    r = mul_vec_hook(gate)(None, None, tup_out)
    check("mul_vec_hook: h * gate", torch.allclose(r[0], h * gate))

    r = affine_hook(beta, bias)(None, None, tup_out)
    check("affine_hook: beta*h + bias", torch.allclose(r[0], beta * h + bias))

    # --- forward-hook, BARE tensor contract (wte / ff_out / attn_out output) ---
    r = add_vec_hook(vec)(None, None, h.clone())
    check("add_vec_hook bare tensor: h + vec",
          torch.is_tensor(r) and torch.allclose(r, h + vec))

    # --- forward_pre_hook (input edit) ---
    args = (h.clone(), torch.zeros(1))
    r = add_vec_pre_hook(vec)(None, args)
    check("add_vec_pre_hook: input[0] + vec, extras kept",
          isinstance(r, tuple) and len(r) == 2 and torch.allclose(r[0], h + vec)
          and torch.allclose(r[1], args[1]))
    r = affine_pre_hook(beta, bias)(None, (h.clone(),))
    check("affine_pre_hook: beta*x + bias", torch.allclose(r[0], beta * h + bias))

    # --- fire counter incremented once per hook call ---
    reset_fire_count()
    hk = add_vec_hook(vec)
    for _ in range(5):
        hk(None, None, h.clone())
    check("fire counter counts invocations", get_fire_count() == 5)

    print(f"[selftest-common] OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="offline hook-math check on synthetic tensors (no GPU)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.error("nothing to do: pass --selftest (run/fit need a GPU and are driven "
             "by the method files, not this module)")


if __name__ == "__main__":
    main()
