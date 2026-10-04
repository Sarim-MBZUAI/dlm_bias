#!/usr/bin/env python
"""Shared driver for the activation-steering baselines.

Port of the Activation Transport (AcT; Rodriguez et al., 2025) family of
steering baselines onto the LLaDA-8B-Instruct masked-diffusion BBQ
bias-injection harness. All method files import from here, so model loading,
the per-item generation loop, the hook tuple-contract, scoring and the result
file format are identical across methods and identical to steering/pid_steer.py.

ROOT (env DLM_BIAS_ROOT, default: repo root) holds the model, arrows.pt and
data; eval/ and steering/ are put on sys.path to import bbq_eval and pid_steer.

Injection semantics: positive strength pushes the model toward the target
("Black") answer. When a method fits an Optimal-Transport map, the target is
the OT destination (mu2 / dst) and the non-target option is the source
(mu1 / src).

Intervention position is always all tokens, bidirectional (the hook fires
once per denoising step over every position). There is no last-token mode.

Usage:
    python baselines/common.py --selftest   # offline hook-math check, no GPU
"""
import argparse
import json
import os
import sys
import time

import torch

# Harness import: ROOT holds model/arrows/data (same convention as pid_steer.py).
ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
CACHE_DIR = (os.environ.get("DLM_BASELINE_CACHE_DIR")
             or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))
# Code tree (this checkout), distinct from ROOT (the data/model tree): sibling
# code modules like multirace/targets.py come from this checkout, while data
# reads honor DLM_BIAS_ROOT.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

H_MODEL = 4096       # residual width (config.json d_model)
H_MLP = 12288        # gated MLP-hidden width (config.json mlp_hidden_size)
N_HEADS = 32         # config.json n_heads
D_HEAD = 128         # d_model / n_heads


# Target parameterization. target="black" is the default; other targets use
# the multirace manifest heldout entries, multirace/arrows_<target>.pt, cache
# artifacts *_<target>.pt, and multirace/targets.py's target_idx_of for eval
# classification. A new target needs a manifest entry and an arrows_<target>.pt.
SUPPORTED_TARGETS = ("black", "woman", "man", "arab", "lowses", "old")


def target_suffix(target):
    """Cache-artifact filename suffix: '' for the black default, '_<target>'
    otherwise."""
    assert target in SUPPORTED_TARGETS, \
        f"target must be one of {SUPPORTED_TARGETS}, got {target!r}"
    return "" if target == "black" else f"_{target}"


def sweep400_path(target):
    """Default eval items file for a target (the unrotated 400-item sweep)."""
    if target == "black":
        return SWEEP400
    target_suffix(target)  # validate
    return os.path.join(ROOT, "data", "bbq_items", f"_sweep400_{target}.jsonl")


def result_keys(target):
    """Result/sample key names per target. Black uses black/nonblack/black_idx;
    other targets use the multirace naming (target/nontarget/target_idx)."""
    if target == "black":
        return {"pick": "black", "non": "nonblack", "idx": "black_idx"}
    target_suffix(target)  # validate
    return {"pick": "target", "non": "nontarget", "idx": "target_idx"}


_MULTIRACE_TARGETS_MOD = None


def _multirace_targets():
    """Load multirace/targets.py from this code tree by absolute path (avoids
    putting a module named 'targets' on sys.path)."""
    global _MULTIRACE_TARGETS_MOD
    if _MULTIRACE_TARGETS_MOD is None:
        import importlib.util
        path = os.path.join(CODE_ROOT, "multirace", "targets.py")
        spec = importlib.util.spec_from_file_location("multirace_targets", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MULTIRACE_TARGETS_MOD = mod
    return _MULTIRACE_TARGETS_MOD


def target_index_fn(target):
    """row -> option index of the steered target. Black uses
    pid_steer.black_idx_of; other targets use multirace/targets.py
    (target_idx_of)."""
    if target == "black":
        return black_idx_of
    target_suffix(target)  # validate
    mt = _multirace_targets()
    return lambda row: mt.target_idx_of(row, target)


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
    (act/hooks/intervention_hook.py).  The LLaDA block tuple is kept intact
    via bbq_eval.output_with_hidden."""
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
    (act/hooks/aura_hook.py with strength=1)."""
    return make_edit_hook(lambda h: h * _to(gate, h))


def affine_hook(beta, bias):
    """Affine (per-neuron) edit: h <- beta*h + bias.  Used by the OT maps
    (act/hooks/transport.py std1_2*(x-mu1)+mu2 == beta*x+bias with
    beta=std2/std1, bias=mu2-beta*mu1; and the empirical LinearProj x*w1+b1,
    act/optimal_transport/archs.py)."""
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
# Model load -- identical to eval/bbq_eval.py main.                            #
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
    items_path=None,
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
    target="black",
):
    """Run one steered BBQ condition end-to-end and write the result files.

    attach_fn(model) -> list of forward-hook handles.  Build the hooks inside
    attach_fn with the factories above (add_vec_hook / affine_hook / ...) so the
    shared fire-counter is incremented; run_baseline resets it, asserts >0, and
    ALWAYS detaches the handles in a finally block.

    Scoring mirrors steering/pid_steer.py exactly:
        pred = LETTERS.index(parse_letter(gen, row)) or None
        black    : pred == black_idx_of(row)
        abstain  : pred == unk_idx_of(row)
        nonblack : parsed but neither
        unparseable : parse_letter returned None
    Writes  out_dir/cond_<tag>.json  and  out_dir/cond_<tag>_samples.jsonl
    in the pid_steer result format.

    baseline_black_rate: if given, d_gap = <target>_pick_rate - baseline rate
    is recorded (the injection lift over the clean model).

    target: "black" (default) uses black_idx_of classification and
    black/nonblack/black_idx key names.  A non-black target classifies against
    multirace/targets.py's target_idx_of and emits the multirace key names
    (target/nontarget/target_idx + a "target" field).
    items_path=None resolves to the target's own _sweep400 file.
    """
    if items_path is None:
        items_path = sweep400_path(target)
    if out_dir is None:
        out_dir = CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    keys = result_keys(target)
    tidx_of = target_index_fn(target)

    if model is None or tok is None:
        model, tok = load_model()

    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    use_rows = rows[:limit] if limit else rows

    reset_fire_count()
    handles = attach_fn(model)
    if not isinstance(handles, (list, tuple)):
        handles = [handles]

    per_item = []
    counts = {keys["pick"]: 0, keys["non"]: 0, "abstain": 0, "unparseable": 0}
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
            bidx = tidx_of(row)
            uidx = unk_idx_of(row)
            if pred is None:
                cls = "unparseable"; counts["unparseable"] += 1
            elif pred == bidx:
                cls = keys["pick"]; counts[keys["pick"]] += 1
            elif pred == uidx:
                cls = "abstain"; counts["abstain"] += 1
            else:
                cls = keys["non"]; counts[keys["non"]] += 1
            if row.get("context_condition") == "disambig" and pred is not None:
                dis_n += 1
                dis_correct += int(pred == int(row["label"]))
            item = {
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
                keys["idx"]: bidx, "unk_idx": uidx, "pred_index": pred,
                "pred_letter": letter, "model_output": gen, "pred_class": cls,
            }
            if target != "black":
                item["target"] = target
            per_item.append(item)
            if (idx + 1) % 50 == 0:
                print(f"[{tag}] {idx+1}/{len(use_rows)} b={counts[keys['pick']]} "
                      f"nb={counts[keys['non']]} ab={counts['abstain']} "
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
        f"{keys['pick']}_pick_rate": counts[keys["pick"]] / n,
        f"{keys['non']}_pick_rate": counts[keys["non"]] / n,
        "abstain_rate": counts["abstain"] / n,
        "unparseable_rate": counts["unparseable"] / n,
    }
    d_gap = (rates[f"{keys['pick']}_pick_rate"] - baseline_black_rate
             if baseline_black_rate is not None else None)
    acc_disambig = (dis_correct / dis_n) if dis_n else None
    result = {
        "condition": tag,
        "n": n,
        "counts": counts,
        "rates": rates,
        "d_gap": d_gap,
        f"baseline_{keys['pick']}_rate": baseline_black_rate,
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
    if target != "black":
        result["target"] = target
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

    # --- target parameterization (offline, no GPU) ---
    check("target_suffix: black -> '' (default paths)",
          target_suffix("black") == "")
    check("target_suffix: woman/man -> '_<t>'",
          target_suffix("woman") == "_woman" and target_suffix("man") == "_man")
    try:
        target_suffix("klingon")
        bad = False
    except AssertionError:
        bad = True
    check("target_suffix rejects unknown target", bad)
    check("sweep400_path black == SWEEP400 (identical object path)",
          sweep400_path("black") == SWEEP400)
    check("sweep400_path woman -> _sweep400_woman.jsonl",
          sweep400_path("woman").endswith("data/bbq_items/_sweep400_woman.jsonl"))
    check("result_keys black == default names",
          result_keys("black") == {"pick": "black", "non": "nonblack",
                                   "idx": "black_idx"})
    check("result_keys gender == multirace names (strict_pool target_idx)",
          result_keys("man") == {"pick": "target", "non": "nontarget",
                                 "idx": "target_idx"})
    # classification dispatch: black uses pid_steer.black_idx_of UNCHANGED;
    # gender uses the multirace registry (whole-tag, strict sets).
    check("target_index_fn('black') IS black_idx_of", target_index_fn("black") is black_idx_of)
    row_b = {"answer_info": {"ans0": ["x", "Black"], "ans1": ["y", "White"],
                             "ans2": ["z", "unknown"]}}
    row_g = {"answer_info": {"ans0": ["x", "F"], "ans1": ["y", "M"],
                             "ans2": ["z", "unknown"]}}
    check("black fn: Black row -> 0", target_index_fn("black")(row_b) == 0)
    check("woman fn: F row -> 0, man fn -> 1",
          target_index_fn("woman")(row_g) == 0 and target_index_fn("man")(row_g) == 1)
    check("woman fn on race row -> None (no gender tag)",
          target_index_fn("woman")(row_b) is None)
    check("man fn: trans_m excluded (strict registry)",
          target_index_fn("man")({"answer_info": {"ans0": ["x", "trans_m"],
                                                  "ans1": ["y", "F"],
                                                  "ans2": ["z", "unknown"]}}) is None)

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
