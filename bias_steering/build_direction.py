#!/usr/bin/env python
"""Build social/demographic bias STEERING DIRECTIONS for LLaDA-8B-Instruct.

Training-free port of "Implicit Bias Injection" (IBI) to LLaDA's masked-diffusion
architecture. We compute INPUT-EMBEDDING-LAYER directions that point from the
anti-stereotype side toward the stereotype side of minimal pairs.

Two sources:
  * --source crows (default): download CrowS-Pairs, group by bias_type, and build
    ONE direction PER CATEGORY (+ a combined "all" direction). For each category
    we also report COHERENCE metrics (norm-ratio, split-half cosine).
  * --source json: the legacy single-direction behavior from a JSON probe file
    (default example_pairs.json -> direction.pt). Kept so old workflows still run.

Method (frozen model, no gradients):
  1. Register a forward hook on the token-embedding module (model.transformer.wte
     reached via model.model.transformer.wte for the AutoModel object). The hook
     only CAPTURES the embedding output (B, T, H) -- it does not modify it.
  2. For each (stereotype, anti_stereotype) pair, run a forward pass on each
     sentence, take the masked-mean of the captured embeddings over real tokens
     -> a single (H,) vector per sentence.
  3. direction = mean over pairs of (emb_mean[stereotype] - emb_mean[anti]).

Because LLaDA applies NO embedding scaling, NO additive positional embedding
(RoPE is applied inside attention), and embedding dropout is a no-op at p=0,
the captured wte output is exactly the true input embedding fed into block 0.
The saved norms let you calibrate alpha in bias_llada.py.

NOTE: run this on a GPU; this script invokes the model. The user runs it.
"""
import argparse
import csv
import io
import json
import os
import re
import urllib.request

import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_PATH = "/home/lukas/users/shashmi/dlm_bias/LLaDA-8B-Instruct"
DEFAULT_PAIRS = "/home/lukas/users/shashmi/dlm_bias/bias_steering/example_pairs.json"
DEFAULT_OUT = "/home/lukas/users/shashmi/dlm_bias/bias_steering/direction.pt"
DEFAULT_OUT_DIR = "/home/lukas/users/shashmi/dlm_bias/bias_steering/directions"
DEFAULT_HOOK_MODULE = "model.transformer.wte"
BLOCKS_PATH = "model.transformer.blocks"  # mid-residual-layer steering target
DEFAULT_LAYER = "emb"  # backward-compatible default (input-embedding layer)
DEFAULT_CROWS_URL = (
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
    "data/crows_pairs_anonymized.csv"
)
CROWS_CACHE = (
    "/home/lukas/users/shashmi/dlm_bias/bias_steering/.crows_cache/crows_pairs.csv"
)
PAD_TOKEN_ID = 126081
SPLITHALF_SEED = 1234


def resolve_module(model, dotted_path):
    """Resolve a dotted attribute path (e.g. 'model.transformer.wte') on model."""
    obj = model
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- #
# SHARED layer-spec machinery (imported by bias_llada.py / bbq_eval.py).
#
# A "layer spec" is either the string "emb" (input-embedding / wte behavior,
# backward compatible) or an int L = a 0-based transformer BLOCK index. The
# block.forward RETURNS a 2-tuple (hidden_state, cache); the wte forward returns
# a bare tensor. The capture/inject helpers below handle BOTH return shapes.
# --------------------------------------------------------------------------- #
def parse_layer_spec(s):
    """Parse a CLI --layer value -> 'emb' or an int block index."""
    if s is None:
        return DEFAULT_LAYER
    s = str(s).strip()
    if s.lower() == "emb":
        return "emb"
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"--layer must be 'emb' or an int block index, got {s!r}")


def resolve_layer_module(model, spec, hook_module=DEFAULT_HOOK_MODULE):
    """Return the module to hook for a layer spec.

    'emb' -> the wte embedding module (via the dotted hook_module path);
    int L -> model.model.transformer.blocks[L].
    """
    if spec == "emb":
        return resolve_module(model, hook_module)
    return resolve_module(model, BLOCKS_PATH)[int(spec)]


def hidden_from_output(output):
    """Extract the hidden-state tensor from a hook output (tensor OR block tuple)."""
    if isinstance(output, tuple):
        return output[0]  # block.forward -> (hidden, cache); hidden at index 0
    return output


def output_with_hidden(output, new_hidden):
    """Rebuild a hook output preserving its original type (tensor OR tuple)."""
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])  # keep cache element(s)
    return new_hidden


def safe_name(category):
    """Sanitize a bias_type token into a filesystem-safe stem: lower, non-alnum->'_'."""
    return re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_") or "unknown"


# --------------------------------------------------------------------------- #
# CrowS-Pairs loader (stdlib only).
# --------------------------------------------------------------------------- #
def load_crows(url, cache_path):
    """Download (or reuse cached) CrowS-Pairs CSV and return rows grouped by bias_type.

    Returns: dict[bias_type] -> list of (stereotype_sentence, antistereotype_sentence).

    Orientation rule (mirrors metric.py): by dataset convention sent_more is ALWAYS
    the more-stereotypical sentence and sent_less the less-stereotypical one, so
    stereotype_sentence = sent_more and antistereotype_sentence = sent_less in ALL
    cases. The stereo_antistereo field only flips the scoring comparison (handled in
    eval), not the more/less mapping, so we keep the universal assignment here.
    """
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            with open(cache_path, "wb") as f:
                f.write(data)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to download CrowS-Pairs CSV from {url}: {e}. "
                f"Place the file manually at the cache path: {cache_path}"
            )
        if not os.path.getsize(cache_path) > 0:
            raise RuntimeError(
                f"Downloaded CrowS-Pairs CSV is empty; cache path: {cache_path}"
            )

    with open(cache_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        groups = {}
        for row in reader:
            bias_type = (row.get("bias_type") or "").strip()
            sent_more = (row.get("sent_more") or "").strip()
            sent_less = (row.get("sent_less") or "").strip()
            if not bias_type or not sent_more or not sent_less:
                continue
            # Universal mapping: stereotype = sent_more, anti = sent_less.
            groups.setdefault(bias_type, []).append((sent_more, sent_less))
    if not groups:
        raise RuntimeError(
            f"Parsed 0 usable rows from CrowS-Pairs CSV at cache path: {cache_path}"
        )
    return groups


def parse_args():
    p = argparse.ArgumentParser(
        description="Build embedding-layer bias steering direction(s) for LLaDA."
    )
    p.add_argument("--source", default="crows", choices=["crows", "json"],
                   help="crows = per-category dirs from CrowS-Pairs; "
                        "json = legacy single direction from a probe file")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--hook-module", default=DEFAULT_HOOK_MODULE,
                   help="dotted module path used ONLY for layer spec 'emb'")
    p.add_argument("--layer", default=DEFAULT_LAYER,
                   help="layer spec: 'emb' (input embedding, default) OR an int "
                        "transformer BLOCK index (e.g. 14)")
    p.add_argument("--layers", default=None,
                   help="comma list of layer specs to build in ONE run "
                        "(e.g. 'emb,12,14,16'); overrides --layer when set")
    p.add_argument("--device", default="cuda")
    # crows source
    p.add_argument("--crows-url", default=DEFAULT_CROWS_URL)
    p.add_argument("--categories", default=None,
                   help="comma list of bias_type tokens to keep (default all)")
    p.add_argument("--subset-terms", default=None,
                   help="comma list of case-insensitive substrings; keep only pairs "
                        "whose STEREOTYPE sentence contains one (e.g. a referent "
                        "group). Builds a group-specific direction from the subset.")
    p.add_argument("--subset-name", default=None,
                   help="output stem to use when --subset-terms is set (e.g. "
                        "'race_black' -> L{L}/race_black.pt); defaults to the "
                        "category stem, which may overwrite the full-category file")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                   help="(crows) output dir for per-category .pt files")
    p.add_argument("--min-pairs", type=int, default=20,
                   help="(crows) warn if a category has fewer pairs than this")
    # legacy json source
    p.add_argument("--pairs", default=DEFAULT_PAIRS,
                   help="(json) minimal-pair probe file")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="(json) legacy single-direction output path")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# MULTI-LAYER capture machinery.
#
# Registers ONE capture hook per requested layer spec and does a SINGLE forward
# per sentence; each hook stashes that layer's masked-mean. Handles the block
# tuple return (hidden at index 0) and the bare-tensor wte return identically.
# --------------------------------------------------------------------------- #
def make_multilayer_embedder(model, tok, specs, device, hook_module):
    """Attach capture hooks for every layer spec.

    Returns (embed_means_fn, handles, state) where embed_means_fn(sentence) does
    ONE forward and returns dict[spec] -> (H,) masked-mean tensor (float32).
    """
    captured = {}
    state = {"hidden_size": {}}
    handles = []

    def make_capture(spec):
        def capture_hook(mod, inputs, output):
            captured[spec] = hidden_from_output(output)
            return None  # capture only; never modify
        return capture_hook

    for spec in specs:
        module = resolve_layer_module(model, spec, hook_module)
        handles.append(module.register_forward_hook(make_capture(spec)))

    def embed_means(sentence):
        input_ids = torch.tensor(tok(sentence)["input_ids"], device=device).unsqueeze(0)
        captured.clear()
        with torch.no_grad():
            model(input_ids)
        mask = (input_ids != PAD_TOKEN_ID).to(torch.float32).unsqueeze(-1)  # (1,T,1)
        count = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
        out = {}
        for spec in specs:
            hid = captured[spec].to(torch.float32)  # (1, T, H)
            state["hidden_size"][spec] = hid.shape[-1]
            summed = (hid * mask).sum(dim=1)  # (1, H)
            out[spec] = (summed / count).squeeze(0)  # (H,)
        return out

    return embed_means, handles, state


def cosine(a, b):
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float(torch.dot(a, b) / denom)


# --------------------------------------------------------------------------- #
# CrowS source: per-category directions + coherence metrics.
# --------------------------------------------------------------------------- #
def category_stats(diffs, embed_norms):
    """Compute direction + coherence metrics for one category's diff list.

    diffs: list of (H,) tensors (stereotype - anti).
    embed_norms: list of per-sentence embedding L2 floats.
    Returns dict of metrics.
    """
    stacked = torch.stack(diffs, dim=0)  # (n, H)
    direction = stacked.mean(dim=0).to(torch.float32)  # (H,)
    raw_norm = float(direction.norm())
    mean_diff_norm = float(stacked.norm(dim=1).mean())
    # norm_ratio ~ 1 -> diffs agree in direction (coherent); ~ 0 -> they cancel.
    norm_ratio = raw_norm / mean_diff_norm if mean_diff_norm > 1e-12 else 0.0
    avg_embed_norm = float(sum(embed_norms) / len(embed_norms)) if embed_norms else 0.0

    splithalf = None
    n = stacked.shape[0]
    if n >= 4:
        g = torch.Generator().manual_seed(SPLITHALF_SEED)
        perm = torch.randperm(n, generator=g)
        half = n // 2
        a = stacked[perm[:half]].mean(dim=0)
        b = stacked[perm[half:]].mean(dim=0)
        splithalf = cosine(a, b)

    return {
        "direction": direction.cpu(),
        "raw_norm": raw_norm,
        "avg_embed_norm": avg_embed_norm,
        "norm_ratio": norm_ratio,
        "splithalf_cosine": splithalf,
    }


def layer_out_dir(base_out_dir, spec):
    """Output dir for a layer spec: 'emb' -> base (backward compat); int L -> base/L{L}."""
    if spec == "emb":
        return base_out_dir
    return os.path.join(base_out_dir, f"L{int(spec)}")


def subset_by_terms(groups, terms):
    """Keep only pairs whose STEREOTYPE sentence contains a term (case-insensitive).

    terms: list of lowercase substrings. Prints match count + 5 example pairs so
    the referent subset is auditable. Returns the filtered groups dict.
    """
    out = {}
    total = 0
    for cat, pairs in groups.items():
        kept = [(s, a) for (s, a) in pairs if any(t in s.lower() for t in terms)]
        if kept:
            out[cat] = kept
            total += len(kept)
    print(f"  subset-terms {terms}: matched {total} pairs "
          f"(from {sum(len(v) for v in groups.values())}).")
    examples = [p for pairs in out.values() for p in pairs][:5]
    for i, (s, a) in enumerate(examples, 1):
        print(f"    ex{i} stereotype: {s}")
        print(f"    ex{i} anti      : {a}")
    return out


def run_crows(args, model, tok, specs):
    print("Loading CrowS-Pairs ...")
    groups = load_crows(args.crows_url, CROWS_CACHE)
    print(f"  loaded {sum(len(v) for v in groups.values())} pairs across "
          f"{len(groups)} bias_type categories (cache: {CROWS_CACHE}).")

    if args.categories:
        wanted = {c.strip() for c in args.categories.split(",") if c.strip()}
        missing = wanted - set(groups)
        if missing:
            print(f"  WARNING: requested categories not in data: {sorted(missing)}")
        groups = {k: v for k, v in groups.items() if k in wanted}

    if args.subset_terms:
        terms = [t.strip().lower() for t in args.subset_terms.split(",") if t.strip()]
        groups = subset_by_terms(groups, terms)

    embed_means, handles, state = make_multilayer_embedder(
        model, tok, specs, args.device, args.hook_module
    )

    # Per-spec accumulators.
    summaries = {spec: [] for spec in specs}  # spec -> list of summary tuples
    all_diffs = {spec: [] for spec in specs}
    all_embed_norms = {spec: [] for spec in specs}
    try:
        for category in sorted(groups):
            pairs = groups[category]
            if len(pairs) < args.min_pairs:
                print(f"  WARNING: category '{category}' has only {len(pairs)} pairs "
                      f"(< --min-pairs {args.min_pairs}); metrics may be noisy.")
            diffs = {spec: [] for spec in specs}
            embed_norms = {spec: [] for spec in specs}
            for j, (stereo, anti) in enumerate(pairs):
                s_means = embed_means(stereo)  # ONE forward, all layers
                a_means = embed_means(anti)
                for spec in specs:
                    s_mean, a_mean = s_means[spec], a_means[spec]
                    embed_norms[spec].append(float(s_mean.norm()))
                    embed_norms[spec].append(float(a_mean.norm()))
                    diffs[spec].append(s_mean - a_mean)
                if (j + 1) % 25 == 0 or (j + 1) == len(pairs):
                    print(f"    {category:>20} [{j + 1}/{len(pairs)}]")

            for spec in specs:
                all_diffs[spec].extend(diffs[spec])
                all_embed_norms[spec].extend(embed_norms[spec])
                stats = category_stats(diffs[spec], embed_norms[spec])
                out_dir = layer_out_dir(args.out_dir, spec)
                os.makedirs(out_dir, exist_ok=True)
                stem = safe_name(args.subset_name) if args.subset_name else safe_name(category)
                out_path = os.path.join(out_dir, f"{stem}.pt")
                torch.save({
                    "direction": stats["direction"],
                    "bias_type": category,
                    "n_pairs": len(pairs),
                    "hidden_size": int(state["hidden_size"][spec]),
                    "hook_module": args.hook_module,
                    "layer": spec,
                    "raw_norm": stats["raw_norm"],
                    "avg_embed_norm": stats["avg_embed_norm"],
                    "norm_ratio": stats["norm_ratio"],
                    "splithalf_cosine": stats["splithalf_cosine"],
                    "source": "crows",
                }, out_path)
                print(f"  saved {out_path}")
                summaries[spec].append((category, len(pairs), stats["raw_norm"],
                                        stats["avg_embed_norm"], stats["norm_ratio"],
                                        stats["splithalf_cosine"]))

        # Combined "all" direction per layer across every diff (mixed vector).
        # Skip when subsetting: the subset already IS the target direction.
        for spec in specs:
            if args.subset_terms or not all_diffs[spec]:
                continue
            all_stats = category_stats(all_diffs[spec], all_embed_norms[spec])
            out_dir = layer_out_dir(args.out_dir, spec)
            os.makedirs(out_dir, exist_ok=True)
            all_path = os.path.join(out_dir, "all.pt")
            torch.save({
                "direction": all_stats["direction"],
                "bias_type": "all",
                "n_pairs": len(all_diffs[spec]),
                "hidden_size": int(state["hidden_size"][spec]),
                "hook_module": args.hook_module,
                "layer": spec,
                "raw_norm": all_stats["raw_norm"],
                "avg_embed_norm": all_stats["avg_embed_norm"],
                "norm_ratio": all_stats["norm_ratio"],
                "splithalf_cosine": all_stats["splithalf_cosine"],
                "source": "crows",
            }, all_path)
            print(f"  saved {all_path}")
            summaries[spec].append(("all", len(all_diffs[spec]), all_stats["raw_norm"],
                                    all_stats["avg_embed_norm"], all_stats["norm_ratio"],
                                    all_stats["splithalf_cosine"]))
    finally:
        for h in handles:
            h.remove()

    for spec in specs:
        print_summary(summaries[spec], spec)


def print_summary(summary, spec):
    label = "emb (input embedding)" if spec == "emb" else f"block L{spec}"
    print("=" * 84)
    print(f"SUMMARY [layer={label}]: per-category directions + coherence metrics")
    print("-" * 84)
    hdr = (f"{'category':>20} | {'n_pairs':>7} | {'raw_norm':>9} | {'avg_emb':>9} | "
           f"{'norm_ratio':>10} | {'splithalf_cos':>13}")
    print(hdr)
    print("-" * 84)
    for cat, n, raw, avg, ratio, sh in summary:
        sh_str = f"{'None':>13}" if sh is None else f"{sh:>13.4f}"
        print(f"{cat:>20} | {n:>7} | {raw:>9.4f} | {avg:>9.4f} | "
              f"{ratio:>10.4f} | {sh_str}")
    print("-" * 84)
    print("interpretation: norm_ratio & splithalf_cos NEAR 1 = coherent, usable")
    print("                direction (pairs agree); NEAR 0 = mostly noise -> needs")
    print("                cleaner / more pairs, or steer at a different layer.")
    print("=" * 84)


# --------------------------------------------------------------------------- #
# Legacy JSON source: single mixed direction -> direction.pt (unchanged behavior).
# --------------------------------------------------------------------------- #
def run_json(args, model, tok, spec):
    with open(args.pairs) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} minimal pairs from {args.pairs}.")

    embed_means, handles, state = make_multilayer_embedder(
        model, tok, [spec], args.device, args.hook_module
    )
    try:
        diffs = []
        embed_norms = []
        for i, pair in enumerate(pairs):
            s_mean = embed_means(pair["stereotype"])[spec]
            a_mean = embed_means(pair["anti_stereotype"])[spec]
            embed_norms.append(float(s_mean.norm()))
            embed_norms.append(float(a_mean.norm()))
            diffs.append(s_mean - a_mean)
            print(f"  [{i + 1}/{len(pairs)}] {pair.get('category', '?'):>11} | "
                  f"|diff|={float((s_mean - a_mean).norm()):.4f}")
    finally:
        for h in handles:
            h.remove()

    direction = torch.stack(diffs, dim=0).mean(dim=0).to(torch.float32)  # (H,)
    raw_norm = float(direction.norm())
    avg_embed_norm = float(sum(embed_norms) / len(embed_norms))

    torch.save({
        "direction": direction.cpu(),
        "hook_module": args.hook_module,
        "hidden_size": int(state["hidden_size"][spec]),
        "num_pairs": len(pairs),
        "layer": spec,
        "raw_norm": raw_norm,
        "avg_embed_norm": avg_embed_norm,
    }, args.out)

    print("-" * 60)
    print("Saved steering direction.")
    print(f"  out            : {args.out}")
    print(f"  hidden_size    : {state['hidden_size']}")
    print(f"  num_pairs      : {len(pairs)}")
    print(f"  raw_norm       : {raw_norm:.4f}  (L2 of mean stereotype-anti diff)")
    print(f"  avg_embed_norm : {avg_embed_norm:.4f}  (mean per-sentence embed L2)")
    print(f"  => steering with alpha multiplies a vector of norm {raw_norm:.4f};")
    print(f"     embeddings have norm ~{avg_embed_norm:.4f}, so calibrate alpha")
    print(f"     so that alpha*raw_norm is a meaningful fraction of avg_embed_norm.")
    print("-" * 60)


def main():
    args = parse_args()

    # Resolve the layer spec(s): --layers (comma list) wins over --layer.
    if args.layers:
        specs = [parse_layer_spec(s) for s in args.layers.split(",") if s.strip()]
    else:
        specs = [parse_layer_spec(args.layer)]
    if args.source == "json" and len(specs) != 1:
        raise ValueError("--source json builds a single direction; pass exactly one layer.")

    print("=" * 60)
    print("LLaDA bias-direction builder (per-layer mean difference)")
    print(f"  source      : {args.source}")
    print(f"  model-path  : {args.model_path}")
    print(f"  hook-module : {args.hook_module}  (used only for layer 'emb')")
    print(f"  layers      : {specs}")
    print(f"  device      : {args.device}")
    if args.source == "crows":
        print(f"  crows-url   : {args.crows_url}")
        print(f"  out-dir     : {args.out_dir}")
        print(f"  categories  : {args.categories or 'all'}")
        print(f"  subset-terms: {args.subset_terms or 'none'}")
        print(f"  subset-name : {args.subset_name or 'none'}")
        print(f"  min-pairs   : {args.min_pairs}")
    else:
        print(f"  pairs       : {args.pairs}")
        print(f"  out         : {args.out}")
    print("=" * 60)

    print("Loading model and tokenizer ...")
    model = (
        AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        .to(args.device)
        .eval()
    )
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for spec in specs:
        mod = resolve_layer_module(model, spec, args.hook_module)
        label = "emb" if spec == "emb" else f"block L{spec}"
        print(f"Resolved layer {label} -> {type(mod).__name__}")

    if args.source == "crows":
        run_crows(args, model, tok, specs)
    else:
        run_json(args, model, tok, specs[0])


if __name__ == "__main__":
    main()
