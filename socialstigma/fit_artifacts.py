#!/usr/bin/env python
"""Fit polarity-specific SocialStigmaQA artifacts on the frozen calibration set.

This reuses the exact activation sites and fitting primitives of the BBQ
baseline suite.  It only changes the contrast examples: positive is the
dataset-annotated biased answer and negative is the opposite yes/no answer.
Race-stigma rows never enter this stage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("DLM_BIAS_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "steering"))
sys.path.insert(0, str(ROOT / "baselines"))

import bbq_eval  # noqa: E402
import pid_steer  # noqa: E402
import common  # noqa: E402
import calib  # noqa: E402
import directions  # noqa: E402
import linearact  # noqa: E402
import itic  # noqa: E402


ITEMS_DIR = ROOT / "data" / "socialstigma" / "items"
CACHE_ROOT = ROOT / "socialstigma" / "cache"
MANIFEST_PATH = CACHE_ROOT / "artifact_manifest.json"
POLARITIES = ("yes", "no")
EXPECTED_N = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def contrast_items(rows: list[dict], tok) -> list[dict]:
    items = []
    for row in rows:
        target_idx = pid_steer.black_idx_of(row)
        unknown_idx = pid_steer.unk_idx_of(row)
        other_idx = next(k for k in range(3) if k not in {target_idx, unknown_idx})
        prompt = bbq_eval.build_prompt(row)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        items.append({
            "row": row,
            "target_idx": target_idx,
            "other_idx": other_idx,
            "target_answer_text": str(row[f"ans{target_idx}"]).strip(),
            "other_answer_text": str(row[f"ans{other_idx}"]).strip(),
            "chat_prompt": chat,
        })
    return items


def save_blob(blob: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def fit_one(polarity: str, model, tok) -> dict:
    rows = load_rows(ITEMS_DIR / f"calibration_{polarity}.jsonl")
    assert len(rows) == EXPECTED_N
    assert all(r["biased_answer"] == polarity for r in rows)
    assert all(r["prompt_style"] == "original" for r in rows)
    items = contrast_items(rows, tok)
    out_dir = CACHE_ROOT / polarity
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (f"socialstigma_nonrace_original_{polarity}_biased_target_"
              f"n{EXPECTED_N}_seed20260905")

    print(f"[fit:{polarity}] collecting block activations", flush=True)
    block = calib.collect_activations(
        "block", model=model, tok=tok, items=items, save=False, target="black")
    block.update({"source": source, "benchmark": "SocialStigmaQA-MC3",
                  "polarity": polarity})
    save_blob(block, out_dir / "calib_block.pt")
    acts = block["acts"].to(torch.float32)
    labels = block["labels"]
    assert labels.tolist() == [v for _ in range(EXPECTED_N) for v in (1, 0)]

    per_item_diff = acts[0::2] - acts[1::2]
    arrows = per_item_diff.mean(dim=0)
    assert tuple(arrows.shape) == (common.N_LAYERS, common.H_MODEL)
    assert torch.isfinite(arrows).all() and bool((arrows.norm(dim=1) > 0).all())
    arrows_blob = {
        "r": arrows,
        "n_layers": common.N_LAYERS,
        "n_items": EXPECTED_N,
        "method": "answer_text_anchored_diff_in_means",
        "source": source,
        "benchmark": "SocialStigmaQA-MC3",
        "polarity": polarity,
        "target_semantics": f"biased answer = {polarity}",
        "negative_semantics": "opposite yes/no safe answer",
        "per_layer_raw_norm": [float(arrows[k].norm()) for k in range(common.N_LAYERS)],
    }
    save_blob(arrows_blob, out_dir / "arrows.pt")

    # ActAdd is intentionally a single contrast pair, exactly as in the BBQ
    # baseline.  Reuse the first deterministic calibration pair.
    actadd_r = per_item_diff[0].clone()
    save_blob({
        "r": actadd_r,
        "n_layers": common.N_LAYERS,
        "n_items": 1,
        "pair_index": 0,
        "method": "actadd_single_pair",
        "source": source,
        "benchmark": "SocialStigmaQA-MC3",
        "polarity": polarity,
        "per_layer_raw_norm": [float(actadd_r[k].norm()) for k in range(common.N_LAYERS)],
    }, out_dir / "actadd_dir.pt")

    labels_bool = labels.bool()
    mean_diff = (acts[labels_bool].mean(0) - acts[~labels_bool].mean(0)).to(torch.float32)
    assert torch.allclose(mean_diff, arrows, atol=1e-4, rtol=1e-4)
    save_blob({
        "mean_diff": mean_diff,
        "where": "block",
        "feat": common.H_MODEL,
        "n_layers": common.N_LAYERS,
        "n_items": EXPECTED_N,
        "source": source,
        "target": "black",
        "benchmark": "SocialStigmaQA-MC3",
        "polarity": polarity,
    }, out_dir / "meanact_meandiff_block.pt")
    del per_item_diff, arrows, actadd_r, mean_diff, acts, block
    gc.collect()

    print(f"[fit:{polarity}] collecting MLP-hidden activations", flush=True)
    mlp = calib.collect_activations(
        "mlp_hidden", model=model, tok=tok, items=items, save=False, target="black")
    mlp.update({"source": source, "benchmark": "SocialStigmaQA-MC3",
                "polarity": polarity})
    save_blob(mlp, out_dir / "calib_mlp_hidden.pt")

    stats = linearact.fit_stats_from_blob(mlp)
    stats.update({"target": "black", "source": source,
                  "benchmark": "SocialStigmaQA-MC3", "polarity": polarity})
    save_blob(stats, out_dir / "linearact_stats.pt")

    mlp_acts = mlp["acts"]
    mlp_labels = mlp["labels"]
    auroc = torch.empty(common.N_LAYERS, common.H_MLP, dtype=torch.float32)
    for k in range(common.N_LAYERS):
        auroc[k], _ = directions.auroc_per_neuron(mlp_acts[:, k, :], mlp_labels)
        print(f"[fit:{polarity}:aura] layer {k + 1}/{common.N_LAYERS}", flush=True)
    assert torch.isfinite(auroc).all()
    save_blob({
        "auroc": auroc,
        "where": "mlp_hidden",
        "feat": common.H_MLP,
        "n_layers": common.N_LAYERS,
        "n_items": EXPECTED_N,
        "source": source,
        "target": "black",
        "benchmark": "SocialStigmaQA-MC3",
        "polarity": polarity,
    }, out_dir / "aura_auroc.pt")
    del stats, auroc, mlp_acts, mlp_labels, mlp
    gc.collect()

    print(f"[fit:{polarity}] collecting attention-head activations", flush=True)
    attn = calib.collect_activations(
        "attn_head", model=model, tok=tok, items=items, save=False, target="black")
    attn.update({"source": source, "benchmark": "SocialStigmaQA-MC3",
                 "polarity": polarity})
    save_blob(attn, out_dir / "calib_attn_head.pt")
    attn_acts = attn["acts"].to(torch.float32).view(
        -1, common.N_LAYERS, common.N_HEADS, common.D_HEAD)
    attn_labels = attn["labels"].long()
    theta = torch.zeros(common.N_LAYERS, common.N_HEADS, common.D_HEAD)
    sigma = torch.zeros(common.N_LAYERS, common.N_HEADS)
    val_acc = torch.zeros(common.N_LAYERS, common.N_HEADS)
    margin = torch.zeros(common.N_LAYERS, common.N_HEADS)
    for k in range(common.N_LAYERS):
        for head in range(common.N_HEADS):
            va, th, sg = itic._fit_one_head(attn_acts[:, k, head, :], attn_labels)
            val_acc[k, head] = va
            theta[k, head] = th
            sigma[k, head] = sg
            margin[k, head] = itic.head_margin(attn_acts[:, k, head, :], attn_labels)
        print(f"[fit:{polarity}:itic] layer {k + 1}/{common.N_LAYERS} "
              f"best={float(val_acc[k].max()):.3f}", flush=True)
    assert torch.isfinite(theta).all() and torch.isfinite(sigma).all()
    save_blob({
        "theta": theta,
        "sigma": sigma,
        "val_acc": val_acc,
        "margin": margin,
        "n_layers": common.N_LAYERS,
        "n_heads": common.N_HEADS,
        "d_head": common.D_HEAD,
        "n_items": EXPECTED_N,
        "source": source,
        "method": "iti_c",
        "target": "black",
        "benchmark": "SocialStigmaQA-MC3",
        "polarity": polarity,
        "direction": f"biased_{polarity}_minus_safe_answer",
    }, out_dir / "itic_probes.pt")
    del attn_acts, attn_labels, theta, sigma, val_acc, attn
    gc.collect()
    torch.cuda.empty_cache()

    artifacts = {}
    for path in sorted(out_dir.glob("*.pt")):
        artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return {"polarity": polarity, "source": source, "n_items": EXPECTED_N,
            "artifacts": artifacts}


def validate_manifest(manifest: dict) -> None:
    expected = {
        "arrows.pt": ("r", (common.N_LAYERS, common.H_MODEL)),
        "actadd_dir.pt": ("r", (common.N_LAYERS, common.H_MODEL)),
        "meanact_meandiff_block.pt": ("mean_diff", (common.N_LAYERS, common.H_MODEL)),
        "linearact_stats.pt": ("n_layers", None),
        "aura_auroc.pt": ("auroc", (common.N_LAYERS, common.H_MLP)),
        "itic_probes.pt": ("theta", (common.N_LAYERS, common.N_HEADS, common.D_HEAD)),
    }
    for polarity in POLARITIES:
        out_dir = CACHE_ROOT / polarity
        for name, (key, shape) in expected.items():
            path = out_dir / name
            assert path.exists(), path
            blob = torch.load(path, map_location="cpu")
            assert key in blob
            if shape is not None:
                assert tuple(blob[key].shape) == shape, (path, blob[key].shape)
                assert torch.isfinite(blob[key]).all(), path
    assert set(manifest["fits"]) == set(POLARITIES)


def add_itic_margin(polarity: str) -> dict:
    """Offline (CPU) amendment: add the standardized head margin used as the
    ITI-C val_acc tie-breaker to an existing itic_probes.pt, recomputed from
    the cached attn_head activations of the same fit.  theta/sigma/val_acc are
    left untouched, so the legacy (index tie-break) ITI-C run is unchanged."""
    out_dir = CACHE_ROOT / polarity
    probes_path = out_dir / "itic_probes.pt"
    probes = torch.load(probes_path, map_location="cpu")
    attn = torch.load(out_dir / "calib_attn_head.pt", map_location="cpu")
    assert attn["source"] == probes["source"], (attn["source"], probes["source"])
    acts = attn["acts"].to(torch.float32).view(
        -1, common.N_LAYERS, common.N_HEADS, common.D_HEAD)
    labels = attn["labels"].long()
    # Re-derive theta from the same activations and confirm the cached probes
    # come from these activations before attaching a margin to them.
    va0, th0, sg0 = itic._fit_one_head(acts[:, 0, 0, :], labels)
    assert torch.allclose(th0, probes["theta"][0, 0], atol=1e-5), "probes/activations mismatch"
    assert abs(sg0 - float(probes["sigma"][0, 0])) < 1e-4, "probes/activations mismatch"
    margin = torch.zeros(common.N_LAYERS, common.N_HEADS)
    for k in range(common.N_LAYERS):
        for head in range(common.N_HEADS):
            margin[k, head] = itic.head_margin(acts[:, k, head, :], labels)
    assert torch.isfinite(margin).all()
    probes["margin"] = margin
    probes["margin_definition"] = "||mean_pos - mean_neg|| / std(proj onto unit theta); tie-breaker only"
    save_blob(probes, probes_path)
    n_tied = int((probes["val_acc"] >= float(probes["val_acc"].max()) - 1e-6).sum())
    info = {"polarity": polarity, "n_heads_tied_at_top_val_acc": n_tied,
            "margin_min": float(margin.min()), "margin_max": float(margin.max())}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        manifest["fits"][polarity]["artifacts"]["itic_probes.pt"] = {
            "bytes": probes_path.stat().st_size, "sha256": sha256(probes_path)}
        manifest.setdefault("amendments", []).append(
            {"artifact": f"{polarity}/itic_probes.pt", "added": "margin",
             "reason": "val_acc tie-break for top-K head selection", **info})
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[fit:{polarity}:itic] added margin -> {probes_path} {info}", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--polarity", choices=("all",) + POLARITIES, default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--itic-margin", action="store_true",
                    help="CPU-only: add the ITI-C tie-break margin to existing "
                         "itic_probes.pt files from the cached activations")
    args = ap.parse_args()
    if args.itic_margin:
        for polarity in (POLARITIES if args.polarity == "all" else (args.polarity,)):
            add_itic_margin(polarity)
        return
    if args.polarity == "all" and MANIFEST_PATH.exists() and not args.force:
        try:
            existing = json.loads(MANIFEST_PATH.read_text())
            validate_manifest(existing)
            print(f"[fit] SKIP: validated complete artifact set at {CACHE_ROOT}")
            return
        except (AssertionError, KeyError, json.JSONDecodeError, FileNotFoundError):
            print("[fit] existing artifact set is incomplete/invalid; rebuilding", flush=True)
    model, tok = common.load_model()
    selected = POLARITIES if args.polarity == "all" else (args.polarity,)
    fits = {}
    for polarity in selected:
        fits[polarity] = fit_one(polarity, model, tok)
    manifest = {
        "benchmark": "SocialStigmaQA-MC3",
        "model": "LLaDA-8B-Instruct",
        "calibration": "non-race original prompts; separate yes/no target fits",
        "fits": fits,
    }
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.polarity == "all":
        validate_manifest(manifest)
    print(f"[fit] PASS -> {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
