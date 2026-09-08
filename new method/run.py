#!/usr/bin/env python3
"""Run automatic option identification and semantic-vector steering on local LLaDA.

Default: auto_pi. --compare adds paired controls; --plan-only prepares and checks
local inputs without importing torch or loading a model. Custom --items may omit
all annotations. Live inference requires an existing Slurm GPU allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from cohort import (annotated_indices, prepare_cohort, read_rows, score_output,
                    sha256, summarize)

COMPARE_ARMS = ["clean", "selector_only", "fixed_openloop", "auto_openloop", "auto_pi"]
DEFAULT_EXCLUSIONS = ["data/bbq_items/_sweep400.jsonl"] + [
    f"results/balanced_seeds/seed{seed}/decode_pid/rot0/cond_dpid_PI_samples.jsonl"
    for seed in (1, 2, 3)]
VISIBLE_KEYS = ("context", "question", "ans0", "ans1", "ans2", "prompt_override")


def model_preflight(path):
    """Read only local model metadata; never hash or load large weights."""
    path = Path(path)
    result = {"path": str(path), "available": False, "missing": []}
    if not path.is_dir():
        result["missing"] = ["model directory"]
        return result
    if not (path / "config.json").is_file():
        result["missing"].append("config.json")
    index = next((path / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
                  if (path / name).is_file()), None)
    if index:
        manifest = json.loads(index.read_text())
        names = sorted(set(manifest.get("weight_map", {}).values()))
        if not names:
            result["missing"].append("nonempty weight_map")
    else:
        names = [name for name in ("model.safetensors", "pytorch_model.bin") if (path / name).is_file()]
        if not names:
            result["missing"].append("local model weights")
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe shard filename in local model index")
        if not (path / name).is_file():
            result["missing"].append(name)
    result["available"] = not result["missing"]
    result["weight_file_count"] = len(names)
    result["validation"] = "local config/shard existence; tensor/tokenizer compatibility checked at load"
    return result


def planned_forward_calls(arms, n_items, steps=64):
    automatic = any(arm in ("auto_pi", "auto_openloop", "selector_only") for arm in arms)
    generation = {arm: (0 if arm == "selector_only" else steps if arm in ("clean", "fixed_openloop")
                         else steps + 1) for arm in arms}
    shared = 3 if automatic else 0
    standalone = {arm: calls + (3 if arm in ("auto_pi", "auto_openloop", "selector_only") else 0)
                  for arm, calls in generation.items()}
    return {"shared_selector_per_visible_item": shared,
            "generation_per_arm_per_visible_item": generation,
            "standalone_per_arm_per_visible_item": standalone,
            "actual_per_visible_item": shared + sum(generation.values()),
            "actual_total": n_items * (shared + sum(generation.values()))}


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=("smoke", "pilot", "full"), default="pilot")
    ap.add_argument("--data-root", type=Path, default=Path(os.environ.get("DLM_BIAS_ROOT", CODE_ROOT)))
    ap.add_argument("--model", type=Path)
    ap.add_argument("--arrows", type=Path)
    ap.add_argument("--fit-examples", type=Path, help="exact direction-fitting example manifest")
    ap.add_argument("--items", type=Path, help="custom canonical visible-item JSONL; annotations optional")
    ap.add_argument("--exclude-items", action="append", type=Path, default=[], help="additional prior-cohort manifest")
    ap.add_argument("--out-root", type=Path, required=True, help="new or empty output directory")
    ap.add_argument("--limit", type=int, help="semantic items; 0=all; presets choose 2/100/all for benchmark")
    ap.add_argument("--rotations", type=int, choices=(1, 3), help="benchmark default3, custom-input default1")
    ap.add_argument("--compare", action="store_true", help="run paired clean/readout/fixed/static/PI arms")
    ap.add_argument("--oracle-diagnostic", action="store_true", help="add separately labeled annotation-assisted PI")
    ap.add_argument("--plan-only", action="store_true", help="validate/prepare CPU plan only; missing weights reported")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--kp", type=float, default=3.0)
    ap.add_argument("--ki", type=float, default=0.1)
    ap.add_argument("--setpoint", type=float, default=0.9)
    ap.add_argument("--alpha-max", type=float, default=6.0)
    ap.add_argument("--fixed-alpha", type=float, default=4.0)
    ap.add_argument("--sensor-case", choices=("upper", "both"), default="upper")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--seconds-per-forward", type=float,
                    help="optional measured hardware rate for rough plan ETA; no throughput is assumed")
    return ap


def make_plan(args):
    import math
    data_root = args.data_root.resolve()
    arrows = (args.arrows or data_root / "steering/arrows.pt").resolve()
    fit = (args.fit_examples or arrows.with_name("direction_examples.jsonl")).resolve()
    model = (args.model or data_root / "LLaDA-8B-Instruct").absolute()
    source = (args.items or data_root / "data/bbq_cache/Race_ethnicity.jsonl").resolve()
    benchmark = args.items is None
    limit = args.limit if args.limit is not None else ({"smoke": 2, "pilot": 100, "full": 0}[args.preset] if benchmark else 0)
    rotations = args.rotations or (3 if benchmark else 1)
    exclusions = [(data_root / name).resolve() for name in DEFAULT_EXCLUSIONS] if benchmark else []
    exclusions.extend(path.resolve() for path in args.exclude_items)
    for path in [source, arrows, fit] + exclusions:
        if not path.is_file():
            raise FileNotFoundError(f"required local artifact is missing: {path}")
    values = [args.kp, args.ki, args.setpoint, args.alpha_max, args.fixed_alpha]
    if (not all(math.isfinite(value) for value in values) or args.steps < 1
            or min(args.kp, args.ki, args.alpha_max, args.fixed_alpha) < 0
            or not 0 <= args.setpoint <= 1 or args.fixed_alpha > args.alpha_max):
        raise ValueError("invalid PI/sampler settings")
    if args.seconds_per_forward is not None and (not math.isfinite(args.seconds_per_forward) or args.seconds_per_forward <= 0):
        raise ValueError("seconds-per-forward must be positive and finite")
    fit_rows = read_rows(fit)
    if not fit_rows:
        raise ValueError("direction-fit manifest must contain its exact fitting examples")
    excluded = [row for path in exclusions for row in read_rows(path)]
    records, cohort = prepare_cohort(read_rows(source), fit_rows, excluded, limit=limit,
                                     rotations=rotations, seed=args.seed, benchmark=benchmark)
    arms = list(COMPARE_ARMS) if args.compare else ["auto_pi"]
    if args.oracle_diagnostic:
        arms.append("oracle_pi")
    budget = planned_forward_calls(arms, len(records), args.steps)
    settings = {name: getattr(args, name) for name in
                ("steps", "kp", "ki", "setpoint", "alpha_max", "fixed_alpha", "sensor_case")}
    code_files = [HERE / name for name in ("run.py", "pipeline.py", "cohort.py")] + [
        CODE_ROOT / name for name in ("steering/target_selector.py", "steering/denoise_pid.py", "eval/bbq_eval.py", "eval/bias_metrics.py")]
    plan = {
        "schema_version": 1, "method": "automatic_raw_direction_mapping_then_semantic_PI",
        "experimental_status": "implementation ready; mapping quality and comparative efficacy unvalidated",
        "data_root": str(data_root), "code_root": str(CODE_ROOT), "model": model_preflight(model),
        "arms": arms, "settings": settings, "cohort": cohort, "forward_budget": budget,
        "artifacts": [{"path": str(path), "sha256": sha256(path)} for path in [source, arrows, fit] + exclusions],
        "code_files": [{"path": str(path), "sha256": sha256(path)} for path in code_files],
        "arrows_path": str(arrows), "fit_examples_path": str(fit),
        "selected_semantic_keys": [record["semantic_key"] for record in records if record["rotation"] == 0],
        "direction_source_block": 14, "actuator_blocks": list(range(32)),
        "selector": "all3 answer-span raw projections; no annotation filters or fallback; ABC tie order",
        "auto_openloop_policy": "freeze exact first PI.update(initial unsteered selected-letter probability)",
        "readout_control": "selector_only directly outputs the selected letter; different output family",
        "sampler": {"gen_length": 32, "block_length": 32, "temperature": 0.0, "cfg_scale": 0.0},
        "error_policy": "fail visibly; preserve raw outputs; never drop wrong mappings or failed items",
        "runtime_estimate_s": (budget["actual_total"] * args.seconds_per_forward if args.seconds_per_forward else None),
        "runtime_estimate_note": "approximate supplied average forward rate; excludes loading, prompt-length variation, and I/O",
    }
    return plan, records


def write_json(path, value):
    with Path(path).open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def execute(plan, cohort, out_root):
    """Load once, share selection across arms, and score only after output exists."""
    if not plan["model"]["available"]:
        raise ValueError("model weights are unavailable; supply --model pointing to a complete local LLaDA model")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("live inference requires an existing Slurm GPU allocation (SLURM_JOB_ID)")
    # These flags and local_files_only prevent accidental downloads or API use.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tqdm import tqdm
    import torch
    from transformers import AutoModel, AutoTokenizer
    from pipeline import Engine, Settings, load_direction, visible_fields
    if not torch.cuda.is_available():
        raise RuntimeError("allocated CUDA device is unavailable")
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < 24000 * 1024 * 1024:
        raise RuntimeError("allocated logical CUDA device 0 needs at least 24,000 MiB free before model loading")
    for artifact in plan["artifacts"] + plan["code_files"]:
        if sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("planned input or implementation changed before execution")
    torch.manual_seed(plan["cohort"]["selection_seed"])
    started = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(plan["model"]["path"], trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(plan["model"]["path"], trust_remote_code=True,
                                     local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    vector = load_direction(plan["arrows_path"])
    torch.cuda.synchronize()
    loading_s = time.perf_counter() - started
    records, item_timings, mapping_calls = [], [], 0
    inference_started = time.perf_counter()
    with Engine(model, tok, vector, Settings(**plan["settings"])) as engine, \
            (out_root / "generations.jsonl").open("x") as raw_file, \
            (out_root / "samples.jsonl").open("x") as scored_file:
        with tqdm(total=len(cohort), desc="Visible items", unit="item", file=sys.stderr) as progress:
            for item in cohort:
                item_started = time.perf_counter()
                visible = visible_fields(item["row"])
                mapping = engine.map_options(visible)
                mapping_calls += mapping["selector_forward_calls"]
                generated = []
                # All automatic/control outputs finish before evaluation metadata
                # is touched. The optional oracle arm is a separate diagnostic.
                for arm in plan["arms"]:
                    if arm == "oracle_pi":
                        continue
                    record = engine.generate(visible, arm, mapping=mapping)
                    record.update(semantic_key=item["semantic_key"], rotation=item["rotation"])
                    raw_file.write(json.dumps(record, allow_nan=False) + "\n")
                    raw_file.flush()
                    generated.append(record)
                if "oracle_pi" in plan["arms"]:
                    annotation = annotated_indices(item["row"])
                    if annotation is None:
                        raise ValueError("oracle diagnostic requires explicit evaluation annotations")
                    record = engine.generate(visible, "oracle_pi", oracle_target_index=annotation[0])
                    record.update(semantic_key=item["semantic_key"], rotation=item["rotation"])
                    raw_file.write(json.dumps(record, allow_nan=False) + "\n")
                    raw_file.flush()
                    generated.append(record)
                for record in generated:
                    record["evaluation"] = score_output(record, item["row"])
                    scored_file.write(json.dumps(record, allow_nan=False) + "\n")
                    records.append(record)
                scored_file.flush()
                item_timings.append(time.perf_counter() - item_started)
                progress.update(1)
                progress.set_postfix(forwards=engine.model.calls, refresh=False)
        actual_calls = engine.model.calls
    if actual_calls != plan["forward_budget"]["actual_total"]:
        raise RuntimeError("actual whole-run model calls differ from the frozen plan")
    result = {
        "schema_version": 1, "status": "complete", "cohort": plan["cohort"],
        "results": summarize(records, plan["cohort"]["rotations"]),
        "actual_model_forward_calls": actual_calls, "shared_selector_forward_calls": mapping_calls,
        "model_loading_elapsed_s": loading_s, "inference_elapsed_s": time.perf_counter() - inference_started,
        "mean_visible_item_elapsed_s": sum(item_timings) / len(item_timings),
        "timing_note": "paired run pays selector once per displayed prompt; standalone arm times include that shared measurement",
        "gpu_name": torch.cuda.get_device_name(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "limitations": ["raw-projection identity is experimental", "fixed gains are not evaluation-tuned",
                        "paired bootstrap resamples semantic items; rotations are not independent samples"],
    }
    write_json(out_root / "summary.json", result)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    out = args.out_root.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory must be new or empty; runs are never mixed or overwritten")
    plan, cohort = make_plan(args)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", plan)
    with (out / "cohort.jsonl").open("x") as handle:
        for item in cohort:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out / "visible_inputs.jsonl").open("x") as handle:
        for item in cohort:
            handle.write(json.dumps({key: item["row"][key] for key in VISIBLE_KEYS if key in item["row"]}, ensure_ascii=False) + "\n")
    print(f"Prepared {len(cohort)} visible items from {plan['cohort']['n_semantic_items']} semantic items; "
          f"{plan['forward_budget']['actual_total']} planned model forwards.", flush=True)
    if args.plan_only:
        write_json(out / "status.json", {"status": "planned", "inference_executed": False})
        print(f"Plan written to {out / 'plan.json'}. Local model available: {plan['model']['available']}. No inference run.")
        return 0
    write_json(out / "status.json", {"status": "running"})
    try:
        result = execute(plan, cohort, out)
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error), "note": "partial outputs retained; use a new output directory"})
        raise
    write_json(out / "status.json", {"status": "complete"})
    print(f"Finished {result['actual_model_forward_calls']} forwards. Results: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"New method stopped: {error}", file=sys.stderr)
        sys.exit(1)
