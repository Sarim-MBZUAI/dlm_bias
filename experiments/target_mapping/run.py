"""Plan a frozen paired pilot; execute only with the explicit ``run`` command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
PILOT_TAG = "target-mapping-pilot-20260908"


def row_key(row):
    return (str(row.get("category", "Race_ethnicity")),
            str(row["example_id"]), str(row["question_index"]))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def select_rows(rotations, limit, tag=PILOT_TAG):
    """Hash-select semantic items and preserve their complete three-rotation cluster."""
    if len(rotations) != 3:
        raise ValueError("exactly three rotation files are required")
    indexed = []
    for rows in rotations:
        index = {row_key(row): row for row in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate semantic item key within a rotation")
        indexed.append(index)
    keys = set(indexed[0])
    if not keys or any(set(index) != keys for index in indexed[1:]):
        raise ValueError("rotation files must contain the same nonempty semantic item set")
    for key in keys:
        base = indexed[0][key]
        for rot in (1, 2):
            row = indexed[rot][key]
            for field in ("context", "question"):
                if row.get(field) != base.get(field):
                    raise ValueError(f"changed {field} across rotations: {key}")
            for j in range(3):
                source = (j - rot) % 3
                if (row[f"ans{j}"] != base[f"ans{source}"]
                        or row["answer_info"][f"ans{j}"] != base["answer_info"][f"ans{source}"]):
                    raise ValueError(f"incorrect option/annotation rotation: {key}")
    if limit < 1 or limit > len(keys):
        raise ValueError(f"limit must be between 1 and {len(keys)}")
    ordered = sorted(keys, key=lambda key: hashlib.sha256(
        (tag + json.dumps(key, separators=(",", ":"))).encode()).hexdigest())[:limit]
    return ordered, [[index[key] for key in ordered] for index in indexed]


def condition_commands(code_root, python, items, out_root, arrows, secondary=False):
    conditions = [("clean", "oracle", "base"), ("oracle_pi", "oracle", "PI"),
                  ("direction_pi", "direction", "PI")]
    jobs = []
    for name, mapping, cond in conditions:
        for rot, item_path in enumerate(items):
            out = out_root / name / f"rot{rot}"
            argv = [python, str(code_root / "steering/denoise_pid.py"),
                    "--cond", cond, "--target-mapping", mapping,
                    "--kp", "3", "--ki", "0.1", "--kd", "1", "--amax", "6",
                    "--setpoint", "0.9", "--sensor-case", "upper", "--steps", "64",
                    "--items", str(item_path), "--out-dir", str(out), "--tag", name]
            jobs.append({"condition": name, "rotation": rot, "argv": argv,
                         "samples": str(out / f"cond_{name}_samples.jsonl")})
    for name, alpha in [("normal_a4", "4")] + ([("normal_a3p28", "3.28")] if secondary else []):
        for rot, item_path in enumerate(items):
            out = out_root / name / f"rot{rot}"
            stem = "normal_a" + alpha.replace(".", "p")
            argv = [python, str(code_root / "steering/pid_steer.py"), "--mode", "normal",
                    "--source-layer", "14", "--alpha", alpha, "--arrows", str(arrows),
                    "--gen-length", "32", "--block-length", "32", "--steps", "64",
                    "--items", str(item_path), "--out-dir", str(out), "--tag-prefix", "normal"]
            jobs.append({"condition": name, "rotation": rot, "argv": argv,
                         "samples": str(out / f"cond_{stem}_samples.jsonl")})
    return jobs


def create_plan(data_root, out_root, limit=100, python=sys.executable,
                code_root=CODE_ROOT, secondary=False):
    data_root, out_root, code_root = map(lambda p: Path(p).resolve(), (data_root, out_root, code_root))
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError("use a new or empty output directory; existing runs are never overwritten")
    sources = [data_root / "results/balanced" / f"_sweep400_rot{r}.jsonl" for r in range(3)]
    keys, subsets = select_rows([read_rows(p) for p in sources], limit)
    arrows = Path(os.environ.get("DLM_ARROWS_PATH", data_root / "steering/arrows.pt")).resolve()
    if not arrows.is_file():
        raise FileNotFoundError(arrows)
    # The legacy normal runner imports eval from data_root. Require byte-identical
    # samplers, then freeze both locations as well as the modified controller.
    code_paths = [code_root / "steering" / name for name in
                  ("denoise_pid.py", "target_selector.py", "pid_steer.py")]
    for name in ("bbq_eval.py", "bias_metrics.py"):
        local, legacy = code_root / "eval" / name, data_root / "eval" / name
        if sha256(local) != sha256(legacy):
            raise ValueError(f"code/data roots contain different eval/{name}; align samplers before planning")
        code_paths.extend((local, legacy))
    code_paths.extend((Path(__file__).resolve(), Path(__file__).with_name("analyze.py").resolve()))
    inputs = out_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    items = []
    for rot, rows in enumerate(subsets):
        path = inputs / f"items_rot{rot}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        items.append(path)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    plan = {
        "schema_version": 1, "pilot_tag": PILOT_TAG, "code_root": str(code_root),
        "code_revision": revision, "data_root": str(data_root), "out_root": str(out_root),
        "code_files": [{"path": str(p), "sha256": sha256(p)} for p in sorted(set(code_paths))],
        "n_semantic_items": len(keys), "selected_keys": keys,
        "cohort_status": "historically exposed feasibility pilot; not a fresh holdout",
        "sources": [{"path": str(p), "sha256": sha256(p)} for p in sources],
        "items": [{"path": str(p), "sha256": sha256(p)} for p in items],
        "arrows": {"path": str(arrows), "sha256": sha256(arrows)},
        "frozen_settings": {"kp": 3, "ki": 0.1, "kd_PI": 0, "setpoint": 0.9,
                            "amax": 6, "steps": 64, "gen_length": 32, "sensor_case": "upper",
                            "selector": "raw direction projection; no thresholds or eval tuning"},
        "jobs": condition_commands(code_root, python, items, out_root, arrows, secondary),
    }
    path = out_root / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path, plan


def execute_plan(path):
    plan = json.loads(Path(path).read_text())
    for artifact in plan["sources"] + plan["items"] + [plan["arrows"]] + plan["code_files"]:
        if sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"artifact changed after planning: {artifact['path']}")
    for job in plan["jobs"]:
        if Path(job["samples"]).parent.exists() and any(Path(job["samples"]).parent.iterdir()):
            raise ValueError("run directory already contains output; create a new plan rather than mixing runs")
    env = dict(os.environ, DLM_BIAS_ROOT=plan["data_root"], DLM_ARROWS_PATH=plan["arrows"]["path"])
    for job in plan["jobs"]:
        subprocess.run(job["argv"], cwd=plan["code_root"], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="write cohort/commands only; no model or GPU is loaded")
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get("DLM_BIAS_ROOT", CODE_ROOT)))
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--secondary-alpha3p28", action="store_true")
    p = sub.add_parser("run", help="execute a saved plan on an already allocated GPU")
    p.add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        path, plan = create_plan(args.data_root, args.out_root, args.limit, args.python,
                                 secondary=args.secondary_alpha3p28)
        print(f"Wrote {path}: {plan['n_semantic_items']} semantic items, {len(plan['jobs'])} jobs. No inference run.")
    else:
        execute_plan(args.plan)


if __name__ == "__main__":
    main()
