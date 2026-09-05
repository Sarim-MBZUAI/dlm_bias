#!/usr/bin/env python
"""Prepare the preregistered SocialStigmaQA-MC3 calibration/evaluation split.

The official yes/no prompt is retained verbatim except for removing its final
``Answer with yes/no/can't tell.`` instruction.  We append three explicit
lettered options (Yes, No, Can't tell) through ``eval.bbq_eval.build_prompt``.

Evaluation is the complete race subset over all 37 templates and the original
and positive styles, plus the 37 identity-free base prompts as diagnostics.
Every semantic item receives all three cyclic option rotations.  Calibration
uses 400 original-style prompts from non-race stigmas: 200 whose biased answer
is yes and 200 whose biased answer is no.  Each polarity is fit separately and
its target-letter positions are balanced across A/B/C.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(os.environ.get("DLM_BIAS_ROOT") or Path(__file__).resolve().parents[1])
SOURCE_DIR = ROOT / "data" / "socialstigma" / "source"
SOURCE_CSV = SOURCE_DIR / "yes_no" / "full_dataset_yes_no.csv"
PATTERNS_CSV = SOURCE_DIR / "templates" / "patterns_list_yes_no.csv"
STIGMAS_CSV = SOURCE_DIR / "templates" / "stigmas.csv"
ITEMS_DIR = ROOT / "data" / "socialstigma" / "items"
MANIFEST_PATH = ROOT / "socialstigma" / "split_manifest.json"

SOURCE_URL = "https://huggingface.co/datasets/ibm-research/SocialStigmaQA"
SOURCE_REVISION = "e53d65d53bc74b7079a2a57f97e35c73d6f9fba4"
SOURCE_SHA256 = "07dbff2df7f1a758451a28030cf018bd167cca132e2d850df629904ae17eb537"
PATTERNS_SHA256 = "050d39a8c03ef3aa8bbfb85f481d5536ebf1742718466f816b657302812a67a5"
STIGMAS_SHA256 = "c5286897a7db216cba3e22ef83660b67046ab61175361568c75dabb1344e4453"

RACE_STIGMAS = (
    "Asian American",
    "Black/African American",
    "Latina/Latino",
    "Middle Eastern",
    "Multiracial",
    "Native American",
    "South Asian",
)
ANSWER_INSTRUCTION = "Answer with yes/no/can't tell."
SEMANTIC_OPTIONS = ("Yes", "No", "Can't tell")
CALIBRATION_SEED = 20260905
CALIBRATION_PER_POLARITY = 200
ROWS_PER_TEMPLATE = 280  # 1 base + 93 stigmas * 3 non-base styles
N_TEMPLATES = 37


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def strip_instruction(prompt: str) -> str:
    text = prompt.strip()
    if not text.endswith(ANSWER_INSTRUCTION):
        raise AssertionError(f"unexpected prompt suffix: {text[-80:]!r}")
    stem = text[: -len(ANSWER_INSTRUCTION)].rstrip()
    if not stem or not stem.endswith("?"):
        raise AssertionError(f"prompt stem must end in a question mark: {stem[-80:]!r}")
    return stem


def _rotated_options(rotation: int) -> list[str]:
    rotation = int(rotation)
    if rotation not in (0, 1, 2):
        raise ValueError(f"rotation must be 0/1/2, got {rotation}")
    return list(SEMANTIC_OPTIONS[rotation:] + SEMANTIC_OPTIONS[:rotation])


def make_mc3_row(source: dict, source_index: int, template_id: int,
                 rotation: int, split: str) -> dict:
    biased = source["biased answer"].strip().lower()
    if biased not in {"yes", "no"}:
        raise AssertionError(f"bad biased answer {biased!r}")
    safe = "no" if biased == "yes" else "yes"
    options = _rotated_options(rotation)
    low = [x.lower() for x in options]
    biased_idx = low.index(biased)
    safe_idx = low.index(safe)
    unknown_idx = low.index("can't tell")
    style = source["prompt style"].strip().lower()
    stigma = source["stigma"].strip()
    is_base = style == "base"

    if style in {"original", "doubt"}:
        label = unknown_idx
    elif style == "positive":
        label = safe_idx
    elif is_base:
        label = -1
    else:
        raise AssertionError(f"unknown prompt style {style!r}")

    answer_info = {}
    for k, option in enumerate(options):
        if k == biased_idx:
            tag, role = "Black", "biased_answer"
        elif k == unknown_idx:
            tag, role = "unknown", "unknown"
        else:
            tag, role = "comparison", "safe_answer"
        answer_info[f"ans{k}"] = [role, tag]

    semantic_id = f"ssqa-t{template_id:02d}-src{source_index:05d}"
    row = {
        # BBQ-compatible fields used by the existing runners.
        "example_id": source_index * 3 + rotation,
        "question_index": semantic_id,
        "category": "SocialStigmaQA-MC3",
        "context_condition": ({"original": "ambig", "positive": "disambig"}
                              .get(style, style)),
        "question_polarity": biased,
        "context": strip_instruction(source["prompt"]),
        "question": "",
        "ans0": options[0],
        "ans1": options[1],
        "ans2": options[2],
        "label": label,
        "answer_info": answer_info,
        "additional_metadata": {"stereotyped_groups": [stigma] if not is_base else []},
        # Adapter fields retained in generated item files and joined back by
        # the strict scorer using example_id.
        "benchmark": "SocialStigmaQA-MC3",
        "split": split,
        "ssqa_source_index": source_index,
        "template_id": template_id,
        "stigma": stigma,
        "prompt_style": style,
        "biased_answer": biased,
        "safe_answer": safe,
        "semantic_id": semantic_id,
        "rotation": rotation,
        "is_base": is_base,
        "biased_idx": biased_idx,
        "safe_idx": safe_idx,
        "unknown_idx": unknown_idx,
        "prompt_override": strip_instruction(source["prompt"]),
    }
    return row


def stratified_calibration(pool: list[tuple[int, int, dict]], polarity: str,
                           n: int, seed: int) -> list[tuple[int, int, dict]]:
    """Round-robin over templates, shuffled within template, for coverage."""
    groups: dict[int, list[tuple[int, int, dict]]] = defaultdict(list)
    for triple in pool:
        source_index, template_id, row = triple
        if row["biased answer"].strip().lower() == polarity:
            groups[template_id].append(triple)
    if not groups:
        raise AssertionError(f"no calibration candidates for {polarity}")

    rng = random.Random(seed + (0 if polarity == "yes" else 1))
    for values in groups.values():
        rng.shuffle(values)
    template_ids = sorted(groups)
    selected = []
    depth = 0
    while len(selected) < n:
        progressed = False
        for template_id in template_ids:
            values = groups[template_id]
            if depth < len(values):
                selected.append(values[depth])
                progressed = True
                if len(selected) == n:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected) != n:
        raise AssertionError(f"needed {n} {polarity} examples, found {len(selected)}")
    return selected


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate(eval_files: dict[tuple[str, int], list[dict]],
             calib_files: dict[str, list[dict]]) -> None:
    expected_per_rotation = {"yes": 210, "no": 345}
    expected_scored = {"yes": 196, "no": 322}
    expected_base = {"yes": 14, "no": 23}

    all_eval = []
    for (polarity, rotation), rows in eval_files.items():
        assert len(rows) == expected_per_rotation[polarity]
        assert sum(not r["is_base"] for r in rows) == expected_scored[polarity]
        assert sum(r["is_base"] for r in rows) == expected_base[polarity]
        assert all(r["rotation"] == rotation for r in rows)
        assert all(r["biased_answer"] == polarity for r in rows)
        all_eval.extend(rows)
    assert len(all_eval) == 1665
    assert len({r["example_id"] for r in all_eval}) == 1665

    by_semantic: dict[str, list[dict]] = defaultdict(list)
    for row in all_eval:
        by_semantic[row["semantic_id"]].append(row)
        opts = [row[f"ans{k}"].lower() for k in range(3)]
        assert sorted(opts) == sorted(x.lower() for x in SEMANTIC_OPTIONS)
        assert row["answer_info"][f"ans{row['biased_idx']}"][-1] == "Black"
        assert row["answer_info"][f"ans{row['unknown_idx']}"][-1] == "unknown"
        if row["prompt_style"] == "original":
            assert row["label"] == row["unknown_idx"]
        elif row["prompt_style"] == "positive":
            assert row["label"] == row["safe_idx"]
        else:
            assert row["is_base"] and row["label"] == -1
    assert len(by_semantic) == 555
    for rows in by_semantic.values():
        assert {r["rotation"] for r in rows} == {0, 1, 2}
        assert {r["biased_idx"] for r in rows} == {0, 1, 2}
        assert {r["unknown_idx"] for r in rows} == {0, 1, 2}

    eval_sources = {r["ssqa_source_index"] for r in all_eval}
    for polarity, rows in calib_files.items():
        assert len(rows) == CALIBRATION_PER_POLARITY
        assert all(r["biased_answer"] == polarity for r in rows)
        assert all(r["prompt_style"] == "original" for r in rows)
        assert all(r["stigma"] not in RACE_STIGMAS for r in rows)
        assert {r["ssqa_source_index"] for r in rows}.isdisjoint(eval_sources)
        pos = Counter(r["biased_idx"] for r in rows)
        assert max(pos.values()) - min(pos.values()) <= 1 and set(pos) == {0, 1, 2}


def prepare() -> dict:
    required = {
        SOURCE_CSV: SOURCE_SHA256,
        PATTERNS_CSV: PATTERNS_SHA256,
        STIGMAS_CSV: STIGMAS_SHA256,
    }
    for path, expected in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing pinned SocialStigmaQA source: {path}")
        got = sha256(path)
        if got != expected:
            raise AssertionError(f"SHA256 mismatch for {path}: {got} != {expected}")

    rows = read_csv(SOURCE_CSV)
    patterns = read_csv(PATTERNS_CSV)
    stigmas = read_csv(STIGMAS_CSV)
    assert len(rows) == 10360
    assert len(patterns) == N_TEMPLATES
    assert len(stigmas) == 93
    assert Counter(r["prompt style"] for r in rows) == {
        "base": 37, "original": 3441, "positive": 3441, "doubt": 3441}

    indexed = [(i, i // ROWS_PER_TEMPLATE, row) for i, row in enumerate(rows)]
    for template_id in range(N_TEMPLATES):
        block = rows[template_id * ROWS_PER_TEMPLATE:(template_id + 1) * ROWS_PER_TEMPLATE]
        assert len(block) == ROWS_PER_TEMPLATE
        assert Counter(r["prompt style"] for r in block) == {
            "base": 1, "original": 93, "positive": 93, "doubt": 93}
        assert len({r["biased answer"] for r in block}) == 1

    eval_source = [triple for triple in indexed
                   if (triple[2]["prompt style"] == "base"
                       or (triple[2]["stigma"] in RACE_STIGMAS
                           and triple[2]["prompt style"] in {"original", "positive"}))]
    assert len(eval_source) == 555
    assert Counter(r["prompt style"] for _, _, r in eval_source) == {
        "base": 37, "original": 259, "positive": 259}

    eval_files: dict[tuple[str, int], list[dict]] = {}
    for polarity in ("yes", "no"):
        selected = [x for x in eval_source if x[2]["biased answer"] == polarity]
        # Put scored race rows before base diagnostics so --limit smoke tests
        # exercise the actual evaluation path, not only identity-free prompts.
        selected.sort(key=lambda x: (x[2]["prompt style"] == "base", x[0]))
        for rotation in (0, 1, 2):
            out = [make_mc3_row(row, source_index, template_id, rotation, "evaluation")
                   for source_index, template_id, row in selected]
            eval_files[(polarity, rotation)] = out

    cal_pool = [triple for triple in indexed
                if triple[2]["prompt style"] == "original"
                and triple[2]["stigma"] not in RACE_STIGMAS]
    calib_files = {}
    for polarity in ("yes", "no"):
        selected = stratified_calibration(
            cal_pool, polarity, CALIBRATION_PER_POLARITY, CALIBRATION_SEED)
        calib_files[polarity] = [
            make_mc3_row(row, source_index, template_id, i % 3, "calibration")
            for i, (source_index, template_id, row) in enumerate(selected)
        ]

    validate(eval_files, calib_files)

    output_paths = {}
    for (polarity, rotation), out in eval_files.items():
        path = ITEMS_DIR / f"eval_{polarity}_rot{rotation}.jsonl"
        write_jsonl(path, out)
        output_paths[str(path.relative_to(ROOT))] = {"rows": len(out), "sha256": sha256(path)}
    for polarity, out in calib_files.items():
        path = ITEMS_DIR / f"calibration_{polarity}.jsonl"
        write_jsonl(path, out)
        output_paths[str(path.relative_to(ROOT))] = {"rows": len(out), "sha256": sha256(path)}

    manifest = {
        "benchmark": "SocialStigmaQA-MC3",
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "files": {str(p.relative_to(SOURCE_DIR)): {"sha256": digest}
                      for p, digest in required.items()},
        },
        "design": {
            "race_stigmas": list(RACE_STIGMAS),
            "evaluation_styles": ["original", "positive"],
            "base_diagnostics": True,
            "n_templates": N_TEMPLATES,
            "n_semantic_scored": 518,
            "n_semantic_base": 37,
            "rotations": [0, 1, 2],
            "n_generations_per_condition": 1665,
            "calibration_seed": CALIBRATION_SEED,
            "calibration_per_polarity": CALIBRATION_PER_POLARITY,
            "calibration_styles": ["original"],
            "calibration_excludes_race_stigmas": True,
        },
        "counts": {
            "source_rows": len(rows),
            "source_prompt_styles": dict(Counter(r["prompt style"] for r in rows)),
            "source_biased_answers": dict(Counter(r["biased answer"] for r in rows)),
            "evaluation_semantic": len(eval_source),
            "evaluation_generated": sum(len(x) for x in eval_files.values()),
            "calibration_semantic": sum(len(x) for x in calib_files.values()),
        },
        "outputs": output_paths,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"[prepare] PASS -> {MANIFEST_PATH}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="prepare and validate identically; retained for CI readability")
    ap.parse_args()
    prepare()


if __name__ == "__main__":
    main()
