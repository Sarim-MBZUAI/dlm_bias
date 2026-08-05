#!/usr/bin/env python
"""balanced_all/make_rotations.py -- generalized position-balance rotation builder.

Generalization of eval/balanced/make_rotations.py (which is hardwired to the
Black _sweep400.jsonl): takes ANY BBQ items jsonl and writes the 3 cyclic
rotations <stem>_rot{0,1,2}.jsonl into --out-dir.

The content-only remap is REUSED BY IMPORT from eval/balanced/make_rotations.py
(single source of truth, not a port):
    new_ans[j]         = old_ans[(j - r) % 3]
    new answer_info[j] = old answer_info[(j - r) % 3]
    new label          = (old_label + r) % 3
rot0 is byte-identical to the original on the rotated fields; all other fields
are preserved verbatim. Across r=0,1,2 every option visits A, B, C exactly once.

Usage:
    python balanced_all/make_rotations.py \
        --items data/bbq_items/_sweep400_arab.jsonl \
        --out-dir results/balanced_all/rotations [--stem _sweep400_arab]
"""
import argparse
import importlib.util
import json
import os

ROOT = os.environ.get("DLM_BIAS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_reference_rotate():
    """Import rotate_row from eval/balanced/make_rotations.py (reuse, not port)."""
    p = os.path.join(ROOT, "eval", "balanced", "make_rotations.py")
    spec = importlib.util.spec_from_file_location("balanced_make_rotations", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rotate_row


rotate_row = _load_reference_rotate()


def write_rotations(items_path, out_dir, stem=None):
    rows = [json.loads(l) for l in open(items_path) if l.strip()]
    stem = stem or os.path.splitext(os.path.basename(items_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for r in range(3):
        out = os.path.join(out_dir, f"{stem}_rot{r}.jsonl")
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(rotate_row(row, r)) + "\n")
        print(f"wrote {out}  ({len(rows)} rows, rotation r={r})")
        outs.append(out)
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True, help="source BBQ items jsonl")
    ap.add_argument("--out-dir", required=True, help="directory for <stem>_rot{0,1,2}.jsonl")
    ap.add_argument("--stem", default=None,
                    help="output stem (default: items basename sans .jsonl)")
    args = ap.parse_args()
    write_rotations(args.items, args.out_dir, args.stem)


if __name__ == "__main__":
    main()
