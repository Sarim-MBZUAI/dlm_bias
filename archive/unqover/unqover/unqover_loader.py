#!/usr/bin/env python
"""Convert an official UNQOVER `*.source.json` into the jsonl `--items` format
that our harness (unqover_eval.py) consumes.

OFFICIAL SOURCE FORMAT  (allenai/unqover @ 3e47969, generate_underspecified_
templates.py -> compose_subj_act_bij_json). Top-level dict keyed by:

    "{scluster0}|{scluster1}|{subj0}|{subj1}|{tid}|{act_cluster}|{obj0}|{obj1}"

  value = {
    "context": "<paragraph naming subj0 then subj1>",
    "q0": {"question": "<attribute a>",  "ans0": {"text": subj0}, "ans1": {"text": subj1}},
    "q1": {"question": "<neg. attr. a>", "ans0": {"text": subj0}, "ans1": {"text": subj1}},
  }

  * subj0 is named FIRST in the context, subj1 SECOND  (subject POSITION).
  * The dataset contains BOTH orderings as separate keys: (subj0,subj1) and
    (subj1,subj0). One "instance" = an unordered subject pair + template + attr.
  * q0 asks the (negative) attribute a; q1 asks its negation a-bar (antonym).

FLATTENING.  We emit ONE jsonl line per generation the model must run, i.e. one
per (key, question) -> 2 lines per key, 4 lines per unordered instance:
    { order in (subj0-first, subj1-first) } x { q0=a, q1=a-bar }.
The 4 lines sharing an `instance_id` are exactly the terms UNQOVER's metric
B(.)/C(.) averages over (both positions x both polarities); the metric
(unqover_metric.py) regroups by instance_id, so DO NOT split a group across runs.

Output item fields (compatible with a two-choice, NO-"unknown" prompt builder):
    id, instance_id, uqid(=source key), bias_class,
    context, question, qid("q0"/"q1"), polarity("a"/"neg_a"),
    subj0, subj1        # in context order: choice A = subj0, choice B = subj1
    choices=[subj0,subj1],
    s_cluster0, s_cluster1, tid, act_cluster, obj0, obj1

Sampling is by INSTANCE (keeps quadruples intact): --limit N keeps N random
unordered instances (seeded) and emits all 4 items each.

CPU only.  Example:
    python unqover/unqover_loader.py \
      --source data/unqover/generated/ethnicity.source.json \
      --out data/unqover/ethnicity.items.jsonl --limit 2000 --seed 42
"""
import argparse
import json
import os
import random
from collections import defaultdict


def parse_key(key):
    """8-field composite key -> dict. Subjects/clusters never contain '|'."""
    scl0, scl1, subj0, subj1, tid, act_cluster, obj0, obj1 = key.split("|")
    return dict(s_cluster0=scl0, s_cluster1=scl1, subj0=subj0, subj1=subj1,
                tid=tid, act_cluster=act_cluster, obj0=obj0, obj1=obj1)


def instance_id(f):
    """Order-independent id for the unordered (pair, template, attribute)."""
    pair = "::".join(sorted((f["subj0"], f["subj1"])))
    return f"{f['tid']}|{f['act_cluster']}|{f['obj0']}|{f['obj1']}|{pair}"


def bias_class_of(source_path):
    base = os.path.basename(source_path).lower()
    for c in ("ethnicity", "religion", "country", "gender", "nationality"):
        if c in base:
            return "nationality" if c == "country" else c
    return "unknown"


def load_items(source_path):
    with open(source_path) as fh:
        data = json.load(fh)
    bclass = bias_class_of(source_path)
    # group keys by instance so sampling keeps complete quadruples
    by_inst = defaultdict(list)
    for key in data:
        f = parse_key(key)
        by_inst[instance_id(f)].append((key, f))
    return data, bclass, by_inst


def emit(data, bclass, keys, iid):
    """Yield the 2 items (q0,q1) for one source key."""
    for key, f in keys:
        entry = data[key]
        for qid, polarity in (("q0", "a"), ("q1", "neg_a")):
            q = entry[qid]
            yield {
                "id": f"{key}#{qid}",
                "instance_id": iid,
                "uqid": key,
                "bias_class": bclass,
                "context": entry["context"],
                "question": q["question"],
                "qid": qid,
                "polarity": polarity,
                "subj0": f["subj0"],
                "subj1": f["subj1"],
                "choices": [f["subj0"], f["subj1"]],
                "s_cluster0": f["s_cluster0"],
                "s_cluster1": f["s_cluster1"],
                "tid": f["tid"],
                "act_cluster": f["act_cluster"],
                "obj0": f["obj0"],
                "obj1": f["obj1"],
            }


def main():
    ap = argparse.ArgumentParser(description="UNQOVER source.json -> --items jsonl.")
    ap.add_argument("--source", required=True, help="path to a *.source.json")
    ap.add_argument("--out", required=True, help="output items jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep this many random INSTANCES (each = 4 items). "
                         "Default: all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data, bclass, by_inst = load_items(args.source)
    insts = list(by_inst)
    # keep only complete instances (both orderings present) for clean metrics
    complete = [i for i in insts if len(by_inst[i]) == 2]
    dropped = len(insts) - len(complete)
    random.Random(args.seed).shuffle(complete)
    kept = complete[:args.limit] if args.limit is not None else complete

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w") as fh:
        for iid in kept:
            for item in emit(data, bclass, by_inst[iid], iid):
                fh.write(json.dumps(item) + "\n")
                n += 1
    print(f"class={bclass}  instances(total/complete/kept)="
          f"{len(insts)}/{len(complete)}/{len(kept)}  "
          f"(dropped incomplete={dropped})")
    print(f"wrote {n} items ({n // 4} quadruples) -> {args.out}")


if __name__ == "__main__":
    main()
