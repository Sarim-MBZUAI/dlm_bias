#!/usr/bin/env python
"""Score the generated runs and print the convergence-vs-uniform table.

attribute strength:
  sentiment -> P(negative)  (distilbert-sst2)      [we steered toward negative]
  formality -> P(formal)    (s-nlp/roberta-base-formality-ranker) [toward formal]

Reads new_idea_sanity_test/results/{attr}_{cond}.jsonl for cond in
none,uniform,decay,conv. Reports per condition: attribute strength, mean nfe
(#forward passes = the efficiency axis), mean length (crude fluency guard).
"""
import argparse, json, os, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SCORER = {"sentiment": ("distilbert-base-uncased-finetuned-sst-2-english", "NEG"),
          "formality": ("s-nlp/roberta-base-formality-ranker", "FORMAL")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True, choices=["sentiment", "formality"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    name, want = SCORER[args.attr]
    print(f"loading scorer {name} ...", flush=True)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(args.device).eval()
    tk = AutoTokenizer.from_pretrained(name)
    # pick the target-class index whose label matches `want`
    idx = None
    for i, l in m.config.id2label.items():
        if l.upper().startswith(want[:3]): idx = int(i)
    if idx is None: raise SystemExit(f"no label ~{want} in {m.config.id2label}")

    @torch.no_grad()
    def score(texts):
        enc = tk(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(args.device)
        return torch.softmax(m(**enc).logits, -1)[:, idx].tolist()

    print(f"\n{args.attr}: attribute strength = P({m.config.id2label[idx]})")
    print(f"{'cond':>8}{'attr':>9}{'nfe':>8}{'len':>7}{'n':>5}")
    rows = {}
    for cond in ["none", "uniform", "decay", "conv", "warmup", "clamp", "clampconf", "amom", "cmom",
                 "sphere", "sadi", "sadidir"]:
        fp = os.path.join(RES, f"{args.attr}_{cond}.jsonl")
        if not os.path.exists(fp):
            print(f"{cond:>8}   (missing)"); continue
        r = [json.loads(l) for l in open(fp) if l.strip()]
        texts = [x["text"] if x["text"].strip() else " " for x in r]
        s = score(texts)
        attr = sum(s)/len(s); nfe = sum(x["nfe"] for x in r)/len(r)
        ln = sum(len(x["text"].split()) for x in r)/len(r)
        rows[cond] = (attr, nfe, ln)
        print(f"{cond:>8}{attr:>9.3f}{nfe:>8.1f}{ln:>7.1f}{len(r):>5}")

    if "uniform" in rows and "conv" in rows:
        (au, nu, _), (ac, nc, lc) = rows["uniform"], rows["conv"]
        print(f"\nconv vs uniform: attr {ac:.3f} vs {au:.3f} | "
              f"nfe {nc:.1f} vs {nu:.1f} ({100*(1-nc/nu):.0f}% fewer steps) | conv len {lc:.1f}")
        print("READ: convergence-aware wins ONLY if it keeps attr≈uniform AND len (fluency) "
              "at fewer nfe. If conv attr/len collapse, early-commit dumped the undecoded tail.")

if __name__ == "__main__":
    main()
