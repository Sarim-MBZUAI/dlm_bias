#!/usr/bin/env python3
"""CPU-only self-test of the scoring and reporting scripts on synthetic data.
No model, no GPU. Run before the experiment to make sure the offline tooling
works, and after any edit to score.py / compare_repro.py / check_setpoints.py /
make_report.py.

    python results/ablation_setpoint/selftest.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_rows(n, outputs, alpha):
    """n rows; option B is the target, C is unknown, A the comparator."""
    rows = []
    for i in range(n):
        rows.append({
            "example_id": 1000 + i, "black_idx": 1, "unk_idx": 2, "target_letter": "B",
            "model_output": outputs[i % len(outputs)],
            "alpha_traj": [alpha] * 64,
        })
    return rows


def make_cond_json(condition, setpoint, n, kd=0.0):
    return {"condition": condition, "gains": {"Kp": 3.0, "Ki": 0.1, "Kd": kd},
            "setpoint": setpoint, "alpha_max": 6.0, "alpha_min": 0.0,
            "sensor_case": "upper", "target_mapping": "oracle",
            "steps": 64, "gen_length": 32, "block_length": 32, "n": n}


def run(*args, check=True, **kw):
    p = subprocess.run([PY, *args], capture_output=True, text=True, **kw)
    if check and p.returncode != 0:
        print(p.stdout); print(p.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {' '.join(args)}")
    return p


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tier = os.path.join(tmp, "tier1")
        ref = os.path.join(tmp, "results_balanced")
        n = 10
        # Condition outputs -> strict classes:
        #   base:   "B" x4 target, "A" x4 comparator, "C" x2 abstain            -> gap 0
        #   s0p9:   "B." x7 target, "A" x2 comparator, "blah" x1 invalid         -> gap +50
        #   s0p5:   "b" x5 target, "A: A: A" x5 comparator                       -> gap 0
        conds = {
            "dpid_base": (["B", "A", "B", "A", "C", "B", "A", "B", "A", "C"], 0.0, "base", 0.9),
            "dpid_PI_s0p9": (["B.", "B.", "B.", "B.", "B.", "B.", "B.", "A", "A", "blah"], 6.0, "PI", 0.9),
            "dpid_PI_s0p5": (["b", "A: A: A"] * 5, 2.0, "PI", 0.5),
        }
        for r in range(3):
            for tag, (outs, alpha, cond, sp) in conds.items():
                write_jsonl(os.path.join(tier, f"rot{r}", f"cond_{tag}_samples.jsonl"), make_rows(n, outs, alpha))
                json.dump(make_cond_json(cond, sp, n),
                          open(os.path.join(tier, f"rot{r}", f"cond_{tag}.json"), "w"))
            # Published reference: 400 rows, first 10 share ids with the rerun; 8/10 identical.
            pub = make_rows(400, ["B.", "B.", "B.", "B.", "B.", "B.", "B.", "A", "A", "blah"], 6.0)
            pub[7]["model_output"] = "A."       # differs textually, same class
            pub[9]["model_output"] = "B"        # differs textually and in class
            write_jsonl(os.path.join(ref, f"rot{r}", "cond_dpid_PI_samples.jsonl"), pub)

        # check_setpoints
        p = run(os.path.join(HERE, "check_setpoints.py"), tier, str(n))
        assert "OK" in p.stdout, p.stdout
        # inject a mismatch and make sure it is caught
        bad = json.load(open(os.path.join(tier, "rot0", "cond_dpid_PI_s0p5.json")))
        bad["setpoint"] = 0.7
        json.dump(bad, open(os.path.join(tier, "rot0", "cond_dpid_PI_s0p5.json"), "w"))
        p = run(os.path.join(HERE, "check_setpoints.py"), tier, str(n), check=False)
        assert p.returncode == 1 and "tag says 0.5" in p.stdout, p.stdout
        bad["setpoint"] = 0.5
        bad["gains"]["Kp"] = 6.0
        json.dump(bad, open(os.path.join(tier, "rot0", "cond_dpid_PI_s0p5.json"), "w"))
        p = run(os.path.join(HERE, "check_setpoints.py"), tier, str(n), check=False)
        assert p.returncode == 1 and "kp=6.0, expected 3" in p.stdout, p.stdout
        bad["gains"]["Kp"] = 3.0
        json.dump(bad, open(os.path.join(tier, "rot0", "cond_dpid_PI_s0p5.json"), "w"))

        # score
        run(os.path.join(HERE, "score.py"), tier)
        s = json.load(open(os.path.join(tier, "summary.json")))
        assert set(s) == set(conds), s.keys()
        assert s["dpid_base"]["n"] == 30 and s["dpid_base"]["gap_pp"] == 0.0, s["dpid_base"]
        assert s["dpid_PI_s0p9"]["gap_pp"] == 50.0 and s["dpid_PI_s0p9"]["invalid"] == 10.0, s["dpid_PI_s0p9"]
        assert s["dpid_PI_s0p9"]["share_ever_at_limit"] == 1.0
        assert s["dpid_PI_s0p9"]["mean_command_steps_1_32"] == 6.0
        assert s["dpid_PI_s0p5"]["gap_pp"] == 0.0 and s["dpid_PI_s0p5"]["target"] == 50.0, s["dpid_PI_s0p5"]
        assert s["dpid_PI_s0p5"]["gap_minus_s0p9_pp"] == -50.0
        assert "gap_minus_s0p9_pp" not in s["dpid_PI_s0p9"]
        assert "mean_command_steps_1_32" not in s["dpid_base"]
        lo, hi = s["dpid_PI_s0p5"]["gap_minus_s0p9_ci95"]
        assert lo <= -50.0 <= hi, (lo, hi)

        # compare_repro
        out = os.path.join(tmp, "repro.json")
        run(os.path.join(HERE, "compare_repro.py"), "--tier-dir", tier, "--ref-dir", ref,
            "--limit", "10", "--out", out)
        rp = json.load(open(out))
        assert rp["pooled"]["n"] == 30
        assert rp["pooled"]["identical_output_frac"] == 0.8, rp["pooled"]
        assert rp["pooled"]["strict_class_agreement_frac"] == 0.9, rp["pooled"]
        assert rp["rot0"]["n_differing"] == 2
        # rerun ids outside the reference slice must be rejected
        p = run(os.path.join(HERE, "compare_repro.py"), "--tier-dir", tier, "--ref-dir", ref,
                "--limit", "5", "--out", out, check=False)
        assert p.returncode != 0 and "not in the first 5" in p.stderr, p.stderr

        # make_report (env.json + timings present, tier2 absent)
        json.dump({"commit": "deadbeef", "gpu": "Synthetic GPU", "python": "3.11", "torch": "x",
                   "transformers": "4.46.2", "numpy": "y", "arrows_path": "steering/arrows.pt",
                   "arrows_sha256": "0" * 64, "fixed_args": "PI", "hostname": "h",
                   "slurm_job_id": None, "modified_tracked_files": 0},
                  open(os.path.join(tmp, "env.json"), "w"))
        os.replace(out, os.path.join(tmp, "repro.json"))
        with open(os.path.join(tmp, "timings.tsv"), "w") as f:
            f.write("tier\trot\ttag\tstart_utc\telapsed_s\texit_code\tgpu\n")
            f.write("tier1\t0\tdpid_PI_s0p9\t2026-01-01T00:00:00Z\t201\t0\tSynthetic GPU\n")
        run(os.path.join(HERE, "make_report.py"), "--dir", tmp)
        rep = open(os.path.join(tmp, "REPORT.md")).read()
        assert "| 0.5 |" in rep and "| 0.9 |" in rep and "base (no steering)" in rep, rep
        assert "80.0% of 30 rows" in rep, rep
        assert "tier2/summary.json not found" in rep
        assert "None recorded in failures.log." in rep
        assert rep.count("cond_dpid_PI_s0p9_samples.jsonl") == 3, rep
        assert "Synthetic GPU" in rep

    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
