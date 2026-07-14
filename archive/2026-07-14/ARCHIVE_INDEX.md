# Archive 2026-07-14 — results + reports snapshot

Repo reset for a fresh start; all prior results/figures and result-narrative
reports moved here. Code, datasets, caches, and built steering directions
(`.pt`) remain live in the repo.

## What was archived

- **baseline/**
  - `results/` — BBQ ghostwriter runs (`.json` + `_samples.jsonl` + evidence)
  - `figs/` — `ghostwriter_vs_steering.png`
- **directional_steering/**
  - `results/` — BBQ all-layer + L14 anchored steering sweeps (add/clamp/cmom, α curves)
  - `figs/` — `anchored_steering.png`
  - `diagnostics/` — controller probes, condition runs, per-layer PID build, logs/pids, plot scripts and figs (**`layer_arrows.pt` and `perlayer_arrows.pt` kept live** in the repo, not archived)
  - `nat_proj_by_layer.json`
- **eval/**
  - `results/` — BBQ clean baseline + L14 race/color α sweeps
- **experiments/**
  - `results/` — `layersweep/`, `matrix/`, `sweep/` (with per-run `logs/`)
  - `RESULTS.md` — results narrative write-up
- **race_steering/**
  - `results/` — BBQ L14 race/black α sweeps
  - `figs/` — `black_steering.png`
- **datasets/unqover/**
  - `results/` — UnQover result outputs
- **docs/**
  - `figs/` — report figures (baseline, coherence, flips, guardrails, pipeline, etc.)
  - `dlm_bias_report.pdf`, `baselines.md`, `embedding_layer_steering.md` (findings write-ups)
- **new_idea_sanity_test/**
  - `results/` — sentiment/formality steering `.jsonl` runs (**`sentiment_direction.pt` and `formality_direction.pt` kept live** in the repo, not archived)
  - `paper_draft.md`
- **HANDOUT.md** (repo root)

## Local-only notes

- `_local_notes/` is **gitignored and local-only** — it is never pushed.
  Contains `handoff.md` and `new_idea_sanity_test_HANDOFF.md` (session notes).
