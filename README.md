# twscholar-lm

**A fully-documented, end-to-end journey of building a Traditional Chinese
academic-writing assistant on a single consumer GPU (RTX 4070, 12GB) — from
hand-curated data through SFT and DPO, rigorous blind evaluation, to a
deployed demo — including every failure and dead end along the way.**

> 🚧 Work in progress. This README will be completed to tech-report quality
> before release (see `FLAGSHIP_PROJECT_PLAN.md` in the parent LLM-lab repo).

## What this is

A Traditional Chinese (Taiwan) academic-writing polisher: give it a rough,
colloquial draft sentence, get back a concise, formal version that reads
like it belongs in a journal paper. Built end-to-end on consumer hardware.

## Why it's different

- **Traditional Chinese academic writing** — a genuinely under-served niche
  (verified: no equivalent dataset exists on the Hub).
- **Full pipeline on one 12GB GPU** — real, reproducible numbers for people
  without an A100.
- **Honest failure log** — a "Debugging Alignment" chapter documenting the
  DPO investigation (β-metric traps, a fix that made things *worse*, a
  hypothesis proven wrong) that most tutorial repos never show.
- **Full scale comparison** — 0.5B full fine-tune vs 1.5B LoRA vs 7B QLoRA.

## Status (milestones)

- [x] M0 — repo skeleton + GitHub online
- [x] M1 — dataset v1 (476 examples, provenance-labeled + 50 held-out; see [DATASET_CARD](docs/DATASET_CARD.md))
- [x] M2 — training: 0.5B full-FT / 1.5B LoRA / 7B QLoRA (SFT) + DPO on 7B (final model: 0 script leakage on held-out)
- [x] M3 — blind randomized evaluation w/ position-swap correction ([report](results/m3_eval_report.md))
- [ ] M4 — release (HF Hub adapters + dataset + README + demo)
- [ ] M5 — interview-readiness (whiteboard derivations)

## Hardware

Single NVIDIA RTX 4070 (12GB VRAM, WDDM). Measured footprints:
7B 4-bit inference 5.6GB · 7B QLoRA SFT 10.25GB · 7B QLoRA DPO 9.03GB.
