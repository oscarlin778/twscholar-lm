---
license: cc-by-nc-4.0
language:
  - zh
task_categories:
  - text-generation
tags:
  - traditional-chinese
  - academic-writing
size_categories:
  - n<1K
---

# Dataset Card — twscholar-lm training data

## Summary

A Traditional Chinese (Taiwan) academic-writing-polish dataset: colloquial
draft sentences paired with polished, journal-register rewrites, plus a
small general-conversation supplement for diversity. Built for supervised
fine-tuning of a writing-assistant model on consumer hardware.

**476 training examples**, merged from three provenance classes.

## Provenance breakdown (this is the honest part)

| Count | Share | Source | What it means |
|------:|------:|--------|---------------|
| 199 | 41.8% | `human_polished` | Claude generated a rough colloquial draft; **the human author wrote every polished (academic) target** — the actual writing-skill demonstration |
| 150 | 31.5% | `external_licensed` | Quality-filtered subset of [`yentinglin/TaiwanChat`](https://huggingface.co/datasets/yentinglin/TaiwanChat) (single-turn, 20–220 char, no code, Traditional-purity checked) for general-conversation diversity |
| 127 | 26.7% | `ai_drafted_human_reviewed` | Claude drafted both draft and polished; the human reviewed and edited each pair |

**Why this split is stated plainly**: the polished academic targets in the
199 `human_polished` pairs are the author's own writing, drawn from real
domains they work in (sports/volleyball analytics, graph neural networks,
eye-tracking, rs-fMRI/neuroimaging). The AI-assisted portions are labeled
as such rather than presented as fully hand-written.

## License

⚠️ **`cc-by-nc-4.0` (non-commercial)** — because the `external_licensed`
subset derives from TaiwanChat (cc-by-nc-4.0), the merged dataset inherits
its non-commercial restriction. The `human_polished` and
`ai_drafted_human_reviewed` portions alone would be freely relicensable;
they are kept in separate files (`data/raw/*.jsonl`) so a commercial-safe
subset can be reconstructed by excluding `external_taiwanchat.jsonl`.

## Categories

The polish pairs span 25 academic-writing situations: abstract polishing,
introduction/motivation, literature review, methods, results, figure/table
description, discussion/limitations, conclusion, contributions, hypotheses,
operational definitions, research questions, cover letters, reviewer
responses, advisor emails, acknowledgements, grant-proposal language,
colloquial→academic register shift, concision, objective-tone conversion,
abbreviation conventions, statistical reporting, and research ethics.

## Editorial principles applied to the polish targets

1. **No fabrication** — polishing changes *how* something is said, never
   *what* is claimed; no adjectives/claims/numbers absent from the draft.
2. **Register consistency** — third-person academic voice throughout
   (本研究 / 本文), not mixed with first-person 我們.
3. **Reserved words** — 顯著 (statistically significant) is used only when
   the draft actually reports a statistical test; otherwise 明顯/大幅 etc.
4. **Traditional Chinese, Taiwan usage** — e.g. 藉助/運用 not 借助.
5. **Concision over inflation** — polishing often shortens; length is not
   padded to seem formal.

## Held-out evaluation set

`data/eval/holdout_prompts.jsonl` — 50 draft prompts, verified to have
zero overlap with the training drafts, for the blind A/B evaluation (M3).

## Building the training set

```bash
python scripts/build_training_set.py   # -> data/processed/sft_train.jsonl
```

Pairs are wrapped into `messages` with a randomly chosen instruction
template (5 variants) so the model learns the task, not one fixed prompt.

## Known limitations

- **Scale** — 476 examples is a small SFT set; sufficient for demonstrating
  the pipeline and the task, not a production-grade corpus.
- **rejected/failure coverage** (for the DPO stage) is narrow — see the
  companion dpo-lab investigation.
- **Single task** — the model is a focused polisher, not a general
  assistant; the external subset only lightly offsets this.
- **AI assistance** — ~59% of examples involved Claude in drafting;
  labeled per-row via `source`.
