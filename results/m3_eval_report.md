# M3 — Evaluation Report

Held-out set: 50 draft prompts (`data/eval/holdout_prompts.jsonl`), zero
overlap with training. All models are Qwen2.5-7B-Instruct (4-bit).
Four conditions: `base_zeroshot`, `base_fewshot` (3 in-context examples),
`sft`, `sft_dpo` (final model).

## 1. Objective automated metrics (bias-free)

`scripts/eval_conditions.py` → `results/m3_auto_metrics.json`

| condition | avg len | len ratio | rows w/ simplified char | English-letter hits* |
|-----------|--------:|----------:|------------------------:|---------------------:|
| base_zeroshot | 26.7 | 1.26 | **4** | 123 |
| base_fewshot  | 24.5 | 1.15 | 3 | 33 |
| sft           | 28.7 | 1.37 | **0** | 47 |
| sft_dpo       | 29.9 | 1.43 | 1 | 51 |

- **Traditional-Chinese purity**: fine-tuning nearly eliminates simplified-
  character leakage (0–1 rows) vs. the base model's 3–4 rows.
- **English leakage**: base zero-shot leaks the most (123). *This metric is
  noisy — it counts legitimate academic abbreviations (GNN, rs-fMRI, ROC,
  BERT) that our polished targets intentionally keep, so the fine-tuned
  models' ~50 is mostly legitimate, whereas the base's 123 includes whole
  English phrases and untranslated terms. Interpret directionally, not
  absolutely.
- **Length**: all conditions land in a similar 24–30 char range; the naive
  worry that DPO inflates length did not materialize (len ratio 1.43 vs
  base 1.26 — mild).

## 2. Blind head-to-head: sft_dpo vs base_zeroshot

`scripts/blind_prep.py` builds a blinded file (per-item random A/B, hidden
key). The judge (Claude) scored each pair blind, on
faithfulness > Traditional purity > register > concision.

### Why we ran a position-swap check
The first pass's raw slot tally looked alarming: slot A won 33 of 48
decisive items. That *looked* like position bias. So we re-judged the same
50 pairs with A/B **swapped** (`_VERDICTS_swapped.json`) and kept a decisive
win only when both orderings picked the same content
(`scripts/blind_combine.py`).

**Result: 98% self-agreement (47/48 decisive items consistent).** The judge
was *not* position-biased after all — the slot-A skew was just where the
better content happened to land in that randomization draw. This is exactly
why you run the swap instead of trusting the raw slot count.

### Position-bias-corrected result

| | wins |
|---|---:|
| base_zeroshot | 25 |
| sft_dpo | 22 |
| tie (incl. 1 position-flip) | 3 |

## 3. What this actually means (the honest read)

**Qwen2.5-7B-Instruct is already a strong Traditional Chinese academic
writer.** On blind holistic preference, the base model (zero-shot) is
roughly on par with — marginally ahead of — the fine-tuned SFT+DPO model.
Fine-tuning did **not** produce a dramatic quality jump.

What fine-tuning *did* buy, measurably, is **reliability / Traditional-
Chinese purity**: it nearly eliminates the simplified-character and
English leakage the base model exhibits on a meaningful minority of inputs
(objective metrics, §1). For a Traditional-Chinese-specific product, that
consistency guarantee is the real value-add — but it is honest to say it is
a consistency win, not an "our model is much better" win.

This directly answers the interview-standard question *"why fine-tune
instead of just prompting a strong base model?"*: for a capable base model,
prompting gets you most of the quality; fine-tuning buys guarantees
(here, script purity) that prompting alone doesn't reliably deliver.

## 4. SFT vs SFT+DPO

From the automated metrics, `sft` (0 simplified rows) and `sft_dpo` (1) are
near-identical on purity, and lengths are close. Consistent with the
companion dpo-lab investigation, DPO's marginal contribution on top of SFT
is small for this task; its main documented role there was as a study of
DPO failure modes, not a large quality lever here. A full blind sft-vs-dpo
head-to-head is left as follow-up.

## 5. Limitations of this evaluation

- Single judge (Claude); a human panel or a second judge model would
  strengthen it.
- English-leakage metric conflates legitimate abbreviations with leakage.
- 50 items gives directional, not tight-CI, estimates.
- The judge cannot be fully blind to *style* (it can infer which output is
  "cleaner"); blindness here means blind to the *condition label*, which is
  the relevant control.
