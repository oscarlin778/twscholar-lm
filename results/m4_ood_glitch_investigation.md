# M4 — OOD Sentence-Type Glitch: Discovery, Investigation, and Fix

Discovered live during manual demo testing (not by the eval suite), which
is itself worth noting: the M3 blind evaluation's 50 held-out prompts are
almost entirely declarative statements (matching the training distribution),
so it never surfaced this failure mode. **A human clicking around found a
bug the automated eval missed** — a reminder that held-out sets inherit the
same distributional blind spots as the training data unless deliberately
designed not to.

## The bug

Input: `所以我們目前的進度到底怎樣` (an informal status-check question).

Output from the then-current model (`dpo-qlora-7b`, SFT 3 epochs + DPO
1 epoch): `本研究目前之主要成果與技術驗證已完cheng，具體階段性成果詳見...`

"完成" (complete) is garbled into "完" + the pinyin romanization "cheng" —
a token-level generation glitch, not a script-consistency issue (Finding 7
in the companion dpo-lab investigation was about simplified-character
leakage; this is a different symptom).

## Step 1: was this a known bug already sitting in our own eval data?

Yes. Grepping `data/eval/conditions_outputs.jsonl` (M3's raw generations)
turned up the *exact same glitch*, on a different draft, in both the `sft`
and `sft_dpo` conditions:

```
draft: 老師,我論文的第三章寫好了,想請你幫我看看。
sft:     教授您好：學生已完cheng論文第三章初稿，懇請您抽空指導。
sft_dpo: 教授您好：學生已完cheng論文第三章初稿，懇請您抽空指導。
base:    尊敬的指導老师，我已经完成了論文第三章的撰寫...  (no glitch)
```

This was sitting in the M3 data the whole time and wasn't flagged as a
distinct finding during that writeup — a miss worth being honest about.
Critically, **`sft` alone (no DPO) already has the glitch** — this rules
out DPO as the cause and points to the SFT stage.

## Step 2: is this a broad "informal question/request" problem, or narrow?

Training-data audit: sentences resembling questions or informal requests
make up only **~2% of the 326 draft/polished pairs** (6 question-like, 4
request-like) — a training blind spot, consistent with an OOD-input
hypothesis.

Tested the hypothesis broadly: 5 new declarative controls, 5 questions, 5
informal requests (none seen in training) on the then-current model.

**Result: 0/15 glitches.** The broad "whole category is unstable"
hypothesis is **wrong** — none of 15 constructed question/request examples
reproduced it. The failure is narrower and more input-specific than
"informal register in general."

## Step 3: checkpoint-intensity diagnostic (reusing the morning's method)

Same technique that found Finding 7 in dpo-lab: compare the two known
trigger prompts across saved checkpoints of increasing training exposure.

| Checkpoint | "老師,我論文…" | "所以我們目前…" |
|---|---|---|
| SFT checkpoint-30 (~epoch 1) | ✅ clean | ✅ clean |
| SFT checkpoint-60 (~epoch 2) | 🔴 glitch | ✅ clean |
| SFT checkpoint-90 (epoch 3, final) | 🔴 glitch | ✅ clean |
| + DPO (1 epoch on top of epoch-3 SFT) | 🔴 glitch | 🔴 glitch |

Glitch count is monotonic in training exposure: 0/2 → 1/2 → 1/2 → 2/2.
**Second independent confirmation of the same root cause as Finding 7**
(narrow-dataset LoRA fine-tuning destabilizes properties never directly
targeted by training, and instability compounds with more steps) — this
time the symptom is word-level pinyin insertion on rare sentence
structures, not script consistency.

## Step 4: does reducing SFT to 1 epoch actually fix it — and at what cost?

Unlike DPO (where 1 epoch already reached near-zero loss — cutting it was
a free lunch), **SFT's loss at 3 epochs was still improving** (0.81 final
vs. 1.47 at 1 epoch; token accuracy 0.93 vs. 0.68). This is a real
apples-to-oranges risk: lower loss numbers don't guarantee the *generated
text* is worse, but it needed checking rather than assuming.

Retrained: SFT 1 epoch (`sft-qlora-7b-epoch1`) → DPO 1 epoch on top
(`dpo-qlora-7b-v2`). Validation:

| Check | Result |
|---|---|
| 2 known glitch triggers | ✅ both clean |
| 15-item OOD battery (control/question/request) | ✅ 0/15 glitches |
| Full 50-item held-out: glitch hits | ✅ 0/50 |
| Full 50-item held-out: simplified-char hits | ✅ 0/50 (was 1/50 for the old 3-epoch+DPO model) |
| Full 50-item held-out: avg length | 30.7 (old: 29.9 — no inflation) |
| Qualitative spot-check (10 items, manual read) | Comparable fluency/faithfulness to the old model, no fabrication observed |

**The fix works, and the training-loss concern didn't materialize as a
visible quality regression** on this eval battery. `dpo-qlora-7b-v2`
(built on `sft-qlora-7b-epoch1`) is promoted to the project's production
model, superseding `dpo-qlora-7b` (SFT 3 epochs + DPO 1 epoch).

## Updated recommendation

**Both SFT and DPO should use 1 epoch** for this dataset size (~450
examples) and LoRA r=64 config — not just DPO as Finding 8/9 in dpo-lab
originally concluded. The 3-epoch SFT loss curve looking healthy
(monotonically decreasing) is not sufficient evidence that more training
is better; it can be quietly trading held-out robustness for a lower
training-set loss.

## Honest residual uncertainty

- The OOD battery (15 items) is still small; a genuinely rare trigger could
  exist that this battery didn't happen to sample.
- The 2 known triggers and the 50 held-out prompts are not independent of
  each other's distributional assumptions (both are academic-writing-polish
  style prompts) — this doesn't test robustness on inputs entirely outside
  the task (e.g. small talk, which the demo's `MIN_CJK_RATIO`/length guards
  handle at the UI level, not the model level).
- This was found by one person clicking around for a few minutes. Treat
  "0 glitches in N tests" as evidence of a much lower rate, not zero rate.
