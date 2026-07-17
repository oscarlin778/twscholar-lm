---
license: cc-by-nc-4.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - lora
  - qlora
  - dpo
  - traditional-chinese
  - academic-writing
language:
  - zh
---

# twscholar-lm — SFT+DPO (final model)

A Traditional Chinese (Taiwan) academic-writing polisher, fine-tuned
end-to-end on a single consumer GPU (RTX 4070, 12GB). Full project:
https://github.com/oscarlin778/twscholar-lm

## What it does

Input a colloquial Traditional Chinese draft sentence, get back a concise,
formal rewrite matching academic-journal register.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

base_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(base_id)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="cuda:0")
model = PeftModel.from_pretrained(base, "<this-repo-id>")

system = "你是一位協助使用者潤飾繁體中文學術寫作的助手,回覆時力求精簡、正式,符合學術期刊慣例,只輸出潤飾後的文字,不需額外說明。"
draft = "我們發現受試者在做困難任務時,瞳孔會放大。"
msgs = [{"role": "system", "content": system},
        {"role": "user", "content": f"請幫我把這段話潤飾成學術寫作的語氣:{draft}"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=150, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Training recipe

- Base: Qwen2.5-7B-Instruct, 4-bit NF4 (QLoRA)
- LoRA r=64, alpha=128, target_modules = all attention + MLP projections
- SFT: 1 epoch on 476 examples (see dataset card), then DPO: 1 epoch, β=0.1,
  on 326 preference pairs (chosen=polished, rejected=draft)
- **Both stages deliberately use 1 epoch, not the more-obvious 3** — see
  "Why 1 epoch" below.

## Why 1 epoch (this is the interesting part)

An earlier version (SFT 3 epochs + DPO 1 epoch) passed all held-out checks
until manual demo testing surfaced a rare bug: informal/question-style
input (e.g. "所以我們目前的進度到底怎樣", <2% of training data) could
trigger a token-level glitch — "完成" garbled into "完cheng" (a stray
pinyin insertion). Checking saved checkpoints showed the glitch rate rising
monotonically with training steps (0/2 → 1/2 → 1/2 → 2/2 across increasing
SFT/DPO exposure) — the same overfitting-induced-instability mechanism
documented in the companion dpo-lab project's DPO investigation, this time
surfacing as a different symptom (word-level glitches, not script leakage).

Retraining both stages at 1 epoch fixed it: 0/50 glitches and 0/50
simplified-character leaks on the held-out set, with no visible quality
regression in spot-checks (despite a *higher* training loss than the
3-epoch version — the loss number and actual output robustness diverged).
Full writeup: `results/m4_ood_glitch_investigation.md` in the repo.

## Evaluation (honest summary)

Blind randomized A/B against the *untuned* base model, position-swap
corrected (98% judge self-agreement): **base 25 wins / this model 22 wins
/ 3 ties** on holistic quality — Qwen2.5-7B is already a strong Traditional
Chinese writer, and fine-tuning did not produce a dramatic quality jump.

What fine-tuning does measurably buy: **Traditional-Chinese script
consistency**. Objective metrics on 50 held-out prompts: base model leaks
simplified characters on 4/50 outputs; this model leaks on 0-1/50. For a
Traditional-Chinese-specific tool, that reliability guarantee — not a
quality leap — is the honest value proposition. Full report:
`results/m3_eval_report.md`.

## Limitations

- Trained on 476 examples (small-scale, not production-grade corpus)
- Single focused task (academic polishing), not a general assistant
- Rare, low-frequency generation glitches on far-out-of-distribution input
  cannot be ruled out beyond what was tested (see investigation report)
- License: cc-by-nc-4.0, inherited from the training dataset's
  `external_licensed` subset (derived from TaiwanChat, cc-by-nc-4.0)
