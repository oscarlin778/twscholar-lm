---
license: cc-by-nc-4.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - lora
  - qlora
  - sft
  - traditional-chinese
  - academic-writing
language:
  - zh
---

# twscholar-lm — SFT only (intermediate checkpoint)

QLoRA SFT adapter for Qwen2.5-7B-Instruct on the twscholar-lm dataset —
the SFT stage on its own, before DPO. Most users should prefer
`twscholar-lm-dpo-7b-final` (SFT + DPO); this checkpoint is published for
reproducibility and for anyone studying the SFT-vs-SFT+DPO comparison
directly.

- r=64, alpha=128, 1 epoch (see the main model card / project repo for why
  1 epoch was chosen over 3 — `results/m4_ood_glitch_investigation.md`)
- Full project: https://github.com/oscarlin778/twscholar-lm
- License: cc-by-nc-4.0 (inherited from the training dataset)
