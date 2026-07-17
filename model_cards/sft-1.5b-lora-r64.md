---
license: cc-by-nc-4.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - lora
  - sft
  - traditional-chinese
  - academic-writing
language:
  - zh
---

# twscholar-lm — 1.5B LoRA (scale-comparison checkpoint)

Part of a three-line scale comparison (0.5B full fine-tune / 1.5B LoRA /
7B QLoRA) documenting VRAM and quality trade-offs on a single 12GB
consumer GPU. This is the 1.5B point — not the recommended model for
actual use (see `twscholar-lm-dpo-7b-final`), published for the
comparison's reproducibility.

- LoRA r=64, alpha=128, 3 epochs
- Peak VRAM during training: 6.07GB
- Full comparison table: `results/m2_sft_comparison.json` in the project
  repo — https://github.com/oscarlin778/twscholar-lm
- License: cc-by-nc-4.0 (inherited from the training dataset)
