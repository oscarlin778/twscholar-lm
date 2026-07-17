"""
SFTTrainer on Qwen2.5-Instruct, parameterized so the same script runs full
fine-tuning or LoRA/QLoRA at different scales (the M2 three-line comparison:
0.5B full-FT / 1.5B LoRA / 7B QLoRA).

Data: data/processed/sft_train.jsonl (built by scripts/build_training_set.py
from the provenance-labeled raw sources).

Usage:
    python scripts/train_sft.py --mode full --model_id Qwen/Qwen2.5-0.5B-Instruct
    python scripts/train_sft.py --mode lora --r 64 --alpha 128
    python scripts/train_sft.py --mode lora --r 64 --alpha 128 \\
        --model_id Qwen/Qwen2.5-7B-Instruct --load_in_4bit --run_name qlora-7b
"""

import argparse
import json
import time

# On Windows, importing torch AFTER matplotlib.pyplot segfaults (OpenMP/MKL
# DLL load-order conflict) -- torch must come first.
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = "data/processed/sft_train.jsonl"
SEED = 42


def plot_training_curves(log_history, output_dir):
    rows = [r for r in log_history if "loss" in r and "step" in r]
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    accs = [(r["step"], r["mean_token_accuracy"]) for r in rows if "mean_token_accuracy" in r]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(steps, losses, marker="o", color="tab:blue")
    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss"); ax_loss.set_title("SFT training loss"); ax_loss.grid(alpha=0.3)
    if accs:
        ax_acc.plot([s for s, _ in accs], [a for _, a in accs], marker="o", color="tab:green")
    ax_acc.set_xlabel("step"); ax_acc.set_ylabel("mean_token_accuracy"); ax_acc.set_title("SFT token accuracy")
    ax_acc.set_ylim(0, 1.05); ax_acc.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{output_dir}/training_curves.png", dpi=150); plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora"], required=True)
    parser.add_argument("--r", type=int, default=16, help="LoRA rank (ignored for --mode full)")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha (ignored for --mode full)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--load_in_4bit", action="store_true", help="QLoRA: quantize base weights to nf4")
    parser.add_argument("--run_name", type=str, default=None, help="Overrides the auto-derived run/output name")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    return parser.parse_args()


def count_trainable_params(model):
    # bnb's Linear4bit packs two nf4 values per uint8 byte, so `.numel()` on
    # a quantized weight tensor undercounts its true parameter count by 2x.
    def real_numel(p):
        return p.numel() * 2 if p.dtype == torch.uint8 else p.numel()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(real_numel(p) for p in model.parameters())
    return trainable, total


def main():
    args = parse_args()
    default_name = "full" if args.mode == "full" else f"lora_r{args.r}"
    run_name = args.run_name or default_name
    output_dir = args.output_dir or f"outputs/sft-{run_name}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    quantization_config = None
    if args.load_in_4bit:
        # QLoRA: weights at rest are 4-bit (nf4, double-quantized), matmuls
        # are dequantized on the fly to bf16 (Task 1.1 discussion). Only
        # meaningful combined with --mode lora — the base weights stay
        # frozen either way, so quantizing them for full FT would just
        # break gradient flow into params that need to be updated.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="cuda:0",
    )

    lora_config = None
    if args.mode == "lora":
        # r: LoRA rank, the bottleneck dimension of ΔW = BA. Larger r = more
        # trainable parameters = higher capacity, but higher VRAM/compute
        # cost and overfitting risk on a small dataset.
        # lora_alpha: scaling factor on ΔW (effectively ΔW * alpha/r);
        # alpha = 2*r is a common starting heuristic (used here for both r=8
        # and r=64 so the scaling ratio stays constant across the sweep).
        lora_config = LoraConfig(
            r=args.r,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=2e-4 if args.mode == "lora" else 2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        # Only train on assistant-turn tokens (Task 1.1: loss masking). TRL
        # auto-swaps in a template with {% generation %} markers if the
        # tokenizer's default template lacks them.
        assistant_only_loss=True,
        report_to="tensorboard",
        seed=SEED,
    )

    dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable, total = count_trainable_params(trainer.model)
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    trainer.train()

    elapsed = time.time() - start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    trainer.save_model(output_dir)
    plot_training_curves(trainer.state.log_history, output_dir)

    metrics = {
        "run_name": run_name,
        "mode": args.mode,
        "r": args.r if args.mode == "lora" else None,
        "alpha": args.alpha if args.mode == "lora" else None,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_runtime_sec": round(elapsed, 1),
        "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
    }
    with open(f"{output_dir}/run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
