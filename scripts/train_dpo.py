"""
Task 2.3/2.4: DPOTrainer starting from a sft-lab SFT adapter, parameterized
for the beta sweep (Task 2.4).

Uses TRL's built-in "disable adapter as reference" trick: passing a PeftModel
with a pretrained ("default") adapter and ref_model=None makes DPOTrainer
clone that adapter as a frozen "ref" adapter internally (see
trl/trainer/dpo_trainer.py ~line 616) -- no second full model copy needed.

Data: data/processed/dpo_train.jsonl (Task 2.2 output). chosen = sft-lab's
reviewed polished text; rejected = the original draft (a "failed to polish"
stand-in -- see prepare_dpo_data.py docstring for the known limitation).

Usage:
    python scripts/train_dpo.py --beta 0.1 \\
        --model_id Qwen/Qwen2.5-1.5B-Instruct \\
        --sft_adapter_path ../sft-lab/outputs/sft-1.5b-lora_r64
    python scripts/train_dpo.py --beta 0.1 --load_in_4bit \\
        --model_id Qwen/Qwen2.5-7B-Instruct \\
        --sft_adapter_path ../sft-lab/outputs/sft-qlora-7b
"""

import argparse
import json
import time

# On Windows, importing torch AFTER matplotlib.pyplot segfaults (OpenMP/MKL
# runtime DLL load-order conflict) -- torch must come first.
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

import matplotlib

matplotlib.use("Agg")  # headless: no display available for GUI backends
import matplotlib.pyplot as plt

TRAIN_DATA_PATH = "data/processed/dpo_train.jsonl"
SEED = 42


def plot_training_curves(log_history, output_dir):
    rows = [row for row in log_history if "loss" in row]
    steps = [row["step"] for row in rows]
    losses = [row["loss"] for row in rows]
    accuracies = [row.get("rewards/accuracies") for row in rows]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(steps, losses, marker="o", color="tab:blue")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("DPO training loss")
    ax_loss.grid(alpha=0.3)

    if any(a is not None for a in accuracies):
        ax_acc.plot(steps, accuracies, marker="o", color="tab:green")
    ax_acc.set_xlabel("step")
    ax_acc.set_ylabel("rewards/accuracies")
    ax_acc.set_title("DPO preference accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft_adapter_path", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--data_path", type=str, default=TRAIN_DATA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"dpo-beta{args.beta}"
    output_dir = f"outputs/{run_name}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="cuda:0",
    )

    # is_trainable=True: this adapter keeps training as DPOTrainer's policy
    # ("default"). DPOTrainer will clone its current weights into a frozen
    # "ref" adapter for reference log-prob computation -- this frozen clone
    # IS pi_ref (the SFT model) from the DPO derivation.
    model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=True)

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        max_length=512,
        report_to="tensorboard",
        seed=SEED,
    )

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    elapsed = time.time() - start
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

    trainer.save_model(output_dir)

    log_history = trainer.state.log_history
    plot_training_curves(log_history, output_dir)
    last_train_log = next((row for row in reversed(log_history) if "rewards/accuracies" in row), {})

    metrics = {
        "run_name": run_name,
        "beta": args.beta,
        "model_id": args.model_id,
        "sft_adapter_path": args.sft_adapter_path,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_runtime_sec": round(elapsed, 1),
        "final_loss": last_train_log.get("loss"),
        "final_rewards_chosen": last_train_log.get("rewards/chosen"),
        "final_rewards_rejected": last_train_log.get("rewards/rejected"),
        "final_rewards_margin": last_train_log.get("rewards/margins"),
        "final_rewards_accuracy": last_train_log.get("rewards/accuracies"),
    }
    with open(f"{output_dir}/run_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
