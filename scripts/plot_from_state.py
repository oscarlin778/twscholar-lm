"""
Regenerate a training-curve plot from a Trainer's saved trainer_state.json
(works retroactively for any SFT or DPO run). Plots loss plus whichever
accuracy metric is present (mean_token_accuracy for SFT,
rewards/accuracies for DPO).

Usage:
    python scripts/plot_from_state.py outputs/sft-sft-1.5b-lora_r64
    python scripts/plot_from_state.py path/to/checkpoint-90/trainer_state.json
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_state(path):
    if path.endswith(".json"):
        return path
    # an output dir: prefer the latest checkpoint's trainer_state.json
    candidates = []
    for root, _, files in os.walk(path):
        if "trainer_state.json" in files:
            candidates.append(os.path.join(root, "trainer_state.json"))
    if not candidates:
        raise FileNotFoundError(f"no trainer_state.json under {path}")
    # latest checkpoint = highest step number in the dir name
    def step_of(p):
        d = os.path.basename(os.path.dirname(p))
        return int(d.split("-")[-1]) if d.startswith("checkpoint-") else -1
    return max(candidates, key=step_of)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python scripts/plot_from_state.py <output_dir|trainer_state.json>")
    state_path = find_state(sys.argv[1])
    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    rows = [r for r in state["log_history"] if "loss" in r]
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]

    acc_key = "mean_token_accuracy" if any("mean_token_accuracy" in r for r in rows) else \
              "rewards/accuracies" if any("rewards/accuracies" in r for r in rows) else None

    ncols = 2 if acc_key else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4), squeeze=False)

    ax = axes[0][0]
    ax.plot(steps, losses, marker="o", color="tab:blue")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training loss"); ax.grid(alpha=0.3)

    if acc_key:
        acc_steps = [r["step"] for r in rows if acc_key in r]
        accs = [r[acc_key] for r in rows if acc_key in r]
        ax2 = axes[0][1]
        ax2.plot(acc_steps, accs, marker="o", color="tab:green")
        ax2.set_xlabel("step"); ax2.set_ylabel(acc_key); ax2.set_title(acc_key); ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)

    out_dir = os.path.dirname(os.path.dirname(state_path)) if "checkpoint-" in state_path else os.path.dirname(state_path)
    out_png = os.path.join(out_dir, "training_curves.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"Wrote {out_png} ({len(rows)} logged steps, acc metric: {acc_key})")


if __name__ == "__main__":
    main()
