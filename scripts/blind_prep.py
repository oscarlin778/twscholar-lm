"""
M3 step 2a: build a BLINDED head-to-head file for LLM-as-judge.

For each held-out item, pair two conditions' outputs, randomly assign them
to slots A/B (per-item coin flip), and write:
  - a blinded file (id, draft, A, B) with NO condition labels
  - a separate KEY file (id -> which slot is which condition)

The judge reads only the blinded file, records verdicts (A/B/tie), then
blind_unblind.py maps back and tallies wins + checks position balance.

Usage:
  python scripts/blind_prep.py sft_dpo base_zeroshot
"""

import json
import random
import sys

CONDITIONS_FILE = "data/eval/conditions_outputs.jsonl"
SEED = 7


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/blind_prep.py <condA> <condB>")
    c1, c2 = sys.argv[1], sys.argv[2]
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(CONDITIONS_FILE, encoding="utf-8")]

    blinded, key = [], {}
    for i, r in enumerate(rows):
        # per-item coin flip: which condition goes to slot A
        if rng.random() < 0.5:
            a_cond, b_cond = c1, c2
        else:
            a_cond, b_cond = c2, c1
        blinded.append({"id": i, "draft": r["draft"], "A": r[a_cond], "B": r[b_cond]})
        key[str(i)] = {"A": a_cond, "B": b_cond}

    tag = f"{c1}_vs_{c2}"
    with open(f"data/eval/blind_{tag}.jsonl", "w", encoding="utf-8") as f:
        for b in blinded:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    with open(f"data/eval/blind_{tag}_KEY.json", "w", encoding="utf-8") as f:
        json.dump({"c1": c1, "c2": c2, "key": key}, f, ensure_ascii=False, indent=2)

    print(f"Wrote data/eval/blind_{tag}.jsonl ({len(blinded)} items) + KEY")
    # print the blinded items for the judge (no labels)
    print("\n=== BLINDED ITEMS (judge these) ===")
    for b in blinded:
        print(f"[{b['id']}] draft: {b['draft']}")
        print(f"     A: {b['A']}")
        print(f"     B: {b['B']}")


if __name__ == "__main__":
    main()
