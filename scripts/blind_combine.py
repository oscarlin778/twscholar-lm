"""
M3 step 2c: combine the two judging passes (original + position-swapped) to
produce a position-bias-CORRECTED result.

pass1 is in the original blinded framing (A = orig A, B = orig B).
pass2 is judged on the SWAPPED file (new A = orig B, new B = orig A), so a
pass2 verdict of "A" means orig-B content, "B" means orig-A content.

A decisive win is counted only when BOTH passes pick the SAME content
(position-consistent). Items where the pick flips with position are
position-sensitive -> counted as tie. Consistent picks are then mapped to
conditions via the key.

Usage:
  python scripts/blind_combine.py sft_dpo_vs_base_zeroshot
"""

import json
import sys
from collections import Counter


def main():
    tag = sys.argv[1]
    key_obj = json.load(open(f"data/eval/blind_{tag}_KEY.json", encoding="utf-8"))
    p1 = json.load(open(f"data/eval/blind_{tag}_VERDICTS.json", encoding="utf-8"))["verdicts"]
    p2 = json.load(open(f"data/eval/blind_{tag}_VERDICTS_swapped.json", encoding="utf-8"))["verdicts"]
    c1, c2, key = key_obj["c1"], key_obj["c2"], key_obj["key"]

    # convert each pass to "which original slot won" (origA / origB / tie)
    def p1_slot(v):
        return v  # already orig framing
    def p2_slot(v):
        return {"A": "B", "B": "A", "tie": "tie"}[v]  # swapped -> orig

    wins = Counter()
    consistent = 0
    flipped = 0
    ties = 0
    for i in p1:
        s1, s2 = p1_slot(p1[i]), p2_slot(p2[i])
        if s1 == "tie" or s2 == "tie":
            ties += 1
            wins["tie"] += 1
            continue
        if s1 == s2:
            consistent += 1
            wins[key[i][s1]] += 1  # map orig slot -> condition
        else:
            flipped += 1
            wins["tie"] += 1  # position-sensitive -> tie

    total = len(p1)
    print(f"=== {tag}: position-bias-corrected (two-pass) ===")
    print(f"  position-consistent decisive: {consistent}/{total}")
    print(f"  position-sensitive (flipped) -> tie: {flipped}")
    print(f"  double-tie: {ties}")
    print(f"\n  {c1}: {wins[c1]} wins")
    print(f"  {c2}: {wins[c2]} wins")
    print(f"  tie (incl. flips): {wins['tie']}")
    agree = 100 * consistent / (consistent + flipped) if (consistent + flipped) else 0
    print(f"\n  judge self-agreement under position swap: {agree:.0f}% "
          f"({consistent}/{consistent + flipped} decisive items consistent)")


if __name__ == "__main__":
    main()
