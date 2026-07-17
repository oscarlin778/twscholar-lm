"""
M3 step 2b: unblind the judge verdicts against the hidden key and tally
wins per condition, plus a position-balance check (did slot A win more
often than slot B? — a signal of position bias in the judging).

Usage:
  python scripts/blind_unblind.py sft_dpo_vs_base_zeroshot
"""

import json
import sys
from collections import Counter


def main():
    tag = sys.argv[1]
    key_obj = json.load(open(f"data/eval/blind_{tag}_KEY.json", encoding="utf-8"))
    verdicts = json.load(open(f"data/eval/blind_{tag}_VERDICTS.json", encoding="utf-8"))["verdicts"]
    c1, c2, key = key_obj["c1"], key_obj["c2"], key_obj["key"]

    wins = Counter()
    slot_wins = Counter()
    slot_of = Counter()          # how often each condition sat in slot A / B
    win_when_in = Counter()      # (condition, slot) -> wins
    for item_id, kv in key.items():
        slot_of[(kv["A"], "A")] += 1
        slot_of[(kv["B"], "B")] += 1
    for item_id, v in verdicts.items():
        if v == "tie":
            wins["tie"] += 1
            continue
        slot_wins[v] += 1
        cond = key[item_id][v]
        wins[cond] += 1
        win_when_in[(cond, v)] += 1

    total = len(verdicts)
    print(f"=== {tag} ({total} items) ===")
    print(f"  {c1}: {wins[c1]} wins")
    print(f"  {c2}: {wins[c2]} wins")
    print(f"  tie:  {wins['tie']}")
    print(f"\nposition balance (bias check): slot A won {slot_wins['A']}, slot B won {slot_wins['B']}")
    print("\nslot assignment (how often each condition sat in A vs B):")
    for c in (c1, c2):
        print(f"  {c}: in A {slot_of[(c,'A')]}x, in B {slot_of[(c,'B')]}x")
    print("\nwin-rate conditioned on slot:")
    for c in (c1, c2):
        wa, wb = win_when_in[(c,'A')], win_when_in[(c,'B')]
        na, nb = slot_of[(c,'A')], slot_of[(c,'B')]
        print(f"  {c}: when in A won {wa}/{na}, when in B won {wb}/{nb}")


if __name__ == "__main__":
    main()
