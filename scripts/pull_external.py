"""
M1: pull a small, quality-filtered subset of yentinglin/TaiwanChat
(cc-by-nc-4.0) as `external_licensed` data — light general Traditional
Chinese conversational diversity to reduce over-narrowing to the polishing
task alone.

Filters (all must pass):
  - single user->assistant turn
  - assistant reply 20-220 chars (not trivially short, not a wall of text)
  - no code markers
  - Traditional-Chinese purity: at most 1 simplified character
  - majority-CJK (drops English/translation-task rows)

Output: data/raw/external_taiwanchat.jsonl in `messages` format with
source=external_licensed. NOTE: this makes the *dataset* cc-by-nc-4.0.
"""

import json

from datasets import load_dataset

SOURCE = "yentinglin/TaiwanChat"
OUTPUT_PATH = "data/raw/external_taiwanchat.jsonl"
TARGET_N = 150
SEED = 42

# simplified characters to detect (values of a trad->simp map); if the text
# contains more than a couple of these, it isn't clean Traditional Chinese.
SIMPLIFIED_CHARS = set("发显应审后实对兴趋势学检验时现个这与关长从会处于为说过们么认论议样较导动变观环响级统计资讯网际体团产业确讨广华门问间东车马乐见观觉话语读闻")

CODE_MARKERS = ["def ", "import ", "```", "function", "return ", "print(", "{}", "=>", "public ", "class ", "#include", "SELECT ", "console."]


def is_cjk(ch):
    return "一" <= ch <= "鿿"


def keep(messages):
    if not messages or len(messages) != 2:
        return False
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return False
    reply = messages[1]["content"].strip()
    prompt = messages[0]["content"].strip()
    if not (20 <= len(reply) <= 220):
        return False
    combined = prompt + reply
    if any(m in combined for m in CODE_MARKERS):
        return False
    # Traditional purity
    if sum(1 for c in reply if c in SIMPLIFIED_CHARS) > 1:
        return False
    # majority CJK in the reply (drops English/translation rows)
    cjk = sum(1 for c in reply if is_cjk(c))
    if cjk / max(len(reply), 1) < 0.6:
        return False
    return True


def main():
    ds = load_dataset(SOURCE, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10000)

    kept = []
    for ex in ds:
        msgs = ex.get("messages")
        if msgs and keep(msgs):
            kept.append({
                "messages": [
                    {"role": "user", "content": msgs[0]["content"].strip()},
                    {"role": "assistant", "content": msgs[1]["content"].strip()},
                ],
                "source": "external_licensed",
                "origin": SOURCE,
            })
        if len(kept) >= TARGET_N:
            break

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(kept)} filtered external pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
