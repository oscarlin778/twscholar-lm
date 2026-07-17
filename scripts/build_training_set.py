"""
M1: merge all provenance sources into one SFTTrainer-ready training set.

  - seed_ai_drafted.jsonl / human_polished.jsonl : (draft, polished) pairs,
    wrapped into `messages` with a varied instruction template.
  - external_taiwanchat.jsonl : already `messages` (general Traditional
    Chinese conversation), included as-is with the same system prompt.

Output: data/processed/sft_train.jsonl (shuffled), plus a provenance
breakdown printed for the dataset card.
"""

import json
import random
from collections import Counter

PAIR_SOURCES = ["data/raw/seed_ai_drafted.jsonl", "data/raw/human_polished.jsonl"]
EXTERNAL_SOURCE = "data/raw/external_taiwanchat.jsonl"
OUTPUT_PATH = "data/processed/sft_train.jsonl"
SEED = 42

SYSTEM_PROMPT = (
    "你是一位協助使用者潤飾繁體中文學術寫作的助手,回覆時力求精簡、正式,"
    "符合學術期刊慣例,只輸出潤飾後的文字,不需額外說明。"
)

INSTRUCTION_TEMPLATES = [
    "請幫我把這段話潤飾成學術寫作的語氣:{draft}",
    "請將以下句子改寫為正式學術用語:{draft}",
    "幫我修飾下面這段文字,讓它更符合學術論文的用語習慣:{draft}",
    "以下是我的草稿,麻煩潤成比較正式的學術寫作風格:{draft}",
    "請將這段口語化的敘述改寫成適合放進論文的版本:{draft}",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    rng = random.Random(SEED)
    examples = []
    provenance = Counter()

    for path in PAIR_SOURCES:
        for row in load_jsonl(path):
            template = rng.choice(INSTRUCTION_TEMPLATES)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template.format(draft=row["draft"])},
                    {"role": "assistant", "content": row["polished"]},
                ],
                "source": row["source"],
            })
            provenance[row["source"]] += 1

    for row in load_jsonl(EXTERNAL_SOURCE):
        examples.append({
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + row["messages"],
            "source": row["source"],
        })
        provenance[row["source"]] += 1

    rng.shuffle(examples)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(examples)
    print(f"Wrote {total} training examples to {OUTPUT_PATH}")
    print("\nProvenance breakdown:")
    for src, n in provenance.most_common():
        print(f"  {n:4d}  ({100*n/total:4.1f}%)  {src}")


if __name__ == "__main__":
    main()
