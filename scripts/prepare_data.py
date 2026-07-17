"""
Task 1.2: convert the hand-curated academic-writing pairs into the unified
`messages` schema SFTTrainer expects, and write out the training set.

General-diversity mixing (lianghsun/wikipedia-pretrain-zh-tw-chat, gated,
pending access approval) is intentionally deferred: it doesn't block
validating the training pipeline (Task 1.3) or the LoRA rank comparison
(Task 1.4), only the quality/generalization of the final published model
(Task 1.5+). Re-run this script with the mixed-in subset once access lands.
"""

import json
import random

RAW_DATA_PATH = "data/raw/academic_writing_pairs.jsonl"
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


def load_raw_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_messages(pair, rng):
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template.format(draft=pair["draft"])},
            {"role": "assistant", "content": pair["polished"]},
        ]
    }


def main():
    rng = random.Random(SEED)
    pairs = load_raw_pairs(RAW_DATA_PATH)

    examples = [to_messages(pair, rng) for pair in pairs]
    rng.shuffle(examples)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
