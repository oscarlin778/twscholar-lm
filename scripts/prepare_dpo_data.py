"""
Task 2.2: convert the sft-lab academic-writing pairs into DPO preference
format. chosen = the reviewed polished version; rejected = the original
draft (a stand-in "failed to polish" response). This is a fast bootstrap,
not a rich failure-mode set — rejected only covers one failure mode
(un-polished text), not others like over-verbosity or script leakage
(see sft-lab Task 1.6 findings). Worth revisiting later.
"""

import json
import random

RAW_DATA_PATH = "data/raw/academic_writing_pairs.jsonl"
OUTPUT_PATH = "data/processed/dpo_train.jsonl"
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


def to_dpo_example(pair, rng):
    template = rng.choice(INSTRUCTION_TEMPLATES)
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": template.format(draft=pair["draft"])},
        ],
        "chosen": [{"role": "assistant", "content": pair["polished"]}],
        "rejected": [{"role": "assistant", "content": pair["draft"]}],
    }


def main():
    rng = random.Random(SEED)
    pairs = load_raw_pairs(RAW_DATA_PATH)

    examples = [to_dpo_example(pair, rng) for pair in pairs]
    rng.shuffle(examples)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} DPO preference pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
