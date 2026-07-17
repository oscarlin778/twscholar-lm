"""
Task 1.6: generate base-vs-fine-tuned outputs on held-out prompts
(data/eval/holdout_prompts.jsonl — none of these appear in the training set)
for manual/LLM-as-judge comparison. Uses the Task 1.5 QLoRA-7B adapter.
"""

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "outputs/sft-qlora-7b"
HOLDOUT_PATH = "data/eval/holdout_prompts.jsonl"
OUTPUT_PATH = "data/eval/holdout_outputs.jsonl"
SYSTEM_PROMPT = (
    "你是一位協助使用者潤飾繁體中文學術寫作的助手,回覆時力求精簡、正式,"
    "符合學術期刊慣例,只輸出潤飾後的文字,不需額外說明。"
)
INSTRUCTION_TEMPLATE = "請幫我把這段話潤飾成學術寫作的語氣:{draft}"


def generate(model, tokenizer, draft):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION_TEMPLATE.format(draft=draft)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def main():
    with open(HOLDOUT_PATH, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cuda:0"
    )

    print("Generating base model outputs...")
    base_outputs = [generate(base_model, tokenizer, p["draft"]) for p in prompts]

    print("Attaching QLoRA adapter and generating fine-tuned outputs...")
    ft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    ft_outputs = [generate(ft_model, tokenizer, p["draft"]) for p in prompts]

    results = [
        {
            "category": p["category"],
            "draft": p["draft"],
            "base_output": base_out,
            "finetuned_output": ft_out,
        }
        for p, base_out, ft_out in zip(prompts, base_outputs, ft_outputs)
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} comparisons to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
