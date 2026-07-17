"""
M3 step 1: generate held-out outputs for four conditions and compute
objective automated metrics (no LLM judge involved here).

Conditions (all on Qwen2.5-7B-Instruct, 4-bit):
  base_zeroshot : base model, polish instruction only
  base_fewshot  : base model, polish instruction + 3 in-context examples
  sft           : base + SFT adapter (outputs/sft-sft-qlora-7b)
  sft_dpo       : base + SFT+DPO adapter (outputs/dpo-qlora-7b) -- final model

Metrics per condition:
  avg_len            : mean output length (chars)
  len_ratio          : mean (output_len / draft_len)
  simp_hits          : total simplified-character hits
  rows_with_simp     : # outputs containing any simplified char
  eng_hits           : total ASCII-letter hits (English leakage)

Output: data/eval/conditions_outputs.jsonl, results/m3_auto_metrics.json
"""

import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SFT_ADAPTER = "outputs/sft-sft-qlora-7b"
DPO_ADAPTER = "outputs/dpo-qlora-7b"
HELDOUT = "data/eval/holdout_prompts.jsonl"
OUT_JSONL = "data/eval/conditions_outputs.jsonl"
OUT_METRICS = "results/m3_auto_metrics.json"

SYSTEM_PROMPT = (
    "你是一位協助使用者潤飾繁體中文學術寫作的助手,回覆時力求精簡、正式,"
    "符合學術期刊慣例,只輸出潤飾後的文字,不需額外說明。"
)
TMPL = "請幫我把這段話潤飾成學術寫作的語氣:{d}"

# few-shot exemplars (fixed, drawn from the task; not from the held-out set)
FEWSHOT = [
    ("我們這個實驗大概找了三十個人來測。", "本實驗共招募 30 名受試者。"),
    ("這個結果蠻讓人意外的。", "此結果出乎預期,值得進一步探討。"),
    ("之前的研究都沒有考慮到這個因素。", "既有文獻普遍未將此因素納入考量。"),
]

SIMP = set("发显应审后实对兴趋势学检验时现个这与关长从会处于为说过们么认论议样较导动变观环响级统计资讯网际体团产业确讨广华问间东车马乐见觉话语读闻这")


def build_messages(draft, fewshot=False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fewshot:
        for d, g in FEWSHOT:
            msgs.append({"role": "user", "content": TMPL.format(d=d)})
            msgs.append({"role": "assistant", "content": g})
    msgs.append({"role": "user", "content": TMPL.format(d=draft)})
    return msgs


def gen(model, tok, draft, fewshot=False):
    msgs = build_messages(draft, fewshot)
    inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    out = model.generate(**inp, max_new_tokens=150, do_sample=False)
    return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def metrics_for(outputs, drafts):
    lens = [len(o) for o in outputs]
    ratios = [len(o) / max(len(d), 1) for o, d in zip(outputs, drafts)]
    simp_hits = sum(sum(1 for c in o if c in SIMP) for o in outputs)
    rows_with_simp = sum(1 for o in outputs if any(c in SIMP for c in o))
    eng_hits = sum(sum(1 for c in o if ("a" <= c <= "z" or "A" <= c <= "Z")) for o in outputs)
    return {
        "avg_len": round(sum(lens) / len(lens), 1),
        "len_ratio": round(sum(ratios) / len(ratios), 2),
        "simp_hits": simp_hits,
        "rows_with_simp": rows_with_simp,
        "eng_hits": eng_hits,
    }


def main():
    prompts = [json.loads(l) for l in open(HELDOUT, encoding="utf-8")]
    drafts = [p["draft"] for p in prompts]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="cuda:0")

    results = {"base_zeroshot": [], "base_fewshot": [], "sft": [], "sft_dpo": []}

    print("base zero-shot...")
    results["base_zeroshot"] = [gen(base, tok, d) for d in drafts]
    print("base few-shot...")
    results["base_fewshot"] = [gen(base, tok, d, fewshot=True) for d in drafts]

    print("SFT...")
    sft = PeftModel.from_pretrained(base, SFT_ADAPTER)
    results["sft"] = [gen(sft, tok, d) for d in drafts]
    sft = sft.unload()

    print("SFT+DPO...")
    dpo = PeftModel.from_pretrained(base, DPO_ADAPTER)
    results["sft_dpo"] = [gen(dpo, tok, d) for d in drafts]

    rows = []
    for i, p in enumerate(prompts):
        rows.append({
            "category": p["category"], "draft": p["draft"],
            "base_zeroshot": results["base_zeroshot"][i],
            "base_fewshot": results["base_fewshot"][i],
            "sft": results["sft"][i],
            "sft_dpo": results["sft_dpo"][i],
        })
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = {cond: metrics_for(results[cond], drafts) for cond in results}
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n=== automated metrics ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
