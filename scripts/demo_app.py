"""
M4: Gradio side-by-side demo — base Qwen2.5-7B vs twscholar-lm (SFT+DPO),
with a live Traditional-Chinese purity badge on each output.

Design rationale: the M3 evaluation found the base model already writes
strong academic Traditional Chinese; fine-tuning's measurable value is
script purity/consistency. This demo makes that finding *visible*: the same
prompt runs through both models (one weight copy — the adapter is toggled
via disable_adapter()), and each output gets an automatic purity check
using the same simplified-character table as the eval suite.

Run locally (needs ~6GB VRAM):
    python scripts/demo_app.py
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import gradio as gr

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DPO_ADAPTER = "outputs/dpo-qlora-7b"

SYSTEM_PROMPT = (
    "你是一位協助使用者潤飾繁體中文學術寫作的助手,回覆時力求精簡、正式,"
    "符合學術期刊慣例,只輸出潤飾後的文字,不需額外說明。"
)
TMPL = "請幫我把這段話潤飾成學術寫作的語氣:{d}"

SIMP = set("发显应审后实对兴趋势学检验时现个这与关长从会处于为说过们么认论议样较导动变观环响级统计资讯网际体团产业确讨广华问间东车马乐见觉话语读闻")

EXAMPLES = [
    "我們發現受試者在做困難任務時,瞳孔會放大。",
    "老師,我這週跑的模型結果怪怪的,想約時間討論。",
    "基本上我想講的重點就是,這個模型真的很吃資料。",
    "委員覺得我們的統計方法怪怪的,我們有換了一個方法重新分析。",
    "雖然電動車越來越普及,但充電基礎設施還是跟不上。",
]

print("Loading model (7B, 4-bit)...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="cuda:0")
model = PeftModel.from_pretrained(base, DPO_ADAPTER)
print("Ready.")


def _generate(draft: str) -> tuple[str, str]:
    """One weight copy, two generations: adapter off = base, on = fine-tuned."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TMPL.format(d=draft)},
    ]
    inp = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    def run():
        out = model.generate(**inp, max_new_tokens=150, do_sample=False)
        return tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    with model.disable_adapter():
        base_out = run()
    ft_out = run()
    return base_out, ft_out


def _badge(text: str, draft: str) -> str:
    simp_chars = [c for c in text if c in SIMP]
    eng = sum(1 for c in text if ("a" <= c <= "z" or "A" <= c <= "Z"))
    ratio = len(text) / max(len(draft), 1)
    if simp_chars:
        purity = f"⚠️ 偵測到簡體字 {len(simp_chars)} 個:{'、'.join(sorted(set(simp_chars)))}"
    else:
        purity = "✅ 全繁體,無簡體字外漏"
    eng_note = f"|英文字母 {eng} 個" if eng else ""
    return f"{purity}{eng_note}|長度比 {ratio:.2f}x"


def polish(draft: str):
    draft = (draft or "").strip()
    if not draft:
        return "", "", "", ""
    base_out, ft_out = _generate(draft)
    return base_out, _badge(base_out, draft), ft_out, _badge(ft_out, draft)


with gr.Blocks(title="twscholar-lm demo") as demo:
    gr.Markdown(
        "# twscholar-lm — 繁體中文學術寫作潤飾\n"
        "輸入口語草稿,並排比較 **未微調的 Qwen2.5-7B** 與 **twscholar-lm(SFT+DPO)**。"
        "每個輸出附自動繁體純度檢查——這正是微調可量化的價值所在(見 repo 的 M3 評估報告)。"
    )
    with gr.Row():
        draft_box = gr.Textbox(label="口語草稿", placeholder="例:我們這個實驗大概找了三十個人來測。", lines=2)
    btn = gr.Button("潤飾", variant="primary")
    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[draft_box], label="範例(點一下填入)")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Qwen2.5-7B(未微調)")
            base_out_box = gr.Textbox(label="輸出", lines=3)
            base_badge = gr.Markdown()
        with gr.Column():
            gr.Markdown("### twscholar-lm(SFT+DPO)")
            ft_out_box = gr.Textbox(label="輸出", lines=3)
            ft_badge = gr.Markdown()
    btn.click(polish, inputs=[draft_box], outputs=[base_out_box, base_badge, ft_out_box, ft_badge])
    draft_box.submit(polish, inputs=[draft_box], outputs=[base_out_box, base_badge, ft_out_box, ft_badge])


if __name__ == "__main__":
    demo.launch()
