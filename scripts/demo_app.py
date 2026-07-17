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
DPO_ADAPTER = "outputs/dpo-qlora-7b-v2"  # SFT 1 epoch + DPO 1 epoch; see results/m4_ood_glitch_investigation.md

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

# guardrails: the training data is single sentences, 10-60 chars. Inputs far
# outside that range or with little CJK content are out-of-distribution and
# the model degrades (hallucination on 1-char input, off-task on English-only,
# summarization instead of polishing on very long input -- all observed
# empirically before adding these checks).
MIN_LEN = 4
MAX_LEN = 200
MIN_CJK_RATIO = 0.4


def _is_cjk(ch: str) -> bool:
    return "一" <= ch <= "鿿"


def _validate(draft: str) -> str | None:
    """Returns a user-facing error message, or None if input is OK to generate."""
    if len(draft) < MIN_LEN:
        return f"⚠️ 輸入過短(少於 {MIN_LEN} 字),過短的句子容易讓模型產生不相關的內容,請輸入完整的句子。"
    if len(draft) > MAX_LEN:
        return f"⚠️ 輸入過長(超過 {MAX_LEN} 字)。本模型是針對單句/短段落訓練,請拆成較短的句子分別潤飾,以獲得穩定結果。"
    cjk_ratio = sum(1 for c in draft if _is_cjk(c)) / len(draft)
    if cjk_ratio < MIN_CJK_RATIO:
        return "⚠️ 偵測到輸入的中文字元比例偏低。本模型專門訓練於繁體中文學術寫作潤飾,非中文輸入的結果可能不準確或答非所問。"
    return None

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

    warning = _validate(draft)
    if warning:
        return "", warning, "", warning

    try:
        base_out, ft_out = _generate(draft)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        msg = "⚠️ GPU 記憶體不足,請稍後再試或縮短輸入內容。"
        return "", msg, "", msg
    except Exception as e:
        msg = f"⚠️ 生成過程發生錯誤,請稍後再試。({type(e).__name__})"
        return "", msg, "", msg

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
