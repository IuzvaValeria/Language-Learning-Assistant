from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from backend.prompts import load_full_prompt

MODEL_NAME = "mistralai/Ministral-3b-instruct-2512"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = None
model = None

def load_model() -> None:
    global tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    if ADAPTER_PATH.exists():
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        model = base_model

    model.eval()

def generate_response(mode: str, user_text: str) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    system_prompt = load_full_prompt(mode)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
