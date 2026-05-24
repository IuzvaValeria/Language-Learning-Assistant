from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from backend.prompts import load_full_prompt

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

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

    if USE_MOCK:
        print("Mock backend model loaded")
        return
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if ADAPTER_PATH.exists():
        print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print("LoRA adapter not found. Using base model only.")
        model = base_model

    model.eval()
    print("Model loaded successfully")


def generate_mock_response(mode: str, user_text: str) -> str:
    system_prompt = load_full_prompt(mode)

    return f"""
[MOCK BACKEND RESP]
Mode:
{mode}
Input:
{user_text}
Prompt preview:
{system_prompt[:500]}
Example answer:
This is a mock response. Later will change for the Ministra :P
""".strip()


def generate_response(mode: str, user_text: str) -> str:
    if USE_MOCK:
        return generate_mock_response(mode, user_text)
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
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)