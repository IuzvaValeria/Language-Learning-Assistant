import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from backend.prompts import load_full_prompt

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
BASE_MODEL = os.getenv("BASE_MODEL", "ministral/Ministral-3b-instruct")
LORA_PATH = Path(os.getenv("LORA_PATH", "models/n5_lora_v4_translation_vocab_grammar"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))

tokenizer = None
model = None
def load_model() -> None:
    global tokenizer, model
    if USE_MOCK:
        print("Mock backend model loaded")
        return

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if LORA_PATH.exists():
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        print(f"LoRA adapter loaded from: {LORA_PATH}")
    else:
        model = base_model
        print(f"No LoRA adapter found at {LORA_PATH}, using base model")

    model.eval()
    print("Model loaded successfully")


def generate_mock_response(mode: str, user_text: str) -> str:
    system_prompt = load_full_prompt(mode)

    return f"""[MOCK BACKEND RESP]
Mode:
{mode}
Input:
{user_text}
Prompt preview:
{system_prompt[:500]}
Example answer:
This is a mock response. Later it will be replaced by the fine-tuned LoRA model.
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

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]

    decoded_output = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return decoded_output.strip()