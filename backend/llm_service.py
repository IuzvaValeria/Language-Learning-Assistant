import os
import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from backend.prompts import load_full_prompt

USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"
MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"

tokenizer = None
model = None

def load_model() -> None:
    global tokenizer, model
    if USE_MOCK:
        print("Mock backend model loaded")
        return
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = MistralCommonBackend.from_pretrained(MODEL_NAME)
    model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_NAME, device_map="auto")
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
This is a mock response. Later will change for the Ministral model :P
""".strip()

def generate_response(mode: str, user_text: str) -> str:
    if USE_MOCK:
        return generate_mock_response(mode, user_text)
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    system_prompt = load_full_prompt(mode)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]
    tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
    tokenized["input_ids"] = tokenized["input_ids"].to("cuda")
    with torch.no_grad():
        output = model.generate(**tokenized, max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "100")))[0]
    decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
    return decoded_output.strip()