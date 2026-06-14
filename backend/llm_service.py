import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.prompts import load_full_prompt


USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
BASE_MODEL = os.getenv("BASE_MODEL", "ministral/Ministral-3b-instruct")
LORA_PATH = Path(os.getenv("LORA_PATH", "models/n5_lora_v4_translation_vocab_grammar"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))

tokenizer = None
model = None


N5_VOCAB_FALLBACK = {
    "水": {
        "reading": "みず",
        "meaning": "water",
        "example": "水を飲みます。",
        "translation": "I drink water.",
    },
    "本": {
        "reading": "ほん",
        "meaning": "book",
        "example": "これは本です。",
        "translation": "This is a book.",
    },
    "学校": {
        "reading": "がっこう",
        "meaning": "school",
        "example": "学校に行きます。",
        "translation": "I go to school.",
    },
    "学生": {
        "reading": "がくせい",
        "meaning": "student",
        "example": "私は学生です。",
        "translation": "I am a student.",
    },
    "先生": {
        "reading": "せんせい",
        "meaning": "teacher",
        "example": "先生に聞きます。",
        "translation": "I ask the teacher.",
    },
    "猫": {
        "reading": "ねこ",
        "meaning": "cat",
        "example": "猫がいます。",
        "translation": "There is a cat.",
    },
    "犬": {
        "reading": "いぬ",
        "meaning": "dog",
        "example": "犬が好きです。",
        "translation": "I like dogs.",
    },
    "人": {
        "reading": "ひと",
        "meaning": "person",
        "example": "あの人は先生です。",
        "translation": "That person is a teacher.",
    },
    "今日": {
        "reading": "きょう",
        "meaning": "today",
        "example": "今日は学校に行きます。",
        "translation": "Today I go to school.",
    },
    "明日": {
        "reading": "あした",
        "meaning": "tomorrow",
        "example": "明日、勉強します。",
        "translation": "Tomorrow I will study.",
    },
}


TRANSLATION_FALLBACK = {
    "this is a book": "これは本です。",
    "i go to school": "学校に行きます。",
    "i drink water": "水を飲みます。",
    "i am a student": "私は学生です。",
    "i like cats": "猫が好きです。",
    "これは本です": "This is a book.",
    "私は学生です": "I am a student.",
    "水を飲みます": "I drink water.",
    "学校に行きます": "I go to school.",
}


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
        raise FileNotFoundError(f"LoRA adapter not found: {LORA_PATH}")

    model.eval()
    print("Model loaded successfully")


def clean_user_text(user_text: str) -> str:
    text = user_text.strip()

    prefixes = [
        "Translate to Japanese:",
        "Translate to English:",
        "Explain the N5 word:",
        "Explain this word:",
        "Word:",
        "Input:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text


def generate_vocab_fallback(user_text: str) -> str | None:
    word = clean_user_text(user_text)

    if word not in N5_VOCAB_FALLBACK:
        return None

    item = N5_VOCAB_FALLBACK[word]

    return (
        f"Word: {word}\n"
        f"Reading: {item['reading']}\n"
        f"Meaning: {item['meaning']}\n"
        f"Example: {item['example']}\n"
        f"Translation: {item['translation']}"
    )


def generate_translation_fallback(user_text: str) -> str | None:
    text = clean_user_text(user_text)
    normalized_text = text.lower().rstrip(".。!?！？")

    if normalized_text not in TRANSLATION_FALLBACK:
        return None

    return TRANSLATION_FALLBACK[normalized_text]


def generate_mock_response(mode: str, user_text: str) -> str:
    if mode == "vocabulary":
        fallback_answer = generate_vocab_fallback(user_text)
        if fallback_answer is not None:
            return fallback_answer

    if mode == "translation":
        fallback_answer = generate_translation_fallback(user_text)
        if fallback_answer is not None:
            return fallback_answer

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


def generate_model_response(mode: str, user_text: str) -> str:
    if tokenizer is None or model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    system_prompt = load_full_prompt(mode)

    prompt = f"""You are a Japanese language tutor for English-speaking beginners.

Task mode: {mode}

Instructions:
{system_prompt}

User input:
{user_text}

Write the answer in English.
Do not repeat the user input.
Do not explain what you are going to do.
Give only the final tutor answer.

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
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


def generate_response(mode: str, user_text: str) -> str:
    if mode == "vocabulary":
        fallback_answer = generate_vocab_fallback(user_text)
        if fallback_answer is not None:
            return fallback_answer

    if mode == "translation":
        fallback_answer = generate_translation_fallback(user_text)
        if fallback_answer is not None:
            return fallback_answer

    if USE_MOCK:
        return generate_mock_response(mode, user_text)

    return generate_model_response(mode, user_text)