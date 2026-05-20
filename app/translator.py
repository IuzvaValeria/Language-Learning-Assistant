from transformers import pipeline
from app.prompts import build_translation_prompt

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

translator = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device_map="auto"
)

def translate_to_japanese(english_text: str) -> str:
    prompt = build_translation_prompt(english_text)
    result = translator(
        prompt,
        max_new_tokens=300,
        do_sample=False
    )

    return result[0]["generated_text"]