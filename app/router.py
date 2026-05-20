from app.prompts import (
    build_translate_prompt,
    build_correct_prompt,
    build_explain_prompt,
    build_chat_prompt,
)

def build_prompt_by_mode(mode: str, text: str) -> str:
    if mode == "translate":
        return build_translate_prompt(text)
    if mode == "correct":
        return build_correct_prompt(text)
    if mode == "explain":
        return build_explain_prompt(text)
    if mode == "chat":
        return build_chat_prompt(text)
    raise ValueError(f"Unknown mode: {mode}")