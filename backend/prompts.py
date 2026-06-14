from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

AVAILABLE_MODES = {
    "translation": "translation_prompt.txt",
    "vocabulary": "vocab_prompt.txt",
    "chat": "system_n5.txt",
}

def load_prompt(mode: str) -> str:
    if mode not in AVAILABLE_MODES:
        available = ", ".join(AVAILABLE_MODES.keys())
        raise ValueError(f"Unknown mode: {mode}. Available modes: {available}")

    prompt_file = PROMPTS_DIR / AVAILABLE_MODES[mode]

    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    return prompt_file.read_text(encoding="utf-8")


def load_full_prompt(mode: str) -> str:
    return load_prompt(mode)