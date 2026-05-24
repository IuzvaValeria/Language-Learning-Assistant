from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

AVAILABLE_MODES = {
    "translation": "translation_prompt.txt",
    "grammar": "grammar_prompt.txt",
    "correction": "correction_prompt.txt",
    "exercise": "exercise_prompt.txt",
    "chat": "system_n5.txt"
}

FEW_SHOT_FILE = PROMPTS_DIR / "few_shot_examples.md"

def load_prompt(mode: str) -> str:
    if mode not in AVAILABLE_MODES:
        available = ", ".join(AVAILABLE_MODES.keys())
        raise ValueError(f"Unknown mode: {mode}. Available modes: {available}")
    prompt_file = PROMPTS_DIR / AVAILABLE_MODES[mode]
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")

def load_few_shot() -> str:
    if not FEW_SHOT_FILE.exists():
        return ""
    return FEW_SHOT_FILE.read_text(encoding="utf-8")

def load_full_prompt(mode: str) -> str:
    return load_prompt(mode) + "\n\n" + load_few_shot()