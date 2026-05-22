from pathlib import Path

PROMPTS_DIR = Path("prompts")

def load_prompt(file_name: str) -> str:
    prompt_path = PROMPTS_DIR / file_name

    if not prompt_path.exists():
        return ""

    return prompt_path.read_text(encoding="utf-8")