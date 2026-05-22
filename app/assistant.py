from app.prompt_loader import load_prompt


def get_prompt_for_mode(mode: str) -> tuple[str, str]:
    if mode == "translate":
        return "translation_prompt.txt", load_prompt("translation_prompt.txt")

    if mode == "grammar":
        return "grammar_prompt.txt", load_prompt("grammar_prompt.txt")

    if mode == "vocabulary":
        return "system_n5.txt", load_prompt("system_n5.txt")

    if mode == "correction":
        return "correction_prompt.txt", load_prompt("correction_prompt.txt")

    if mode == "examples":
        return "exercise_prompt.txt", load_prompt("exercise_prompt.txt")

    return "system_prompt.txt", load_prompt("system_prompt.txt")


def make_mode_response(mode_name: str, prompt_file: str, user_instruction: str) -> str:
    return (
        f"{mode_name} mode selected.\n"
        f"Prompt loaded: {prompt_file}\n"
        f"{user_instruction}"
    )


def answer_user(message: str) -> str:
    message_lower = message.lower().strip()

    if message_lower in ["translate", "translation", "1"]:
        prompt_file, prompt_text = get_prompt_for_mode("translate")

        if not prompt_text:
            return "Translation mode selected, but translation prompt file was not found."

        return make_mode_response(
            "Translation",
            prompt_file,
            "Please send a Japanese or English sentence."
        )

    if message_lower in ["grammar", "2"]:
        prompt_file, prompt_text = get_prompt_for_mode("grammar")

        if not prompt_text:
            return "Grammar mode selected, but grammar prompt file was not found."

        return make_mode_response(
            "Grammar",
            prompt_file,
            "Send me your grammar question."
        )

    if message_lower in ["vocabulary", "word", "explain", "3"]:
        prompt_file, prompt_text = get_prompt_for_mode("vocabulary")

        if not prompt_text:
            return "Vocabulary mode selected, but N5 system prompt file was not found."

        return make_mode_response(
            "Vocabulary",
            prompt_file,
            "Send me a Japanese word."
        )

    if message_lower in ["correct", "mistake", "correction", "4"]:
        prompt_file, prompt_text = get_prompt_for_mode("correction")

        if not prompt_text:
            return "Correction mode selected, but correction prompt file was not found."

        return make_mode_response(
            "Correction",
            prompt_file,
            "Send me your Japanese sentence."
        )

    if message_lower in ["examples", "example", "sentences", "5"]:
        prompt_file, prompt_text = get_prompt_for_mode("examples")

        if not prompt_text:
            return "Examples mode selected, but exercise prompt file was not found."

        return make_mode_response(
            "Example sentences",
            prompt_file,
            "Send me a word or topic."
        )

    return (
        "I can help with Japanese-English learning.\n\n"
        "Choose one option:\n"
        "1. Translate a sentence\n"
        "2. Explain grammar\n"
        "3. Explain vocabulary\n"
        "4. Correct a mistake\n"
        "5. Give example sentences"
    )