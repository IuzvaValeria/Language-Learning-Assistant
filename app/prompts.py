def build_translation_prompt(english_text: str) -> str:
    return f"""
You are an English-to-Japanese translator for language learners.
Translate this sentence to Japanese and explain it simply.

English: {english_text}
Return:
Japanese:
Romaji:
Grammar:
Vocabulary:
"""