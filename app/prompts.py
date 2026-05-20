def build_translate_prompt(text: str) -> str:
    return f"""
You are a Japanese-English language tutor.
Task: translate the sentence and explain it for a beginner.

Text: {text}

Return:
Japanese:
Romaji:
Grammar:
Vocabulary:
"""


def build_correct_prompt(text: str) -> str:
    return f"""
You are a Japanese language tutor.
Task: correct the student's Japanese sentence.

Sentence:
{text}

Return:
Corrected sentence:
Mistakes:
Grammar explanation:
Natural version:
"""


def build_explain_prompt(text: str) -> str:
    return f"""
You are a Japanese language tutor.

Task: explain the grammar in this sentence for a beginner.

Sentence: {text}

Return:
Grammar point:
Simple explanation:
Example sentence:
"""


def build_chat_prompt(text: str) -> str:
    return f"""
You are a helpful Japanese-English language learning assistant.
User message: {text}

Answer simply and clearly.
"""