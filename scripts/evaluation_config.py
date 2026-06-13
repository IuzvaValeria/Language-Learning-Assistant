import sys
from pathlib import Path

BASE_MODEL = "ministral/Ministral-3b-instruct"
LORA_PATH = Path("models/n5_lora_v4_translation_vocab_grammar") #change this to the path of your LoRA model
RESULTS_DIR = Path("results/evaluation")

MAX_NEW_TOKENS = 200

SYSTEM_PROMPT = (
    "You are a helpful Japanese-English language learning tutor. "
    "Focus on JLPT N5 level. "
    "Explain simply and clearly."
)

TEST_SETS = {
    "translation_en_ja": {
        "description": "English to Japanese translation",
        "questions": [
            {"question": "Translate to Japanese: This is a book.", "expected": "本"},
            {"question": "Translate to Japanese: I go to school.", "expected": "学校"},
            {"question": "Translate to Japanese: I drink tea.", "expected": "飲"},
            {"question": "Translate to Japanese: I am a student.", "expected": "学生"},
            {"question": "Translate to Japanese: I do not have time.", "expected": "時間"},
        ],
    },
    "translation_ja_en": {
        "description": "Japanese to English translation",
        "questions": [
            {"question": "Translate to English: 私は学生です。", "expected": "student"},
            {"question": "Translate to English: これは本です。", "expected": "book"},
            {"question": "Translate to English: 私は学校に行きます。", "expected": "school"},
            {"question": "Translate to English: 水を飲みます。", "expected": "water"},
            {"question": "Translate to English: 時間がありません。", "expected": "time"},
        ],
    },
    "grammar": {
        "description": "Grammar explanation",
        "questions": [
            {"question": "Explain the grammar pattern: Noun + は", "expected": "topic"},
            {"question": "What does を do in ごはんを食べます?", "expected": "object"},
            {"question": "What is the difference between に and で?", "expected": "action"},
            {"question": "What does ます mean?", "expected": "polite"},
            {"question": "What is the difference between あります and います?", "expected": "living"},
        ],
    },
    "vocabulary": {
        "description": "Vocabulary explanation",
        "questions": [
            {"question": "Explain the N5 word: 学生", "expected": "student"},
            {"question": "Explain the N5 word: 水", "expected": "water"},
            {"question": "Explain the N5 word: 行く", "expected": "go"},
            {"question": "Explain the N5 word: 本", "expected": "book"},
            {"question": "Explain the N5 word: 食べる", "expected": "eat"},
        ],
    },
}

DISPLAY_NAMES = {
    "translation_en_ja": "English → Japanese translation",
    "translation_ja_en": "Japanese → English translation",
    "grammar": "Grammar explanation",
    "vocabulary": "Vocabulary explanation",
}