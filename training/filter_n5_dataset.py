from pathlib import Path
import json

TATOEBA_FILE = Path("data/processed/tatoeba_pairs.jsonl")
N5_VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "n5_translation.jsonl"

MAX_ENGLISH_LEN = 80
MAX_JAPANESE_LEN = 35
MIN_TEXT_LEN = 2
SOURCE_LANG = "en"
TARGET_LANG = "ja"


def load_n5_words() -> set:
    n5_words = set()

    if not N5_VOCAB_FILE.exists():
        print(f"File not found: {N5_VOCAB_FILE}")
        return n5_words

    with N5_VOCAB_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            word = item.get("word", "").strip()
            reading = item.get("reading", "").strip()

            if word:
                n5_words.add(word)
            if reading:
                n5_words.add(reading)

    return n5_words

def is_short_pair(english_text: str, japanese_text: str) -> bool:
    if len(english_text) > MAX_ENGLISH_LEN:
        return False
    if len(japanese_text) > MAX_JAPANESE_LEN:
        return False
    if len(english_text) < MIN_TEXT_LEN or len(japanese_text) < MIN_TEXT_LEN:
        return False
    return True

def has_n5_word(japanese_text: str, n5_words: set) -> bool:
    return any(word in japanese_text for word in n5_words)


def filter_tatoeba_pairs(n5_words: set) -> list:
    filtered_pairs = []

    if not TATOEBA_FILE.exists():
        print(f"File not found: {TATOEBA_FILE}")
        return filtered_pairs

    total = 0
    skipped_lang = 0
    skipped_length = 0
    skipped_vocab = 0

    with TATOEBA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            total += 1
            item = json.loads(line)

            source_lang = item.get("source_lang")
            target_lang = item.get("target_lang")
            source_text = item.get("source_text", "").strip()
            target_text = item.get("target_text", "").strip()

            if source_lang != SOURCE_LANG or target_lang != TARGET_LANG:
                skipped_lang += 1
                continue

            if not is_short_pair(source_text, target_text):
                skipped_length += 1
                continue

            if not has_n5_word(target_text, n5_words):
                skipped_vocab += 1
                continue

            filtered_pairs.append({
                "english": source_text,
                "japanese": target_text,
                "level": "N5",
                "task": "translation",
                "source": "tatoeba",
            })

    print(f"Total pairs processed : {total}")
    print(f"Skipped (wrong lang)  : {skipped_lang}")
    print(f"Skipped (too long)    : {skipped_length}")
    print(f"Skipped (no N5 word)  : {skipped_vocab}")
    print(f"Passed filter         : {len(filtered_pairs)}")

    return filtered_pairs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n5_words = load_n5_words()
    print(f"Loaded N5 words: {len(n5_words)}")

    filtered_pairs = filter_tatoeba_pairs(n5_words)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for pair in filtered_pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()