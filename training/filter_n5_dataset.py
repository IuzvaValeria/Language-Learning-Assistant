from pathlib import Path
import json

TATOEBA_FILE = Path("data/processed/tatoeba_pairs.jsonl")
N5_VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "n5_translation.jsonl"

def load_n5_words():
    n5_words = set()

    if not N5_VOCAB_FILE.exists():
        print(f"File not found: {N5_VOCAB_FILE}")
        return n5_words

    with N5_VOCAB_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            word = item.get("word", "").strip()
            reading = item.get("reading", "").strip()

            if word:
                n5_words.add(word)

            if reading:
                n5_words.add(reading)

    return n5_words

def is_short_pair(english_text: str, japanese_text: str) -> bool:
    if len(english_text) > 80:
        return False

    if len(japanese_text) > 35:
        return False

    if len(english_text) < 2 or len(japanese_text) < 2:
        return False

    return True

def has_n5_word(japanese_text: str, n5_words: set) -> bool:
    for word in n5_words:
        if word in japanese_text:
            return True

    return False

def filter_tatoeba_pairs(n5_words: set):
    filtered_pairs = []

    if not TATOEBA_FILE.exists():
        print(f"File not found: {TATOEBA_FILE}")
        return filtered_pairs

    with TATOEBA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)

            source_lang = item.get("source_lang")
            target_lang = item.get("target_lang")
            source_text = item.get("source_text", "").strip()
            target_text = item.get("target_text", "").strip()

            #for now only English -> Japanese
            if source_lang != "en" or target_lang != "ja":
                continue

            if not is_short_pair(source_text, target_text):
                continue

            if not has_n5_word(target_text, n5_words):
                continue

            filtered_pairs.append({
                "english": source_text,
                "japanese": target_text,
                "level": "N5",
                "task": "translation",
                "source": "tatoeba"
            })

    return filtered_pairs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n5_words = load_n5_words()
    print(f"Loaded N5 words: {len(n5_words)}")

    filtered_pairs = filter_tatoeba_pairs(n5_words)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for pair in filtered_pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved N5 translation pairs: {len(filtered_pairs)}")
    print(f"Output file: {OUT_FILE}")


if __name__ == "__main__":
    main()