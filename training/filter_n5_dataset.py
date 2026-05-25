from pathlib import Path
import json

TATOEBA_FILE = Path("data/processed/tatoeba_pairs.jsonl")
N5_VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")
N5_KANJI_FILE = Path("data/raw/n5_kanji.txt")
BAD_PATTERNS_FILE = Path("data/raw/bad_patterns.txt")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "n5_translation.jsonl"

MAX_ENGLISH_LEN = 50
MAX_JAPANESE_LEN = 25
MIN_TEXT_LEN = 5
MIN_N5_WORD_COUNT = 3
MAX_ENGLISH_WORDS = 6
SOURCE_LANG = "en"
TARGET_LANG = "ja"


def load_n5_kanji() -> set:
    if not N5_KANJI_FILE.exists():
        print(f"File not found: {N5_KANJI_FILE}")
        return set()
    text = N5_KANJI_FILE.read_text(encoding="utf-8")
    return set(c for c in text if "\u4e00" <= c <= "\u9fff")


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


def load_bad_patterns() -> list:
    if not BAD_PATTERNS_FILE.exists():
        print(f"File not found: {BAD_PATTERNS_FILE}")
        return []
    lines = BAD_PATTERNS_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def is_short_pair(english_text: str, japanese_text: str) -> bool:
    if len(english_text) > MAX_ENGLISH_LEN:
        return False
    if len(japanese_text) > MAX_JAPANESE_LEN:
        return False
    if len(english_text) < MIN_TEXT_LEN or len(japanese_text) < MIN_TEXT_LEN:
        return False
    if len(english_text.split()) > MAX_ENGLISH_WORDS:
        return False
    return True


def has_n5_word(japanese_text: str, n5_words: set) -> bool:
    count = sum(1 for word in n5_words if word in japanese_text)
    return count >= MIN_N5_WORD_COUNT


def has_bad_pattern(japanese_text: str, bad_patterns: list) -> bool:
    for pattern in bad_patterns:
        if pattern in japanese_text:
            return True
    return False


def has_unknown_kanji(japanese_text: str, n5_kanji: set) -> bool:
    for char in japanese_text:
        if "\u4e00" <= char <= "\u9fff":
            if char not in n5_kanji:
                return True
    return False


def filter_tatoeba_pairs(n5_words: set, n5_kanji: set, bad_patterns: list) -> list:
    filtered_pairs = []

    if not TATOEBA_FILE.exists():
        print(f"File not found: {TATOEBA_FILE}")
        return filtered_pairs

    total = 0
    skipped_lang = 0
    skipped_length = 0
    skipped_vocab = 0
    skipped_grammar = 0
    skipped_kanji = 0

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

            if has_bad_pattern(target_text, bad_patterns):
                skipped_grammar += 1
                continue

            if has_unknown_kanji(target_text, n5_kanji):
                skipped_kanji += 1
                continue

            filtered_pairs.append({
                "english": source_text,
                "japanese": target_text,
                "level": "N5",
                "task": "translation",
                "source": "tatoeba",
            })

    print(f"Total pairs processed  : {total}")
    print(f"Skipped (wrong lang)   : {skipped_lang}")
    print(f"Skipped (too long)     : {skipped_length}")
    print(f"Skipped (no N5 word)   : {skipped_vocab}")
    print(f"Skipped (bad pattern)  : {skipped_grammar}")
    print(f"Skipped (hard kanji)   : {skipped_kanji}")
    print(f"Passed filter          : {len(filtered_pairs)}")

    return filtered_pairs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n5_kanji = load_n5_kanji()
    n5_words = load_n5_words()
    bad_patterns = load_bad_patterns()

    print(f"Loaded N5 kanji    : {len(n5_kanji)}")
    print(f"Loaded N5 words    : {len(n5_words)}")
    print(f"Loaded bad patterns: {len(bad_patterns)}")

    filtered_pairs = filter_tatoeba_pairs(n5_words, n5_kanji, bad_patterns)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for pair in filtered_pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()