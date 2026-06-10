from pathlib import Path
import json

KFTT_FILE = Path("data/processed/kftt_pairs.jsonl")
N5_VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")
N5_KANJI_FILE = Path("data/raw/n5_kanji.txt")
N4_KANJI_FILE = Path("data/raw/n4_kanji.txt")
BAD_PATTERNS_FILE = Path("data/raw/bad_patterns.txt")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "kftt_n5_filtered.jsonl"

MAX_ENGLISH_LEN = 100
MAX_JAPANESE_LEN = 60
MIN_TEXT_LEN = 5
MIN_N5_WORD_COUNT = 2
MAX_ENGLISH_WORDS = 15
MAX_PAIRS = 3000


# def load_n5_kanji() -> set:
#     if not N5_KANJI_FILE.exists():
#         print(f"File not found: {N5_KANJI_FILE}")
#         return set()
#     text = N5_KANJI_FILE.read_text(encoding="utf-8")
#     return set(c for c in text if "\u4e00" <= c <= "\u9fff")

def load_allowed_kanji() -> set:
    kanji = set()
    for file_path in [N5_KANJI_FILE, N4_KANJI_FILE]:
        if file_path.exists():
            text = file_path.read_text(encoding="utf-8")
            kanji.update(c for c in text if "\u4e00" <= c <= "\u9fff")
    return kanji


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


def filter_kftt(n5_words: set, n5_kanji: set, bad_patterns: list) -> list:
    filtered_pairs = []

    if not KFTT_FILE.exists():
        print(f"File not found: {KFTT_FILE}")
        return filtered_pairs

    total = 0
    skipped_length = 0
    skipped_vocab = 0
    skipped_pattern = 0
    skipped_kanji = 0
    skipped_json = 0

    with KFTT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            if len(filtered_pairs) >= MAX_PAIRS:
                break

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped_json += 1
                continue

            english_text = item.get("source_text", "").strip()
            japanese_text = item.get("target_text", "").strip()

            if not is_short_pair(english_text, japanese_text):
                skipped_length += 1
                continue

            if not has_n5_word(japanese_text, n5_words):
                skipped_vocab += 1
                continue

            if has_bad_pattern(japanese_text, bad_patterns):
                skipped_pattern += 1
                continue

            if has_unknown_kanji(japanese_text, n5_kanji):
                skipped_kanji += 1
                continue

            filtered_pairs.append({
                "english": english_text,
                "japanese": japanese_text,
                "source_lang": "en",
                "target_lang": "ja",
                "level": "N5",
                "task": "translation",
                "source": "kftt",
            })

    print(f"Total processed  : {total}")
    print(f"Skipped json     : {skipped_json}")
    print(f"Skipped length   : {skipped_length}")
    print(f"Skipped vocab    : {skipped_vocab}")
    print(f"Skipped pattern  : {skipped_pattern}")
    print(f"Skipped kanji    : {skipped_kanji}")
    print(f"Passed filter    : {len(filtered_pairs)}")

    return filtered_pairs

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    allowed_kanji = load_allowed_kanji() 
    n5_words = load_n5_words()
    bad_patterns = load_bad_patterns()

    print(f"Loaded allowed kanji : {len(allowed_kanji)}")
    print(f"Loaded N5 words    : {len(n5_words)}")
    print(f"Loaded bad patterns: {len(bad_patterns)}")

    filtered_pairs = filter_kftt(n5_words, allowed_kanji, bad_patterns)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for pair in filtered_pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Saved to: {OUT_FILE}")

if __name__ == "__main__":
    main()