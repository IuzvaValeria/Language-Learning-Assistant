from pathlib import Path
import json

RAW_DIR = Path("data/raw/kftt/kftt-data-1.0/data/orig")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "kftt_pairs.jsonl"

FILES = [
    ("kyoto-train.en", "kyoto-train.ja"),
    ("kyoto-dev.en", "kyoto-dev.ja"),
]

MAX_ENGLISH_LEN = 120
MAX_JAPANESE_LEN = 80

def has_japanese(text: str) -> bool:
    return any(
        "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff"
        for char in text
    )

def has_english(text: str) -> bool:
    return any("a" <= char.lower() <= "z" for char in text)


def parse_kftt_pair(en_file: Path, ja_file: Path) -> list:
    pairs = []

    if not en_file.exists():
        print(f"File not found: {en_file}")
        return pairs

    if not ja_file.exists():
        print(f"File not found: {ja_file}")
        return pairs

    with en_file.open("r", encoding="utf-8") as ef, \
         ja_file.open("r", encoding="utf-8") as jf:

        for en_line, ja_line in zip(ef, jf):
            en_text = en_line.strip()
            ja_text = ja_line.strip()

            if not en_text or not ja_text:
                continue

            if not has_english(en_text) or not has_japanese(ja_text):
                continue

            if len(en_text) > MAX_ENGLISH_LEN or len(ja_text) > MAX_JAPANESE_LEN:
                continue

            pairs.append({
                "source_lang": "en",
                "target_lang": "ja",
                "source_text": en_text,
                "target_text": ja_text,
                "dataset": "kftt",
            })

    print(f"Parsed {en_file.name}: {len(pairs)} pairs")
    return pairs

def deduplicate(pairs: list) -> list:
    seen = set()
    unique = []
    for pair in pairs:
        key = (pair["source_text"], pair["target_text"])
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    for en_filename, ja_filename in FILES:
        en_file = RAW_DIR / en_filename
        ja_file = RAW_DIR / ja_filename
        pairs = parse_kftt_pair(en_file, ja_file)
        all_pairs.extend(pairs)

    before_dedup = len(all_pairs)
    all_pairs = deduplicate(all_pairs)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for pair in all_pairs:
            file.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Total before dedup: {before_dedup}")
    print(f"Total after dedup : {len(all_pairs)}")
    print(f"Saved to          : {OUT_FILE}")

if __name__ == "__main__":
    main()