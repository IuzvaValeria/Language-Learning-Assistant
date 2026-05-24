from pathlib import Path
import json

RAW_DIR = Path("data/raw/tatoeba")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "tatoeba_pairs.jsonl"

SOURCE_TEXT_IDX = 1
TARGET_TEXT_IDX = 3

def read_tsv(file_path: Path, source_lang: str, target_lang: str) -> list:
    pairs = []

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return pairs

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")

            if len(parts) < 4:
                continue

            source_text = parts[SOURCE_TEXT_IDX].strip()
            target_text = parts[TARGET_TEXT_IDX].strip()

            if not source_text or not target_text:
                continue

            pairs.append({
                "source_lang": source_lang,
                "target_lang": target_lang,
                "source_text": source_text,
                "target_text": target_text,
                "dataset": "tatoeba",
            })

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

    # Только en → ja
    eng_jpn = read_tsv(RAW_DIR / "eng-jpn.tsv", "en", "ja")

    all_pairs = deduplicate(eng_jpn)

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for item in all_pairs:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_pairs)} pairs to {OUT_FILE}")

if __name__ == "__main__":
    main()