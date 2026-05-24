from pathlib import Path
import csv
import json

RAW_FILE = Path("data/raw/jlpt/n5_vocab.csv")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "n5_vocab.jsonl"

def parse_n5_vocab():
    words = []

    if not RAW_FILE.exists():
        print(f"File not found: {RAW_FILE}")
        return words

    with RAW_FILE.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            expression = row.get("expression", "").strip()
            reading = row.get("reading", "").strip()
            meaning = row.get("meaning", "").strip()
            tags_raw = row.get("tags", "").strip()

            if not expression or not meaning:
                continue

            tags = [t.strip() for t in tags_raw.split() if t.strip()] if tags_raw else []

            words.append({
                "word": expression,
                "reading": reading,
                "meaning": meaning,
                "level": "N5",
                "tags": tags,
                "source": "open-anki-jlpt-decks",
            })

    return words

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    words = parse_n5_vocab()

    with OUT_FILE.open("w", encoding="utf-8") as file:
        for word in words:
            file.write(json.dumps(word, ensure_ascii=False) + "\n")

    print(f"Saved {len(words)} N5 words to {OUT_FILE}")

if __name__ == "__main__":
    main()