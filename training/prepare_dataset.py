from pathlib import Path
import json
import random

TRANSLATION_FILE = Path("data/processed/n5_translation.jsonl")
VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")

OUT_DIR = Path("data/final")
TRAIN_FILE = OUT_DIR / "train.jsonl"
VAL_FILE = OUT_DIR / "val.jsonl"

SYSTEM_PROMPT = (
    "You are a Japanese tutor for English-speaking JLPT N5 learners. "
    "Use simple English explanations. "
    "Keep Japanese examples beginner-friendly."
)

def load_jsonl(file_path: Path):
    items = []

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return items

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))

    return items

def make_translation_examples(pairs, max_examples=2000):
    examples = []

    for pair in pairs[:max_examples]:
        english = pair.get("english", "").strip()
        japanese = pair.get("japanese", "").strip()

        if not english or not japanese:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate into simple N5 Japanese: {english}"},
                {"role": "assistant", "content": japanese}
            ]
        })

    return examples

def make_vocab_examples(words, max_examples=800):
    examples = []

    for word in words[:max_examples]:
        expression = word.get("word", "").strip()
        reading = word.get("reading", "").strip()
        meaning = word.get("meaning", "").strip()

        if not expression or not meaning:
            continue

        answer = (
            f"{expression} means \"{meaning}\".\n"
            f"Reading: {reading}."
        )

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Explain the N5 word: {expression}"},
                {"role": "assistant", "content": answer}
            ]
        })

    return examples

def save_jsonl(file_path: Path, items):
    with file_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    translation_pairs = load_jsonl(TRANSLATION_FILE)
    vocab_words = load_jsonl(VOCAB_FILE)

    translation_examples = make_translation_examples(translation_pairs)
    vocab_examples = make_vocab_examples(vocab_words)

    all_examples = translation_examples + vocab_examples
    random.shuffle(all_examples)

    split_index = int(len(all_examples) * 0.9)

    train_examples = all_examples[:split_index]
    val_examples = all_examples[split_index:]

    save_jsonl(TRAIN_FILE, train_examples)
    save_jsonl(VAL_FILE, val_examples)

    print(f"Translation examples: {len(translation_examples)}")
    print(f"Vocabulary examples: {len(vocab_examples)}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()