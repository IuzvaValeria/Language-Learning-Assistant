from pathlib import Path
import json
import random

TRANSLATION_FILE = Path("data/processed/n5_translation.jsonl")
VOCAB_FILE = Path("data/processed/n5_vocab.jsonl")

OUT_DIR = Path("data/final")
TRAIN_FILE = OUT_DIR / "train.jsonl"
VAL_FILE = OUT_DIR / "val.jsonl"

SEED = 42
TRAIN_SPLIT = 0.9
MAX_TRANSLATION_EXAMPLES = 2000
MAX_VOCAB_EXAMPLES = 800

SYSTEM_PROMPT = (
    "You are a Japanese tutor for English-speaking JLPT N5 learners. "
    "Use simple English explanations. "
    "Keep Japanese examples beginner-friendly."
)


def load_jsonl(file_path: Path) -> list:
    items = []

    if not file_path.exists():
        print(f"File not found: {file_path}")
        return items

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))

    return items

def make_translation_examples(pairs: list, max_examples: int = MAX_TRANSLATION_EXAMPLES) -> list:
    examples = []

    shuffled = pairs.copy()
    random.shuffle(shuffled)

    for pair in shuffled[:max_examples]:
        english = pair.get("english", "").strip()
        japanese = pair.get("japanese", "").strip()

        if not english or not japanese:
            continue

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate into simple N5 Japanese: {english}"},
                {"role": "assistant", "content": japanese},
            ]
        })

    return examples


def make_vocab_examples(words: list, max_examples: int = MAX_VOCAB_EXAMPLES) -> list:
    examples = []

    shuffled = words.copy()
    random.shuffle(shuffled)

    for word in shuffled[:max_examples]:
        expression = word.get("word", "").strip()
        reading = word.get("reading", "").strip()
        meaning = word.get("meaning", "").strip()

        if not expression or not meaning:
            continue

        answer = f"{expression} means \"{meaning}\"."
        if reading:
            answer += f"\nReading: {reading}."

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Explain the N5 word: {expression}"},
                {"role": "assistant", "content": answer},
            ]
        })

    return examples


def save_jsonl(file_path: Path, items: list) -> None:
    with file_path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED) 

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    translation_pairs = load_jsonl(TRANSLATION_FILE)
    vocab_words = load_jsonl(VOCAB_FILE)

    print(f"Loaded translation pairs: {len(translation_pairs)}")
    print(f"Loaded vocab words: {len(vocab_words)}")

    translation_examples = make_translation_examples(translation_pairs)
    vocab_examples = make_vocab_examples(vocab_words)

    all_examples = translation_examples + vocab_examples
    random.shuffle(all_examples) 

    split_index = int(len(all_examples) * TRAIN_SPLIT)
    train_examples = all_examples[:split_index]
    val_examples = all_examples[split_index:]

    save_jsonl(TRAIN_FILE, train_examples)
    save_jsonl(VAL_FILE, val_examples)

    print(f"\nTranslation examples: {len(translation_examples)}")
    print(f"Vocabulary examples:  {len(vocab_examples)}")
    print(f"Total examples:       {len(all_examples)}")
    print(f"Train examples:       {len(train_examples)}")
    print(f"Validation examples:  {len(val_examples)}")
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()