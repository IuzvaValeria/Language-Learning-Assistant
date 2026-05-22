from pathlib import Path
import json

TRAIN_FILE = Path("data/final/train.jsonl")
VAL_FILE = Path("data/final/val.jsonl")

def check_file(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    total = 0
    errors = 0

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"JSON error in {file_path}, line {line_number}")
                errors += 1
                continue

            messages = item.get("messages")

            if not isinstance(messages, list):
                print(f"Missing messages in {file_path}, line {line_number}")
                errors += 1
                continue

            if len(messages) != 3:
                print(f"Wrong number of messages in {file_path}, line {line_number}")
                errors += 1
                continue

            expected_roles = ["system", "user", "assistant"]

            for message, expected_role in zip(messages, expected_roles):
                role = message.get("role")
                content = message.get("content", "").strip()

                if role != expected_role:
                    print(f"Wrong role in {file_path}, line {line_number}: {role}")
                    errors += 1

                if not content:
                    print(f"Empty content in {file_path}, line {line_number}")
                    errors += 1

    print(f"\nChecked: {file_path}")
    print(f"Total examples: {total}")
    print(f"Errors: {errors}")

def main():
    check_file(TRAIN_FILE)
    check_file(VAL_FILE)

if __name__ == "__main__":
    main()