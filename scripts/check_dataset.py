from pathlib import Path
import json

TRAIN_FILE = Path("data/final/train.jsonl")
VAL_FILE = Path("data/final/val.jsonl")

MAX_CONTENT_CHARS = 1500

def check_file(file_path: Path) -> None:
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    total = 0
    errors = 0
    too_long = 0
    task_types = {"translation": 0, "vocab": 0, "grammar": 0, "unknown": 0}

    expected_roles = ["system", "user", "assistant"]

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            total += 1
            has_error = False

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  JSON error at line {line_number}")
                errors += 1
                continue

            messages = item.get("messages")

            if not isinstance(messages, list):
                print(f"  Missing messages at line {line_number}")
                errors += 1
                continue

            if len(messages) != 3:
                print(f"  Wrong number of messages at line {line_number}: {len(messages)}")
                errors += 1
                continue

            for message, expected_role in zip(messages, expected_roles):
                role = message.get("role")
                content = message.get("content", "").strip()

                if role != expected_role:
                    print(f"  Wrong role at line {line_number}: got '{role}', expected '{expected_role}'")
                    has_error = True

                if not content:
                    print(f"  Empty content at line {line_number}, role: {role}")
                    has_error = True

            if has_error:
                errors += 1
                continue

            total_chars = sum(len(m.get("content", "")) for m in messages)
            if total_chars > MAX_CONTENT_CHARS:
                too_long += 1

            user_content = messages[1].get("content", "")
            user_lower = user_content.lower()

            if "translate" in user_lower:
                task_types["translation"] += 1
            elif "explain the n5 word" in user_lower:
                task_types["vocab"] += 1
            elif any(
                keyword  in user_lower
                for keyword in [
                    "explain",
                    "how do i",
                    "what does", 
                    "what is", 
                    "when do", 
                    "when should"
                ]
            ):
                task_types["grammar"] += 1
            else:
                task_types["unknown"] += 1

    print(f"\nFile: {file_path}")
    print(f"  Total examples : {total}")
    print(f"  Errors         : {errors}")
    print(f"  Too long (>{MAX_CONTENT_CHARS} chars): {too_long}")
    print(f"  Translation    : {task_types['translation']}")
    print(f"  Vocab          : {task_types['vocab']}")
    print(f"  Grammar        : {task_types['grammar']}")
    print(f"  Unknown        : {task_types['unknown']}")

def main():
    check_file(TRAIN_FILE)
    check_file(VAL_FILE)

if __name__ == "__main__":
    main()