from pathlib import Path
from datetime import datetime
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "ministral/Ministral-3b-instruct"
LORA_PATH = Path("models/n5_lora")
RESULTS_DIR = Path("results/lora_tests")

SYSTEM_PROMPT = (
    "You are a helpful Japanese-English language learning tutor. "
    "Focus on JLPT N5 level. "
    "Explain simply and clearly."
)

TEST_SETS = {
    "basic": {
        "description": "Basic mixed test: translation, grammar, vocabulary, correction.",
        "questions": [
            "Translate to Japanese: I eat rice.",
            "Translate to English: 時間がありません。",
            "Explain the grammar in this sentence: 私は日本語を勉強します。",
            "What does 食べる mean? Give reading and example.",
            "Correct this Japanese sentence: これは本ます。",
        ],
    },
    "translate": {
        "description": "Translation test: English to Japanese and Japanese to English.",
        "questions": [
            "Translate to Japanese: This is a book.",
            "Translate to Japanese: I go to school.",
            "Translate to Japanese: I drink tea.",
            "Translate to English: 私は学生です。",
            "Translate to English: これは本です。",
        ],
    },
    "grammar": {
        "description": "Grammar test: particles, polite form, simple sentence structure.",
        "questions": [
            "Explain the particle は in this sentence: 私は学生です。",
            "Explain the particle を in this sentence: 水を飲みます。",
            "Explain why 飲みます is polite and 飲む is plain.",
            "Explain the grammar in this sentence: 私は学校に行きます。",
        ],
    },
    "vocab": {
        "description": "Vocabulary test: meaning, reading, simple example.",
        "questions": [
            "What does 学生 mean? Give reading and example.",
            "What does 水 mean? Give reading and example.",
            "What does 行く mean? Give reading and example.",
            "What does 本 mean? Give reading and example.",
        ],
    },
    "correction": {
        "description": "Correction test: fix simple learner mistakes.",
        "questions": [
            "Correct this Japanese sentence: 私は水を飲むます。",
            "Correct this Japanese sentence: 私は学校を行きます。",
            "Correct this Japanese sentence: これは本ます。",
            "Correct this Japanese sentence: 私は学生をです。",
        ],
    },
}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if LORA_PATH.exists():
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
        print(f"LoRA adapter loaded from: {LORA_PATH}")
    else:
        model = base_model
        print("No LoRA adapter found, using base model")

    model.eval()
    return tokenizer, model


def generate(tokenizer, model, user_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def choose_test_set():
    names = list(TEST_SETS.keys())

    print("\nChoose test set:")
    for index, name in enumerate(names, start=1):
        description = TEST_SETS[name]["description"]
        print(f"  {index} - {name}: {description}")

    choice = input("Enter number: ").strip()

    if not choice.isdigit():
        print("Invalid choice, using basic.")
        return "basic"

    index = int(choice) - 1

    if index < 0 or index >= len(names):
        print("Invalid choice, using basic.")
        return "basic"

    return names[index]


def get_result_paths(test_set_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    jsonl_path = RESULTS_DIR / f"{test_set_name}_{timestamp}.jsonl"
    md_path = RESULTS_DIR / f"{test_set_name}_{timestamp}.md"

    return jsonl_path, md_path


def save_jsonl(path, data):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False) + "\n")


def run_test_set(tokenizer, model, test_set_name):
    test_set = TEST_SETS[test_set_name]
    questions = test_set["questions"]

    jsonl_path, md_path = get_result_paths(test_set_name)

    run_info = {
        "timestamp": datetime.now().isoformat(),
        "test_set": test_set_name,
        "description": test_set["description"],
        "base_model": BASE_MODEL,
        "lora_path": str(LORA_PATH),
        "lora_exists": LORA_PATH.exists(),
    }

    save_jsonl(jsonl_path, {"run_info": run_info})

    with md_path.open("w", encoding="utf-8") as file:
        file.write(f"# LoRA Test: {test_set_name}\n\n")
        file.write(f"**Description:** {test_set['description']}\n\n")
        file.write(f"**Base model:** `{BASE_MODEL}`\n\n")
        file.write(f"**LoRA path:** `{LORA_PATH}`\n\n")
        file.write(f"**LoRA exists:** `{LORA_PATH.exists()}`\n\n")
        file.write("---\n\n")

    print(f"\nRunning test set: {test_set_name}")
    print(f"Saving JSONL: {jsonl_path}")
    print(f"Saving Markdown: {md_path}\n")

    for index, question in enumerate(questions, start=1):
        print("=" * 60)
        print(f"QUESTION {index}: {question}")

        answer = generate(tokenizer, model, question)

        print(f"ANSWER:\n{answer}\n")

        result = {
            "index": index,
            "question": question,
            "answer": answer,
        }

        save_jsonl(jsonl_path, result)

        with md_path.open("a", encoding="utf-8") as file:
            file.write(f"## Test {index}\n\n")
            file.write(f"**Question:** {question}\n\n")
            file.write("**Answer:**\n\n")
            file.write(answer)
            file.write("\n\n---\n\n")

    print("=" * 60)
    print("Testing finished.")
    print(f"Results saved to: {RESULTS_DIR}")


def run_interactive(tokenizer, model):
    print("\nType your own question.")
    print("Type exit, quit, or q to stop.\n")

    while True:
        user_input = input("Input: ").strip()

        if user_input.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        if not user_input:
            continue

        answer = generate(tokenizer, model, user_input)
        print(f"\nTutor:\n{answer}\n")


if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model ready.\n")

    print("Choose mode:")
    print("  1 - Run saved test set")
    print("  2 - User input")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        selected_test_set = choose_test_set()
        run_test_set(tokenizer, model, selected_test_set)
    elif mode == "2":
        run_interactive(tokenizer, model)
    else:
        print("Invalid choice, running basic test.")
        run_test_set(tokenizer, model, "basic")