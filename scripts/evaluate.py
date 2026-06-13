import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from evaluation_config import (
    BASE_MODEL,
    LORA_PATH,
    MAX_NEW_TOKENS,
    SYSTEM_PROMPT,
    TEST_SETS,
    DISPLAY_NAMES,
)

from evaluation_report import save_report

def show_environment() -> None:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU: not available, using CPU")

def show_system_prompt() -> None:
    print("\n--- SYSTEM PROMPT ---")
    print(SYSTEM_PROMPT)

def show_test_sets() -> None:
    print("\n--- TEST SETS ---")

    total = 0

    for set_name, test_set in TEST_SETS.items():
        count = len(test_set["questions"])
        total += count

        print(f"{DISPLAY_NAMES.get(set_name, set_name)}: {count} questions")

    print("Total questions:", total)

def load_model(use_lora: bool = True):
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("Loading base model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_lora:
        if not LORA_PATH.exists():
            raise FileNotFoundError(f"LoRA path not found: {LORA_PATH}")

        print(f"Loading LoRA adapter from: {LORA_PATH}")
        model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print("Using base model without LoRA")

    model.eval()

    return tokenizer, model

def build_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt

def generate_answer(tokenizer, model, user_text: str) -> str:
    prompt = build_prompt(tokenizer, user_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()

def normalize(text: str) -> str:
    return (
        text.lower()
        .strip()
        .replace("。", "")
        .replace(".", "")
        .replace(",", "")
        .replace("!", "")
        .replace("?", "")
    )

def is_correct(answer: str, expected: str) -> bool:
    normalized_answer = normalize(answer)
    normalized_expected = normalize(expected)

    return normalized_expected in normalized_answer

def run_evaluation(tokenizer, model, label: str) -> dict:
    print(f"\n--- EVALUATING: {label} ---")

    results = {}

    for set_name, test_set in TEST_SETS.items():
        questions = test_set["questions"]
        task_results = []
        passed = 0

        print(f"\nTask: {DISPLAY_NAMES.get(set_name, set_name)}")

        for index, item in enumerate(questions, start=1):
            question = item["question"]
            expected = item["expected"]

            print(f"  Question {index}/{len(questions)}")

            answer = generate_answer(
                tokenizer=tokenizer,
                model=model,
                user_text=question,
            )

            correct = is_correct(
                answer=answer,
                expected=expected,
            )

            if correct:
                passed += 1

            task_results.append(
                {
                    "question": question,
                    "expected": expected,
                    "answer": answer,
                    "correct": correct,
                }
            )

        total = len(questions)
        accuracy = passed / total if total else 0

        results[set_name] = {
            "description": test_set["description"],
            "passed": passed,
            "total": total,
            "accuracy": accuracy,
            "items": task_results,
        }

        print(f"  Result: {passed}/{total} ({accuracy:.0%})")

    total_passed = sum(result["passed"] for result in results.values())
    total_questions = sum(result["total"] for result in results.values())

    overall_accuracy = total_passed / total_questions if total_questions else 0

    results["overall"] = {
        "passed": total_passed,
        "total": total_questions,
        "accuracy": overall_accuracy,
    }

    print(f"\nOverall result for {label}: {total_passed}/{total_questions} ({overall_accuracy:.0%})")

    return results

def cleanup_model(tokenizer, model) -> None:
    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main() -> None:
    print("=" * 60)
    print("LLM EVALUATION: Base Model vs LoRA")
    print("=" * 60)

    show_environment()
    show_system_prompt()
    show_test_sets()

    print("\nStep 1: Evaluating base model...")
    tokenizer, base_model = load_model(use_lora=False)

    base_results = run_evaluation(
        tokenizer=tokenizer,
        model=base_model,
        label="Base",
    )

    cleanup_model(
        tokenizer=tokenizer,
        model=base_model,
    )

    print("\nStep 2: Evaluating LoRA model...")

    if not LORA_PATH.exists():
        print(f"LoRA path not found: {LORA_PATH}")
        print("Skipping LoRA evaluation.")

        save_report(
            base_results=base_results,
            lora_results={},
        )

        return

    tokenizer, lora_model = load_model(use_lora=True)

    lora_results = run_evaluation(
        tokenizer=tokenizer,
        model=lora_model,
        label="LoRA",
    )

    cleanup_model(
        tokenizer=tokenizer,
        model=lora_model,
    )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    save_report(
        base_results=base_results,
        lora_results=lora_results,
    )

if __name__ == "__main__":
    main()