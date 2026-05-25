from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "ministral/Ministral-3b-instruct"
LORA_PATH = Path("models/n5_lora")

SYSTEM_PROMPT = "You are a helpful Japanese-English language learning tutor focused on JLPT N5 level."

TEST_QUESTIONS = [
    "Translate to Japanese: I hate my friends",
    "Translate to English: 時間がありません。",
    "Explain the grammar in this sentence: 私は日本語を勉強します。",
    "What does 食べる mean? Give reading and example.",
    "Correct this Japanese sentence: 私は水を飲むます。",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_auto(tokenizer, model):
    print("\nRunning automatic test questions...\n")
    for question in TEST_QUESTIONS:
        print("=" * 60)
        print(f"QUESTION: {question}")
        answer = generate(tokenizer, model, question)
        print(f"ANSWER:\n{answer}")


def run_interactive(tokenizer, model):
    print("\nType it in yourself\n")
    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("exiting")
            break
        if not user_input:
            continue
        answer = generate(tokenizer, model, user_input)
        print(f"\nTutor:\n{answer}\n")


if name == "main":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model ready.\n")

    print("Choose mode:")
    print("  1 - Auto test questions")
    print("  2 - User input")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_auto(tokenizer, model)
    elif choice == "2":
        run_interactive(tokenizer, model)
    else:
        print("Invalid choice, running auto mode")
        run_auto(tokenizer, model)