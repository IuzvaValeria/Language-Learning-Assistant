from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512" 

TRAIN_FILE = Path("data/final/train.jsonl")
VAL_FILE = Path("data/final/val.jsonl")
OUTPUT_DIR = Path("models/n5_lora")


def format_chat_example(example):
    messages = example["messages"]

    text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            text += f"<s>[SYSTEM]\n{content}\n[/SYSTEM]\n"
        elif role == "user":
            text += f"[USER]\n{content}\n[/USER]\n"
        elif role == "assistant":
            text += f"[ASSISTANT]\n{content}\n[/ASSISTANT]</s>"

    return text


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VAL_FILE),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        args=training_args,
        formatting_func=format_chat_example,
        max_seq_length=512,
    )

    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()