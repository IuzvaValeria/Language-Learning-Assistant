from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

MODEL_NAME = "mistralai/Ministral-3b-instruct-2512"

TRAIN_FILE = Path("data/final/train.jsonl")
VAL_FILE = Path("data/final/val.jsonl")
OUTPUT_DIR = Path("models/n5_lora")

MAX_SEQ_LENGTH = 512
SEED = 42

def format_chat_example(example, tokenizer):

    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return text

def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")

    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Using bf16: {use_bf16}, fp16: {use_fp16}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,  
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",   
            "up_proj",     
            "down_proj",   
        ],
    )

    def formatting_func(example):
        return format_chat_example(example, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,              
        per_device_train_batch_size=4,   
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,   
        learning_rate=2e-4,
        lr_scheduler_type="cosine",      
        warmup_ratio=0.05,              
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss",
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=4,
        report_to="none",                
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=load_dataset(
            "json",
            data_files={"train": str(TRAIN_FILE)},
            split="train",
        ),
        eval_dataset=load_dataset(
            "json",
            data_files={"validation": str(VAL_FILE)},
            split="validation",
        ),
        peft_config=lora_config,
        args=training_args,
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=4,              
    )

    print("Starting training...")
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print(f"To load: model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
