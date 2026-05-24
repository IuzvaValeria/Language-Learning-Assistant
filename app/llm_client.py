import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    FineGrainedFP8Config,
)


MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    print("Loading local LLM model...")

    _tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)

    _model = Mistral3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )

    return _tokenizer, _model


def generate_answer(prompt: str, user_input: str) -> str:
    tokenizer, model = load_model()

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input,
                }
            ],
        },
    ]

    tokenized = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    )

    tokenized["input_ids"] = tokenized["input_ids"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            **tokenized,
            max_new_tokens=250,
        )[0]

    generated_tokens = output[len(tokenized["input_ids"][0]):]
    answer = tokenizer.decode(generated_tokens)

    return answer.strip()