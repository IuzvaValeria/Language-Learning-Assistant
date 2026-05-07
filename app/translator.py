from transformers import pipeline

translator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto"
)

prompt = """
You are an English-to-Japanese translator for language learners.

Translate this sentence to Japanese and explain it simply.

English: I want to learn Japanese.

Return:
Japanese:
Romaji:
Grammar:
Vocabulary:
"""

result = translator(
    prompt,
    max_new_tokens=300,
    do_sample=False
)

print(result[0]["generated_text"])