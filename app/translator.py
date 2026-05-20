from transformers import pipeline
from app.config import USE_MOCK, MODEL_NAME, MAX_NEW_TOKENS #settings outside translator.py
from app.router import build_prompt_by_mode  #mode select
generator = None #model doesn't load w file import

def get_generator():
    global generator
    if generator is None:
        generator = pipeline( "text-generation", model=MODEL_NAME, device_map="auto")
    return generator

def generate_mock_response(text: str, mode: str) -> str:
    return f"""
[MOCK RESPONSE]

Mode:
{mode}

Input:
{text}

Example output:
Japanese: 日本語を勉強したいです。
Romaji: Nihongo o benkyou shitai desu.
Grammar: たい means "want to do something".
Vocabulary: 日本語 = Japanese language, 勉強する = to study
"""
#main gen func
def generate_response(text: str, mode: str = "translate") -> str:
    prompt = build_prompt_by_mode(mode, text)
    if USE_MOCK:
        return generate_mock_response(text, mode)
    model = get_generator()
    result = model(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return result[0]["generated_text"]