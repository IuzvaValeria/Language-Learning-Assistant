from app.translator import translate_to_japanese

if __name__ == "__main__":
    text = "I want to learn Japanese."
    result = translate_to_japanese(text)
    print(result)