from app.assistant import answer_user, answer_with_llm
def main():
    current_mode = None
    print("\nHey hello, I'm your Japanese-English learning assistant!")
    print("\nI can help with:")
    print("1. Translation")
    print("2. Grammar explanation")
    print("3. Vocabulary explanation")
    print("4. Mistake correction")
    print("5. Example sentences")
    print()
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        user_input_lower = user_input.lower().strip()

        if user_input_lower in ["exit", "quit"]:
            print("Assistant: Bye!")
            break

        if user_input_lower in ["translate", "translation", "1"]:
            current_mode = "translate"
            print("\nAssistant:")
            print(answer_user("translate"))
            continue

        if user_input_lower in ["grammar", "2"]:
            current_mode = "grammar"
            print("\nAssistant:")
            print(answer_user("grammar"))
            continue

        if user_input_lower in ["vocabulary", "word", "explain", "3"]:
            current_mode = "vocabulary"
            print("\nAssistant:")
            print(answer_user("vocabulary"))
            continue

        if user_input_lower in ["correct", "mistake", "correction", "4"]:
            current_mode = "correction"
            print("\nAssistant:")
            print(answer_user("correct"))
            continue

        if user_input_lower in ["examples", "example", "sentences", "5"]:
            current_mode = "examples"
            print("\nAssistant:")
            print(answer_user("examples"))
            continue

        if current_mode == "translate":
            print("\nAssistant:")
            print(answer_with_llm("translate", user_input))
            continue

        if current_mode == "grammar":
            print("\nAssistant:")
            print(answer_with_llm("grammar", user_input))
            continue

        if current_mode == "vocabulary":
            print("\nAssistant:")
            print(answer_with_llm("vocabulary", user_input))
            continue

        if current_mode == "correction":
            print("\nAssistant:")
            print(answer_with_llm("correction", user_input))
            continue

        if current_mode == "examples":
            print("\nAssistant:")
            print(answer_with_llm("examples", user_input))
            continue

        response = answer_user(user_input)
        print("\nAssistant:")
        print(response)
if __name__ == "__main__":
    main()