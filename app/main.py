from app.translator import generate_response
if __name__ == "__main__":
    result = generate_response(
        text="I want to learn Japanese.",
        mode="translate"
    )
    print(result)