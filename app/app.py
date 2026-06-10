import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="Japanese Tutor", page_icon="日本語")

st.title("Japanese Learning Assistant")
st.write("Beginner-friendly Japanese assistant for translation, grammar, correction, exercises, and chat.")

mode = st.sidebar.selectbox(
    "Choose mode",
    ["translation", "grammar", "correction", "exercise", "chat"],
)

user_text = st.text_area("Your text", height=120)

if st.button("Send"):
    if not user_text.strip():
        st.warning("Please enter text.")
    else:
        response = requests.post(
            API_URL,
            json={
                "mode": mode,
                "text": user_text,
            },
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer")
            st.write(data["response"])
        else:
            st.error(f"Backend error: {response.status_code}")
            st.code(response.text)