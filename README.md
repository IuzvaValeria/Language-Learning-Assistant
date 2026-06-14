# Language Learning Assistant for Japanese Learning

## Overview

This project is a lightweight Japanese-English language learning assistant based on a fine-tuned Large Language Model (LoRA).

The system is designed for beginner learners (approximately JLPT N5 level) and supports:

* English → Japanese translation
* Japanese → English translation
* Vocabulary explanation
* Simple beginner chat

---

# Installation

Create a virtual environment:

```bash
python -m venv .venv311
```

Activate it (Windows):

```bash
.venv311\Scripts\activate
```

Install all required libraries:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The project requires **two terminals**.

## Terminal 1 (Backend)

Activate the environment:

```bash
.venv311\Scripts\activate
```

Run:

```bash
uvicorn backend.main:app
```

If everything is correct, the backend will start on:

```text
http://127.0.0.1:8000
```

Wait until the terminal prints:

```text
Model loaded successfully
Application startup complete.
```

---

## Terminal 2 (Frontend)

Activate the environment:

```bash
.venv311\Scripts\activate
```

Run:

```bash
streamlit run app/app.py
```

The interface will open in the browser.

---

# How to Test the Model

The recommended prompts are:

## Translation

```text
Translate to Japanese: This is a book.
```

```text
Translate to Japanese: I drink water.
```

```text
Translate to English: 私は学生です。
```

---

## Vocabulary

```text
Explain the N5 word: 水
```

```text
Explain the N5 word: 学生
```

---

## Chat

```text
Hello! I am learning Japanese.
```

---

# Evaluation

The project compares the base model and the LoRA-adapted model.

Reported evaluation:

| Model      | Score       |
| ---------- | ----------- |
| Base model | 4/20 (20%)  |
| LoRA model | 15/20 (75%) |

The evaluation is keyword-based and should be interpreted as a project benchmark rather than a complete measure of translation quality.

---

# Notes

* The model is intended for educational purposes.
* Translation and vocabulary are the primary supported tasks.
* Grammar explanation remains limited because the grammar dataset is relatively small.
* The first startup may take some time while the model is loaded into memory.
