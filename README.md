# Language Learning Assistant

A lightweight application for learning Japanese from English.

The project helps users practice basic Japanese through:

* English to Japanese translation;
* Japanese to English translation;
* vocabulary practice;
* simple grammar explanations;
* example sentences.

The assistant is focused on beginner-level Japanese, mainly JLPT N5.
It uses prepared learning data from sentence and dictionary datasets.

## Main features

* Translate simple sentences
* Explain basic Japanese grammar
* Show word meanings
* Give short example sentences
* Help with beginner Japanese practice

## Tech stack

* Python
* Transformer-based language model
* LoRA fine-tuning
* JSONL datasets
* Local model testing

## Datasets

The project uses cleaned and filtered language-learning data from sources such as Tatoeba and JMdict.


# Run the project

## 1. Create and activate a virtual environment
```bash
python -m venv .venv311
```

Windows:

```bash
.venv311\Scripts\activate
```

---

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Start the backend
```bash
$env:USE_MOCK="false"
python -m uvicorn backend.main:app --reload
```

The backend will be available at:

```
http://127.0.0.1:8000
```

API documentation:

```
http://127.0.0.1:8000/docs
```

---

## 4. Start the application

Open a second terminal:
```bash
streamlit run app/app.py
```

The application will open automatically in your browser.

---

## 5. Run LoRA tests
```bash
python training/test_lora.py
```


## Goal

The goal of this project is to create a simple and practical tool for beginner Japanese learners.
