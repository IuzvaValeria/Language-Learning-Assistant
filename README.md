# Language Learning Assistant for Japanese Learning

This project is a lightweight Japanese-English language learning assistant based on a fine-tuned large language model.

The assistant is designed for beginner Japanese learners, especially around JLPT N5 level. It supports simple translation, vocabulary explanation, and short beginner-level interaction. The project uses a pretrained instruction model and adapts it with LoRA fine-tuning on a filtered educational dataset.

The goal is not to build a full professional translator. The goal is to create a small educational assistant that gives more consistent beginner-level responses than the original base model.

## Features

Current supported functionality:

* English to Japanese translation
* Japanese to English translation
* Vocabulary explanation
* Simple chat practice
* FastAPI backend
* Streamlit frontend
* Base model vs LoRA model evaluation

Grammar explanation was part of the project scope, but the final model did not improve reliably on grammar because the grammar dataset was too small. Exercise generation and correction are not presented as final supported features.

## Project Structure

```text
Language-Learning-Assistant/
│
├── backend/
│   ├── main.py
│   ├── llm_service.py
│   └── prompts.py
│
├── app/
│   └── app.py
│
├── prompts/
│   ├── translation_prompt.txt
│   ├── vocab_prompt.txt
│   └── system_n5.txt
│
├── training/
│   ├── filter_n5_dataset.py
│   ├── prepare_dataset.py
│   └── train_lora.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── models/
│   └── n5_lora_v2_translation_vocab_grammar/
│
├── requirements.txt
└── README.md
```

Some folders such as `data/`, `models/`, and `results/` may be ignored by Git because they can contain large files. They may need to be generated locally or provided separately.

## Technologies

The project uses:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT LoRA
* TRL SFTTrainer
* bitsandbytes
* FastAPI
* Uvicorn
* Streamlit
* requests
* python-dotenv

## Dataset

The final dataset was created from beginner-oriented Japanese-English resources.

Main data sources:

* Tatoeba sentence pairs
* JLPT N5 vocabulary list
* JLPT N5 kanji list
* Small project-prepared grammar examples

Other datasets were considered but excluded from final training because they were too advanced, noisy, or too small after filtering:

* KFTT
* JESC
* TED Talks
* Lang-8
* JMdict as a live dictionary component

The final dataset was saved in JSONL format and used for supervised fine-tuning.

Final dataset split:

```text
Training examples: 7,944
Validation examples: 883
```

Task distribution:

```text
Translation: 7,192 training examples
Vocabulary: 653 training examples
Grammar: 99 training examples
```

The dataset is intentionally dominated by translation because translation was the main target capability and the most reliable source of high-quality data.

## Model

Base model:

```text
ministral/Ministral-3b-instruct
```

Fine-tuning method:

```text
LoRA
```

The model is a transformer-based causal language model. It was not trained from scratch. Instead, LoRA was used to adapt the pretrained instruction model with a smaller number of trainable parameters.

Current LoRA adapter path used for application inference:

```text
models/n5_lora_v2_translation_vocab_grammar/
```

If another adapter version is used, update `LORA_PATH` in `backend/llm_service.py` or set it through an environment variable.

## Training Setup

Main training settings:

```text
Maximum sequence length: 512
Epochs: 3
Batch size per device: 1
Gradient accumulation steps: 8
Learning rate: 2e-4
Scheduler: cosine
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Quantization: 4-bit bitsandbytes
```

Training requires a CUDA-compatible NVIDIA GPU.

## Installation

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv311
.\.venv311\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional: check whether CUDA is available:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Expected output should show `True` and the NVIDIA GPU name.

## Environment Configuration

The backend is configured in `backend/llm_service.py`.

Important values:

```python
USE_MOCK = os.getenv("USE_MOCK", "false").lower() == "true"
BASE_MODEL = os.getenv("BASE_MODEL", "ministral/Ministral-3b-instruct")
LORA_PATH = Path(os.getenv("LORA_PATH", "models/n5_lora_v2_translation_vocab_grammar"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))
```

For real model inference, mock mode must be disabled:

```text
USE_MOCK=false
```

The backend should print:

```text
Loading base model: ministral/Ministral-3b-instruct
LoRA adapter loaded from: models\n5_lora_v2_translation_vocab_grammar
Model loaded successfully
Application startup complete.
```

If the LoRA adapter path is wrong, the application should not be considered a test of the fine-tuned model.

## Running the Application

The project uses two processes:

1. FastAPI backend
2. Streamlit frontend

### Step 1. Start the Backend

Open the first terminal in the project root:

```powershell
cd D:\Work\Code\repos\llm
.\.venv311\Scripts\activate
uvicorn backend.main:app
```

Do not close this terminal.

The backend runs at:

```text
http://127.0.0.1:8000
```

Available endpoints:

```text
GET  /
GET  /modes
POST /ask
```

Example backend request body:

```json
{
  "mode": "translation",
  "text": "Translate to Japanese: I drink water."
}
```

Do not use `--reload` for normal model testing because it may reload the model unnecessarily.

### Step 2. Start the Frontend

Open a second terminal in the project root:

```powershell
cd D:\Work\Code\repos\llm
.\.venv311\Scripts\activate
streamlit run app/app.py
```

The frontend opens at:

```text
http://localhost:8501
```

## How to Ask the Model

Use short and direct prompts. The model performs best when the task is explicit.

Recommended translation format:

```text
Translate to Japanese: I drink water.
```

```text
Translate to Japanese: This is a book.
```

```text
Translate to English: 私は学生です。
```

Recommended vocabulary format:

```text
Explain the N5 word: 水
```

```text
Explain the N5 word: 学生
```

Recommended chat format:

```text
Hello. I am learning Japanese.
```

Avoid vague prompts such as:

```text
table
```

or overly informal task names such as:

```text
translate to N5 simple japanese: ...
```

The model is more stable with the format used in the project prompts:

```text
Translate to Japanese: ...
Translate to English: ...
Explain the N5 word: ...
```

## Quick Manual Test Cases

Use these examples to check whether the application is running correctly.

### Translation Test 1

Input:

```text
Translate to Japanese: This is a book.
```

Expected output:

```text
これは本です。
```

### Translation Test 2

Input:

```text
Translate to Japanese: I drink water.
```

Expected output:

```text
私は水を飲みます。
```

### Translation Test 3

Input:

```text
Translate to English: 私は学生です。
```

Expected output:

```text
I am a student.
```

### Vocabulary Test 1

Input:

```text
Explain the N5 word: 水
```

Expected output should mention:

```text
Meaning: water
Reading: みず
```

### Vocabulary Test 2

Input:

```text
Explain the N5 word: 学生
```

Expected output should mention:

```text
Meaning: student
Reading: がくせい
```

## Important Runtime Notes

The model may take time to load at backend startup. A startup time of one or two minutes can happen depending on hardware, GPU memory, and whether files are already cached.

If the terminal shows:

```text
Some parameters are on the meta device because they were offloaded to the cpu.
```

then part of the model is being offloaded, and inference may be slow.

If port `8000` is already in use, stop the old backend process or run:

```powershell
netstat -ano | findstr :8000
taskkill /PID PID_NUMBER /F
```

Then start the backend again:

```powershell
uvicorn backend.main:app
```

## Evaluation

The evaluation compares the base model and the LoRA-adapted model on fixed prompts.

Reported keyword-based evaluation result:

```text
Base model: 4/20, 20%
LoRA model: 15/20, 75%
```

The evaluation uses keyword-based matching. This means that a correct score does not necessarily indicate perfect translation quality. Valid paraphrases may be marked wrong, and weak answers may pass if they contain the expected keyword.

The final evaluated capabilities are translation and vocabulary. Grammar explanation remains a limitation.

## Limitations

The current version has several limitations:

* The dataset is imbalanced toward translation.
* Grammar examples are too few for reliable grammar explanation.
* Evaluation is keyword-based and does not fully measure semantic correctness.
* The system is a learning prototype, not a production translator.
* The model can still produce incorrect or unnatural Japanese.
* Some prompts may produce unstable outputs if they differ strongly from the training format.
* The model should be tested with short beginner-level prompts.

## Future Work

Possible future improvements:

* Add more reliable beginner grammar explanation examples.
* Improve vocabulary explanations using JMdict or a curated dictionary.
* Add larger datasets for JLPT N4 and N3.
* Improve evaluation with semantic or human-based assessment.
* Add more structured learner exercises after reliable evaluation.

## Team Members

* Iuzva Valeria
* Bilinskaia Irina

## Project Status

This is an educational university project and a working prototype of a Japanese-English learning assistant. The strongest current capabilities are beginner translation and vocabulary support. Grammar explanation is identified as a future-work direction because the final grammar dataset was too small for reliable model behavior.
