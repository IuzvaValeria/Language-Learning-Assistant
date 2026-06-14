#Language Learning Assistant for Japanese Learning

##This project is a lightweight Japanese-English language learning assistant based on a fine-tuned large language model.
The assistant is designed for beginner Japanese learners, especially around JLPT N5 level. It can help with simple translation, vocabulary explanation, grammar-related questions, and short practice interaction.

The project uses a pretrained instruction model and adapts it with LoRA fine-tuning on a filtered educational dataset.

Main goal

The goal of the project is not to build a full professional translator.

The goal is to create a small educational assistant that gives more consistent beginner-level responses than the original base model.

Features

Current supported functionality:

- English to Japanese translation
- Japanese to English translation
- Vocabulary explanation
- Basic grammar-related interaction
- Simple chat practice
- FastAPI backend
- Streamlit frontend
- Base model vs LoRA model evaluation

Grammar explanation is included as a planned direction, but the final LoRA model did not improve reliably on grammar because the grammar dataset was too small.

Project structure

Language-Learning-Assistant/
│
├── backend/
│   ├── main.py
│   ├── llm_service.py
│   └── prompts.py
│
├── frontend/
│   └── app.py
│
├── training/
│   ├── parse_tatoeba.py
│   ├── parse_n5_vocab.py
│   ├── filter_n5_dataset.py
│   ├── prepare_dataset.py
│   ├── check_dataset.py
│   ├── train_lora.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
│
├── models/
│   └── n5_lora_v4_translation_vocab_grammar/
│
├── results/
│   └── lora_tests/
│
├── requirements.txt
└── README.md

Some folders such as "data/", "models/", and "results/" may be ignored by Git because they can contain large files.

Technologies

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT LoRA
- TRL SFTTrainer
- bitsandbytes
- FastAPI
- Uvicorn
- Streamlit
- requests
- python-dotenv

Dataset

The final dataset was created from beginner-oriented Japanese-English resources.

Main data sources:

- Tatoeba sentence pairs
- JLPT N5 vocabulary list
- JLPT N5 kanji list
- Small project-prepared grammar examples

Other datasets were considered but excluded from final training because they were too advanced, noisy, or too small after filtering.

Excluded or not used in final training:

- KFTT
- JESC
- TED Talks
- Lang-8
- JMdict as a live dictionary component

The final dataset was saved in JSONL format and used for supervised fine-tuning.

Final dataset split:

Training examples: 7,944
Validation examples: 883

Task distribution:

Translation: 7,192 training examples
Vocabulary: 653 training examples
Grammar: 99 training examples

Model

Base model:

ministral/Ministral-3b-instruct

Fine-tuning method:

LoRA

LoRA was used because it allows adapting a large pretrained model while training only a small number of additional parameters.

This makes the project practical for local hardware.

Training setup

Main training settings:

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

The project was developed and executed inside a Python virtual environment ("venv") to isolate project dependencies from the system Python installation.

Installation

Create and activate a virtual environment.

Windows PowerShell:

python -m venv .venv
.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Dataset preparation

Run preprocessing scripts in order.

python training/parse_tatoeba.py
python training/parse_n5_vocab.py
python training/filter_n5_dataset.py
python training/prepare_dataset.py
python training/check_dataset.py

The final files should be created in:

data/final/train.jsonl
data/final/val.jsonl

Training LoRA

Run:

python training/train_lora.py

The trained LoRA adapter will be saved to:

models/n5_lora_v4_translation_vocab_grammar/

Training requires a CUDA-compatible NVIDIA GPU.

Evaluation

Run:

python training/evaluate.py

The evaluation compares the base model and the LoRA-adapted model on fixed test prompts.

The results are saved to:

results/lora_tests/

Evaluation result:

Base model: 4/20, 20%
LoRA model: 15/20, 75%

Important note:

The evaluation uses keyword-based matching. This means that 100% accuracy on a small task does not mean perfect translation quality. It only means that the expected keyword or phrase was found in the model output.

Run backend

Start the FastAPI backend:

uvicorn backend.main:app --reload

The backend will run at:

http://127.0.0.1:8000

Available endpoints:

GET  /
GET  /modes
POST /ask

Example request body for "/ask":

{
  "mode": "translation",
  "text": "Translate to Japanese: I drink water."
}

Run frontend

In a second terminal, activate the same virtual environment and run:

streamlit run frontend/app.py

The Streamlit interface will open in the browser.

How the system works

Basic pipeline:

Raw datasets
→ parsing
→ filtering
→ instruction-style JSONL dataset
→ dataset validation with check_dataset.py
→ LoRA fine-tuning
→ evaluation
→ FastAPI backend
→ Streamlit frontend

Limitations

The current version has several limitations:

- The dataset is imbalanced toward translation.
- Grammar examples are too few for reliable grammar explanation.
- Evaluation is keyword-based and does not fully measure semantic correctness.
- The system is a learning prototype, not a production translator.
- The model can still produce incorrect or unnatural Japanese.

Future work

Possible future improvements:

- Add more reliable beginner grammar examples.
- Improve vocabulary explanations using JMdict or a curated dictionary.
- Add larger datasets for JLPT N4 and N3.
- Improve evaluation with semantic or human-based assessment.
- Add more structured exercises for learners.

Team members

- Iuzva Valeria
- Bilinskaia Irina

Project status

This is an educational university project and a working prototype of a Japanese-English learning assistant.