# AI-Powered Text Summarizer (`t5-small` & FastAPI)

An end-to-end Machine Learning and Web application that fine-tunes the Hugging Face `t5-small` model on the SAMSum dataset for dialogue summarization, backed by a **FastAPI** server and an interactive web frontend.

---

## 🚀 Project Overview

This repository contains everything needed to train, serve, and interact with a sequence-to-sequence transformer model:
1. **Data Science Notebook (`.ipynb`):** Data loading, cleaning, tokenization, fine-tuning `t5-small` using the Hugging Face `Trainer` API, and model artifact exporting.
2. **Backend Server (`app.py`):** A robust **FastAPI** application that loads the fine-tuned model checkpoint, preprocesses incoming user text, and exposes a RESTful API endpoint alongside a template rendering engine.
3. **Frontend UI (`index.html`):** A clean, responsive user interface allowing users to input raw dialogue or text and instantly view generated summaries.

---

## 📁 Repository Structure

```text
├── app.py                 # FastAPI backend server
├── index.html             # Frontend web interface (HTML/CSS/JS)
├── notebook.ipynb         # Jupyter Notebook (Fine-tuning pipeline)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🛠️ Tech Stack & Libraries

* **Deep Learning & NLP:** Python, PyTorch, Hugging Face `transformers` (`T5ForConditionalGeneration`, `T5Tokenizer`, `Trainer`), `Datasets`
* **Backend Framework:** FastAPI, Uvicorn, Pydantic, Jinja2 Templates
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)
* **Dataset:** SAMSum Corpus (Dialogue Summarization)

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure you have the following installed on your machine:
* Python `3.8+`
* `pip` or `conda`
* Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies
Install all necessary Python packages for PyTorch, Transformers, and FastAPI:
```bash
pip install torch transformers datasets fastapi uvicorn pandas sentencepiece jinja2
```

---

## 🏃‍♂️ Running the Project

### Step 1: Place Model Checkpoints
Ensure your fine-tuned model folder (`fine_tuned_model/`) generated from the Jupyter Notebook is placed in the root directory alongside `app.py`. It should contain:
* `model.safetensors` (or `pytorch_model.bin`)
* `config.json`
* `tokenizer.json`
* `tokenizer_config.json`
* `generation_config.json`

### Step 2: Start the FastAPI Backend
Launch the backend server using Uvicorn:
```bash
uvicorn app:app --reload
```
The server will run locally at **`http://127.0.0.1:8000`**.

### Step 3: Access the Frontend
Open your browser and navigate to:
```text
http://127.0.0.1:8000
```
Type or paste any dialogue into the text area, click **Summarize**, and view your model's real-time summary!

---

## 📊 Model Training Details (From Jupyter Notebook)
* **Base Model:** `t5-small`
* **Epochs:** 6
* **Batch Size:** 8 (Train & Eval)
* **Max Length:** 512 tokens (Input), 150 tokens (Summary/Target)
* **Hardware:** GPU Accelerated (Google Colab T4 / CUDA / MPS supported)

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and open a pull request.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.