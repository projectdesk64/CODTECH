# CODTECH

A collection of four Codtech Internship tasks. This README provides a single, high‑level overview with quick start instructions and links to each task’s own README for full details.

## Contents
- Task 1: Weather Analysis Dashboard
- Task 2: Automated Data Profiling Report Generator
- Task 3: AI Chatbot (Streamlit + CLI)
- Task 4: Email Spam Classifier (Notebook)

## Repository Structure
```
CODTECH/
├── Task1/   # Weather Analysis Dashboard (Python script + figures/data outputs)
├── Task2/   # Automated PDF Data Report Generator (Python)
├── Task3/   # Hybrid AI Chatbot (Streamlit web app + CLI)
└── Task4/   # Email Spam Classifier (Jupyter notebook)
```

## Prerequisites
- Python 3.8+ (3.12 recommended for Task 3)
- pip
- Internet connection (for Task 1 API calls, Task 3 model downloads, Task 4 dataset)

## Quick Start

### Task 1 – Weather Analysis Dashboard
- Purpose: Fetch real‑time weather (OpenWeatherMap) for multiple cities and generate CSVs and visual dashboards.
- Location: `Task1/`
- Setup:
  1) `pip install -r Task1/requirements.txt`
  2) Create `Task1/.env` with `OPENWEATHER_API_KEY=your_api_key`
- Run:
```
python Task1/fetch_weather.py
```
- Outputs: CSVs in `Task1/outputs/data/` and figures in `Task1/outputs/figures/`.
- Details: See `Task1/README.md`.

### Task 2 – Automated Data Profiling Report Generator
- Purpose: Generate professional PDF reports (statistics + visuals, optional AI summary) from CSV/Excel/JSON.
- Location: `Task2/`
- Setup:
```
pip install -r Task2/requirements.txt
```
- (Optional) AI Summary: create `Task2/.env` with `OPENROUTER_API_KEY="your_api_key"`.
- Run (auto-detect latest data file):
```
python Task2/report_generator.py
```
- With AI summary:
```
python Task2/report_generator.py --use-ai
```
- Specify input/output:
```
python Task2/report_generator.py -i data.csv -o my_report.pdf
```
- Output: PDFs in `Task2/reports/`.
- Details: See `Task2/README.md`.

### Task 3 – AI Chatbot (Streamlit + CLI)
- Purpose: Hybrid chatbot combining spaCy rules with a Transformers model (BlenderBot) for conversation.
- Location: `Task3/`
- Setup (recommended virtual env):
```
python -m venv Task3/venv
Task3/venv/Scripts/Activate.ps1   # Windows PowerShell
pip install -r Task3/requirements.txt
python -m spacy download en_core_web_sm
```
- Run (web app):
```
streamlit run Task3/app.py
```
- Run (CLI):
```
python Task3/chat_bot.py
```
- Details: See `Task3/README.md`.

### Task 4 – Email Spam Classifier (Notebook)
- Purpose: Train and evaluate a TF‑IDF + MultinomialNB spam classifier.
- Location: `Task4/`
- Setup:
```
pip install pandas scikit-learn jupyter
```
- Run:
```
jupyter notebook
```
Open `Task4/Task4.ipynb` and run all cells. Dataset is downloaded at runtime.
- Details: See `Task4/README.md`.

## Notes
- Each task is self-contained with its own dependencies; prefer per-task virtual environments.
- Large downloads may occur for NLP models (Task 3) and external datasets (Task 4).
- API keys (Task 1, optional Task 2) should be stored in a `.env` file inside the respective task folder.




