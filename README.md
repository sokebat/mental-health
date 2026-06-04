# Decoding Distress: Mental Health Text Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mental-health-bgqd5euvqkdumfexbnvvwp.streamlit.app/)
![Python](https://img.shields.io/badge/python-3.12-blue)

A machine learning pipeline that classifies social media text into four mental health categories — **Normal, Depression, Suicidal, and Anxiety** — using TF-IDF feature engineering and classical ML models trained on Reddit community posts.

---

## Key Results

| Model | Accuracy | Notes |
|---|---|---|
| **SVM (LinearSVC)** | **0.8900** | Best overall — highest macro F1 |
| Random Forest | 0.8234 | Good on majority classes |
| Logistic Regression | 0.8221 | Strong linear baseline |
| Naïve Bayes | 0.7643 | Fast but weaker on minority class |

SVM reaches F1 > 0.90 on Normal and Suicidal classes. The hardest boundary is Depression ↔ Suicidal due to shared vocabulary of hopelessness and self-reference.

---

## Pipeline

1. **Data Cleaning** — drop duplicates, remove posts under 3 words, filter 99th-percentile length outliers
2. **Text Preprocessing** — Unicode normalisation, lowercase, strip HTML/URLs/mentions
3. **EDA** — text length distribution, top-15 word frequencies per class
4. **Class Balancing** — `class_weight='balanced'` + stratified 80/20 split
5. **TF-IDF Vectorisation** — 50,000 features, unigrams + bigrams (`ngram_range=(1,2)`)
6. **Model Training** — LR, Naïve Bayes, Random Forest, LinearSVC
7. **Inference** — persisted models via `joblib`; real-time classification in Streamlit

---

## Project Structure

```
mental-health/
├── app.py              # Streamlit entry point
├── src/                # Preprocessing, training, inference modules
├── models/             # Saved .pkl files (tfidf, label_encoder, models)
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory notebooks
├── reports/            # Figures and evaluation outputs
├── configs/            # YAML configuration
├── tests/              # pytest test suite
├── requirements.txt
└── .python-version     # Pins Python 3.12 for Streamlit Cloud
```

---

## Getting Started

**First-time setup** (creates virtual environment and installs dependencies):

```
setup.bat
```

**Run the app:**

```
run.bat
```

Then open: `http://localhost:8501`

**Manual setup (if batch files don't work):**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Requirements

- Python 3.12
- Windows

---

## Team

Built as part of an AI & Data Science learning journey under the guidance of **Angat Sitaula**.

| Name | GitHub |
|---|---|
| Sandip Thakuri | [@sandip-thakuri01](https://github.com/sandip-thakuri01) |
| Yubraj Parajuli | [@yubrajparajuli](https://github.com/yubrajparajuli) |
| Ashish Subedi | [@sokebat](https://github.com/sokebat) |

---

## Reference

Full write-up on Medium: [Decoding Distress: A Machine Learning Journey into Mental Health Text](https://medium.com/@sandip122/decoding-distress-a-machine-learning-journey-into-mental-health-text-8e54a242ba24)
