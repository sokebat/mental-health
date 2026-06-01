# Mental Health Classifier

A Streamlit app that classifies mental health text using trained ML models.

---

## First-Time Setup

Run this **once** to create the virtual environment and install dependencies:

```
setup.bat
```

---

## Run the App

Every time you want to start the app:

```
run.bat
```

Then open your browser at:

```
http://localhost:8501
```

---

## Manual (if batch files don't work)

```
venv\Scripts\activate
streamlit run app.py
```

---

## Retrain the Models (optional)

The models are already trained and saved in `models/sklearn/`. Only run this if you want to retrain from scratch.

```
venv\Scripts\activate
python -m src.training.train --config configs/config.yaml
```

Config file: `configs/config.yaml` — edit it to change TF-IDF settings, model hyperparameters, or data paths.

---

## Requirements

- Python 3.11
- Windows
