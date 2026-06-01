.PHONY: install app train-ml train-bert predict test clean

install:
	pip install -r requirements.txt

app:
	streamlit run app.py

train-ml:
	python -m src.training.train --config configs/config.yaml --mode sklearn

train-bert:
	python -m src.training.train --config configs/config.yaml --mode distilbert

predict:
	python -m src.inference.predict --text "I feel hopeless and empty" --backend sklearn

test:
	pytest

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
