.PHONY: train run test

train:
	python src/train.py

run:
	python -m uvicorn api.app:app --reload --port 8000

test:
	python -m pytest -q
