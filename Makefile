# Macromancer — common tasks. Override the interpreter with `make PY=python3.11 ...`.
PY ?= python
PIP ?= $(PY) -m pip

.PHONY: help setup run ui test cov eval train lint docker docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create a venv and install dependencies
	$(PY) -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with:  source venv/bin/activate"

run:  ## Start the FastAPI backend (http://localhost:8000/docs)
	$(PY) run.py

ui:  ## Start the Streamlit dashboard (http://localhost:8501)
	$(PY) -m streamlit run frontend/app.py

test:  ## Run the test suite
	$(PY) -m pytest -q

cov:  ## Run tests with a coverage report
	$(PY) -m pytest -q --cov=backend --cov-report=term-missing

eval:  ## Evaluate the recommender vs baselines (writes ml/eval_results.*)
	$(PY) -m scripts.evaluate_recommender

train:  ## Retrain the XGBoost recommender on synthetic data
	$(PY) -m ml.train_model

lint:  ## Static check with pyflakes (a few known SQLAlchemy false-positives)
	-$(PY) -m pyflakes backend data ml scripts tests

docker:  ## Build + run the full stack in one container (seeded demo)
	docker compose up --build

docker-down:  ## Stop the container and remove the volume
	docker compose down -v

clean:  ## Remove caches and local build artifacts
	rm -rf .pytest_cache .coverage htmlcov **/__pycache__ */__pycache__ __pycache__
