.PHONY: help install fetch train backtest test notebook clean

help:
	@echo "NBA Game Outcome Prediction - Make Targets"
	@echo "=========================================="
	@echo "make install   - Install dependencies from requirements.txt"
	@echo "make fetch     - Fetch NBA and Kalshi data"
	@echo "make train     - Train all three models (logistic, RF, XGBoost)"
	@echo "make backtest  - Run trading strategy backtest"
	@echo "make test      - Run pytest"
	@echo "make notebook  - Launch Jupyter Lab"
	@echo "make clean     - Remove __pycache__, .pytest_cache, etc."

install:
	pip install -r requirements.txt

fetch:
	python scripts/fetch_data.py

train:
	python scripts/train.py

backtest:
	python scripts/backtest.py

test:
	pytest tests/ -v

notebook:
	jupyter lab

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .DS_Store -delete
	rm -rf .coverage htmlcov/
