.PHONY: install lint test train clean all help

help:
	@echo "Available make commands:"
	@echo "  make install  - Upgrade pip and install all pinned dependencies"
	@echo "  make lint     - Run flake8 linter with max-line-length 100"
	@echo "  make test     - Run all unit and quality gate tests via pytest"
	@echo "  make train    - Execute MLflow training, CV tracking, and registry promotion"
	@echo "  make clean    - Remove bytecode, test caches, and temporary files"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src tests --max-line-length=100

test:
	pytest tests/ -v

train:
	python src/train.py

clean:
	python -c "import shutil, pathlib, glob, os; [shutil.rmtree(p, ignore_errors=True) for p in glob.glob('**/__pycache__', recursive=True) + glob.glob('**/.pytest_cache', recursive=True) + glob.glob('**/.coverage', recursive=True)]; [os.remove(f) for f in glob.glob('**/*.pyc', recursive=True) if os.path.exists(f)]"
