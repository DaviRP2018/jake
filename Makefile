setup-pre-commit:
	pip install pre-commit
	pre-commit install

setup: setup-pre-commit
	pip install black
	pip install isort
