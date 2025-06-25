.PHONY: run test clean

run:
	python src/llm.py

test:
	pytest

clean:
	find . -type d -name '__pycache__' -exec rm -r {} +