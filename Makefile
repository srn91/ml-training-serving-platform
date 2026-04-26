.PHONY: train validate test lint verify serve clean

train:
	python3 -m app.cli train

validate:
	python3 -m app.cli validate

test:
	pytest -q

lint:
	ruff check app tests

verify: lint test validate

serve:
	uvicorn app.main:app --reload

clean:
	rm -rf generated artifacts

