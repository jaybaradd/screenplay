.PHONY: install api ui test sample

install:
	python3 -m pip install -e '.[dev,observability]'

api:
	uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000

ui:
	streamlit run frontend/Home.py --server.port 8501

test:
	pytest

sample:
	python scripts/generate_sample.py
