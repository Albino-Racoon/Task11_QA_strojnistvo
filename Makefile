.PHONY: install train eval api frontend test seed docker

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cd frontend && npm install

train:
	. .venv/bin/activate && PYTHONPATH=. bash scripts/setup_demo.sh

eval:
	. .venv/bin/activate && PYTHONPATH=. python training/evaluate.py

seed:
	. .venv/bin/activate && PYTHONPATH=. python api/seed_demo.py

api:
	. .venv/bin/activate && PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

test:
	. .venv/bin/activate && PYTHONPATH=. pytest tests/ -q

docker:
	docker compose up --build
