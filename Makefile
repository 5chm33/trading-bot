<<<<<<< HEAD
.PHONY: run train monitor test deploy clean

# Environment
setup:
	pip install -r requirements.txt
	cp .env.example .env

# Core workflows
run:
	python src/pipeline/deployment/api_server.py

train:
	python src/pipeline/training/train_rl.py

monitor:
	docker-compose -f monitoring/docker-compose.yml up -d

test:
	pytest tests/ --cov=src/ --cov-report=html

deploy:
	scripts/deploy.sh

clean:
	find . -name "*.pyc" -delete
	rm -rf __pycache__/
=======
.PHONY: run train monitor test deploy clean

# Environment
setup:
	pip install -r requirements.txt
	cp .env.example .env

# Core workflows
run:
	python src/pipeline/deployment/api_server.py

train:
	python src/pipeline/training/train_rl.py

monitor:
	docker-compose -f monitoring/docker-compose.yml up -d

test:
	pytest tests/ --cov=src/ --cov-report=html

deploy:
	scripts/deploy.sh

clean:
	find . -name "*.pyc" -delete
	rm -rf __pycache__/
>>>>>>> 60870aec3b9ed2c2cb804ceb4f1eeb5c6af9d852
