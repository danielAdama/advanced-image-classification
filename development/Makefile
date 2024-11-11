.PHONY: initial program setup start_dev


compose_up:
	@echo "Running docker compose..."
	docker-compose up -d

compose_down:
	@echo "Shutting down docker compose..."
	docker-compose down

start_dev:
	@echo "Running app in development.."
	uvicorn src.main:app --reload --host 0.0.0.0 --port 80

start_prod:
	@echo "Running app in production..."
	gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app -b 0.0.0.0:80