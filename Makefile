.PHONY: test check postgres-up postgres-down smoke

test:
	pytest

check:
	python -m compileall -q aster scripts
	pytest

postgres-up:
	docker compose -f docker/compose.yml up -d

postgres-down:
	docker compose -f docker/compose.yml down -v

smoke:
	python scripts/live_smoke.py
