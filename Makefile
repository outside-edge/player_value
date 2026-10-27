.PHONY: sync fetch prepare compare publish rankings lint test check site ci-docker

UV ?= uv
QUARTO ?= quarto
FORMAT ?= ODI
FORMAT_DIR = $(if $(filter Test,$(FORMAT)),tests,$(if $(filter T20I,$(FORMAT)),t20is,odis))
DATA_DIR ?= data/raw/cricsheet/$(FORMAT_DIR)
PREPARE_ARGS ?=
AS_OF ?= $(shell date -u +%Y-%m-%d)
MODEL_ARGS ?=

sync:
	$(UV) sync --locked

fetch:
	$(UV) run --locked player-value fetch --format "$(FORMAT)" --output "$(DATA_DIR)"

prepare:
	$(UV) run --locked player-value prepare --format "$(FORMAT)" --input "$(DATA_DIR)" --as-of "$(AS_OF)" $(PREPARE_ARGS)

compare:
	$(UV) run --locked player-value compare $(MODEL_ARGS)

publish:
	$(UV) run --locked player-value publish

rankings:
	$(UV) run --locked player-value rank --as-of "$(AS_OF)"

lint:
	$(UV) run --locked black --check src tests
	$(UV) run --locked isort --check-only src tests
	$(UV) run --locked flake8 src tests

test:
	$(UV) run --locked pytest

check: lint test

site:
	$(UV) run --locked player-value check-release
	$(UV) run --locked player-value check-release --input out/release/current
	$(UV) run --locked player-value site-content
	$(QUARTO) render
	$(UV) run --locked player-value check-site --input out/site

ci-docker:
	docker run --rm -v "$(CURDIR):/app:ro" -w /tmp/player_value python:3.14-slim sh -c 'cp -R /app/src /app/tests /app/pyproject.toml /app/uv.lock /app/README.md /app/LICENSE /app/.flake8 /app/Makefile /app/docs . && pip install uv && make sync check'
