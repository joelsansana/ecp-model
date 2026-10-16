.PHONY: help data data-refresh model backtest report test lint fmt clean

# Prefer the local virtualenv when there is one, so `make` works without
# activating it first. Override with `make PY=python3`.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

data:          ## pull every public series into data/raw + data/interim
	$(PY) -m ecpmodel.ingest.run_all

data-refresh:  ## same, but ignore the Comext per-cell cache and refetch
	$(PY) -m ecpmodel.ingest.run_all --refresh

model:         ## fit the ethylene state-space model -> data/processed/ecp_latent.parquet
	$(PY) -m ecpmodel.models.ecm_ssm

backtest:      ## rolling-origin evaluation -> data/processed/backtest_raw.parquet
	$(PY) -m ecpmodel.eval.backtest

report:        ## plots + monthly markdown summary -> data/processed/
	$(PY) -m ecpmodel.viz.plots

test:          ## unit tests
	$(PY) -m pytest -q

lint:          ## ruff
	$(PY) -m ruff check src tests

fmt:           ## ruff format
	$(PY) -m ruff format src tests

clean:
	rm -rf data/interim/* data/processed/*
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +