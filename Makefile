# Reproduce everything with:  make all
# Part B needs internet access (~1 GB of CESM2-LE subsets, several GB of nClimGrid).
PY ?= python

.PHONY: install test lint part-a part-a-quick part-b-data part-b part-b-fake part-b-smyle significance figures all clean

install:
	$(PY) -m pip install -e ".[all]"

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

lint:
	ruff check src tests
	ruff format --check src tests

part-a-quick:            ## smoke test, ~1 min on a laptop CPU
	$(PY) -m knownpred.experiments.part_a --quick

part-a:                  ## full Part A (baselines ~8 min CPU; with networks see docs/reproducibility.md)
	$(PY) -m knownpred.experiments.part_a --config configs/part_a.yaml

part-b-data:
	$(PY) -m knownpred.data.download --config configs/part_b.yaml

part-b:
	$(PY) -m knownpred.experiments.part_b --config configs/part_b.yaml --stage all

part-b-fake:             ## Part B pipeline on synthetic stand-in files (no download; results meaningless)
	$(PY) -m knownpred.experiments.part_b --config configs/part_b.yaml --fake-data

part-b-smyle:            ## perfect-model truth from CESM2 SMYLE hindcasts (needs thredds.rda.ucar.edu)
	$(PY) -m knownpred.experiments.part_b --config configs/part_b.yaml --stage smyle

significance:            ## Diebold-Mariano and shift tests with Holm / Benjamini-Hochberg control
	$(PY) -m knownpred.experiments.significance

figures:
	$(PY) -m knownpred.plotting

all: test part-a part-b-data part-b part-b-smyle significance figures

clean:
	rm -rf runs/*_quick runs/part_b_fake data/fake .pytest_cache .ruff_cache
