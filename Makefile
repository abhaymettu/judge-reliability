PY ?= python

.PHONY: all analysis data judge test figures report clean clean-outputs help

## all: everything, from an empty checkout to the finished README
all: data analysis

## data: download the public source data and build the comparison table
data: data/items.parquet

data/raw/human.parquet:
	$(PY) 00_download.py

data/items.parquet: data/raw/human.parquet 00_download.py 01_audit.py 02_build_items.py
	$(PY) 01_audit.py
	$(PY) 02_build_items.py
	$(PY) 03_cache_released.py

## analysis: every number and figure in the README, from the committed cache.
## No API key, no model call, no network.
analysis: test
	$(PY) 05_measure.py
	$(PY) 06_figures.py
	$(PY) 07_report.py

## judge: run a judge over the comparisons and fill the cache. Needs a model.
## Not part of `make all`: the cache is committed.
judge:
	$(PY) 04_run_judges.py

## test: the harness tests. Run before any analysis, which is why analysis depends on it.
test:
	$(PY) tests.py

## figures: redraw the figures from output/metrics.json
figures:
	$(PY) 06_figures.py

## report: rewrite README.md from output/metrics.json
report:
	$(PY) 07_report.py

## clean-outputs: delete generated outputs, keep the downloads and the judgment cache
clean-outputs:
	rm -rf output

## clean: also delete the downloads and the derived table. Keeps data/judgments,
## which is the paid for part and is committed.
clean: clean-outputs
	rm -rf data/raw data/items.parquet

help:
	@grep -E '^## ' Makefile | sed 's/## /  /'
