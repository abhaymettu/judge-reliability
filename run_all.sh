#!/usr/bin/env bash
# The whole pipeline, end to end, in order. `make all` does the same thing with
# dependency tracking. This script is the readable version.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}

$PY 00_download.py        # public data, no API key
$PY 01_audit.py           # audit before building anything on it
$PY 02_build_items.py     # canonical comparison table
$PY 03_cache_released.py  # released GPT-4 judgments into the judgment cache
$PY tests.py              # harness tests
$PY 05_measure.py         # every number, into output/metrics.json
$PY 06_figures.py         # figures, from metrics.json only
$PY 07_report.py          # README.md, from metrics.json only

# Step 04 is not here on purpose. It is the only step that calls a model, and
# its output is committed under data/judgments/. To rerun a judge yourself:
#
#   python 04_run_judges.py --judge-id my-judge --backend anthropic --model claude-sonnet-5
#
# See USAGE.md.
