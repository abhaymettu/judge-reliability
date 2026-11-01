# AGENTS.md

Notes for coding agents working in this repo. Read this before you change anything.

## What the repo does

It measures an LLM judge as a rater. The ground truth is the MT-Bench human
judgments. The judge outputs are the GPT-4 judgments LMSYS released with MT-Bench,
plus a local Qwen2.5 3B judge, all cached in `data/judgments/`, so the analysis
runs without calling a model.

## Layout

| Path | Role |
| --- | --- |
| `00_download.py` | fetch the public human judgments (network) |
| `01_audit.py` | audit the raw data, writes `output/audit.md` |
| `02_build_items.py` | build `data/items.parquet`, one row per comparison |
| `03_cache_released.py` | load the released GPT-4 judgments into the cache |
| `04_run_judges.py` | run a new judge and fill the cache (needs a model) |
| `05_measure.py` | every metric, written to `output/metrics.json` |
| `06_figures.py` | figures in `output/figures/`, from `metrics.json` |
| `07_report.py` | writes `README.md` from `metrics.json` and checks the hand written docs |
| `harness/` | prompts, cache, judge backends, statistics |
| `tests.py` | swap logic, bootstrap, kappa, cache key, verdict parsing |

## Pipeline order

- `make analysis`: `tests.py`, then `05_measure.py`, `06_figures.py`, `07_report.py`.
  Offline, from the committed cache. Use this almost always.
- `make all`: also downloads and rebuilds the data first (`00` to `03`). Needs network.
- `make judge`: runs `04_run_judges.py`. Needs a model and is never part of `make all`.

## How numbers reach the prose

1. `05_measure.py` computes everything and writes `output/metrics.json`.
2. `07_report.py` reads `metrics.json` and builds `README.md` as a string. It
   computes nothing. A number that is not in `metrics.json` cannot go in the README.
3. **Never edit `README.md` by hand.** The next `make analysis` overwrites it. To
   change the README, edit the template in `07_report.py` and rerun.

## The drift check

`LIMITATIONS.md` and `DECISIONS.md` are hand written and quote numbers. Each quoted
number is registered in `check_static_docs()` in `07_report.py` as a phrase built
from `metrics.json`. If a phrase stops appearing in its file, the build fails and
names it. When you add a number to either file, register it there. When the data
changes and the check fails, fix the prose, not the check.

`07_report.py` also refuses to write a README that contains an em dash or an en
dash.

## Adding a judge

Follow [USAGE.md](USAGE.md):

1. Use a shipped backend (Anthropic, OpenAI, Ollama), or add a class with one
   method to `harness/judges.py`.
2. `python 04_run_judges.py --judge-id <id> --backend <backend> --model <model> --n 300`.
   Every call is cached by a hash of the prompt, so a rerun is free.
3. Add the judge id next to the `LOCAL` constant at the top of `05_measure.py`.
   That file is the only one that knows which judges are reported.
4. `make analysis`.

## House rules

- No em dashes or en dashes anywhere: prose, code comments, commit messages.
- Never invent a number. If `metrics.json` does not have it, it does not get quoted.
- No API calls in `make analysis`. Keep it offline.
- Run `make analysis` before committing, so the README, figures and metrics agree.
