# Using the harness on your own judge

The measurement code does not know or care which model produced a verdict. To
measure a new judge you supply a backend and a judge id, and everything else,
the swap, the cache, the bootstrap, the figures, the README, follows.

## Reproduce the published numbers

```bash
pip install -r requirements.txt
make all
```

No API key. `make all` downloads the public data, rebuilds the comparison table,
loads the released GPT-4 judgments into the cache, runs the tests, and rewrites
every number in `README.md` from `output/metrics.json`.

## Add a judge that already has an API backend

Anthropic, OpenAI and Ollama backends ship with the harness.

```bash
export ANTHROPIC_API_KEY=...
python 04_run_judges.py \
    --judge-id claude-sonnet-5 \
    --backend anthropic \
    --model claude-sonnet-5 \
    --n 300
```

```bash
python 04_run_judges.py --judge-id llama3.1-8b --backend ollama --model llama3.1:8b --n 300
```

Then add the judge id to the `LOCAL` style constants at the top of
`05_measure.py` and rerun `make analysis`. Only that one file knows the names of
the judges being reported.

Useful flags:

| Flag | What it does |
| --- | --- |
| `--n` | how many comparisons to judge, stratified by category and turn, seeded |
| `--templates` | which prompt templates to run, default `bare,rubric,cot` |
| `--samples` | samples per comparison for the majority vote mitigation, default 3 |
| `--no-padding` | skip the controlled verbosity test |
| `--limit` | stop after this many model calls, for a cheap smoke test |

Every call is cached by a hash of the exact prompt, so interrupting the run and
restarting it costs nothing. Re-running a finished configuration is free.

## Add a judge with a backend that does not ship here

One class, one method. Put it in `harness/judges.py`:

```python
class MyBackend:
    def __init__(self, model):
        self.model = model

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        ...
        return {
            "text": raw_model_output,   # the harness parses [[A]] / [[B]] / [[C]] out of this
            "in_tokens": 1234,
            "out_tokens": 12,
            "latency_s": 1.8,
        }


BACKENDS["mine"] = MyBackend
```

That is the whole contract. `harness.judges.run` wraps it with the cache and the
verdict parser, so you cannot accidentally skip either.

## Judge your own comparisons instead of MT-Bench

`data/items.parquet` is the only thing the judging step reads. Write your own
with these columns and the rest of the pipeline works unchanged:

| Column | Meaning |
| --- | --- |
| `item_id` | unique, stable string |
| `question` | the user question being answered |
| `answer_a`, `answer_b` | the two responses, in a canonical order that is **not** presentation order |
| `prior_turns` | JSON list of `[user, answer_a, answer_b]` for earlier turns, `[]` if single turn |
| `human_label` | `a`, `b` or `tie` |
| `votes`, `voters` | JSON lists of individual human votes and annotator ids |
| `n_votes` | number of human votes |
| `model_a`, `model_b`, `category`, `turn` | grouping fields used by the leaderboard and the subsample |
| `len_a`, `len_b` | response lengths in characters, for the length baseline |

The one rule the harness enforces on itself: `a` and `b` identify models, never
positions. Presentation order is introduced only in `build_prompt`, and
`tests.py` fails if the two orders are not genuine swaps of each other.

## What to look at first

Measure these three before trusting any judge, in this order:

1. The length baseline. If your judge does not clearly beat "prefer the longer
   answer", it is measuring verbosity.
2. The human ceiling. Agreement means nothing without it.
3. The flip rate under order reversal, next to the judge's rerun to rerun
   disagreement on an identical prompt. If they are the same size, what you have
   is noise, not position bias, and swap averaging is buying you variance
   reduction rather than debiasing.
