"""Load the released GPT-4 judgments into the harness cache.

These are real GPT-4 (0613) judgments published by LMSYS alongside MT-Bench, in
both presentation orders. They are the reason this repo can report a frontier
judge without anyone holding an API key.

Two judges come out of this file:

  gpt-4-0613-pair    pairwise, both orders, prompt templates pair-v2 and
                     pair-math-v1 (the latter is used on math, coding and
                     reasoning questions and includes a reference answer)
  gpt-4-0613-single  independent 1 to 10 grading of each response, turned into
                     a pairwise verdict by comparing the two scores. Order free
                     by construction, which makes it a useful contrast.

800 comparisons were judged twice, weeks apart, under the identical prompt.
Those become sample=1 and give a free measurement of run to run stability.
Token counts are estimated from characters (the released files carry no usage
data) and every record says so.
"""

import json
import os
import sys
from collections import defaultdict

import pandas as pd

from harness import cache

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")

PAIR_JUDGE = "gpt-4-0613-pair"
SINGLE_JUDGE = "gpt-4-0613-single"

CHARS_PER_TOKEN = 4.0  # rough, and labelled as such wherever it is used


def turn_of(judge_field):
    return 2 if "multi-turn" in str(judge_field) else 1


def positional(winner, model_1, model_2, model_first):
    """The released files name the winner as 'model_1' or 'model_2', which refer
    to the row's own field names, not to presentation order. Resolve to a model
    first, then to a position. Skipping that step silently marks everything an
    error, which is exactly the kind of bug the tests in tests.py exist to catch.
    """
    winner_model = {"model_1": model_1, "model_2": model_2}.get(winner)
    if winner_model is None:
        return "tie" if winner == "tie" else "error"
    return "first" if winner_model == model_first else "second"


def store(judge_id, template, prompt, sample, verdict, text, context, extra=None):
    key = cache.cache_key(judge_id, template, prompt, sample)
    existing = cache.get(judge_id, key)
    if existing is not None:
        # Same prompt, so the same released answer serves both contexts. This is
        # what happens when a comparison's two responses are byte identical.
        cache.add_context(existing, context)
        cache.put(judge_id, key, existing)
        return key
    record = {
        "key": key,
        "judge_id": judge_id,
        "template": template,
        "sample": sample,
        "verdict": verdict,
        "text": text,
        "in_tokens": round(len(prompt) / CHARS_PER_TOKEN),
        "out_tokens": round(len(text) / CHARS_PER_TOKEN),
        "token_counts_estimated": True,
        "latency_s": None,
        "source": "lmsys/mt-bench released judgments",
    }
    record.update(extra or {})
    cache.add_context(record, context)
    cache.put(judge_id, key, record)
    return key


def main():
    items = pd.read_parquet(os.path.join(HERE, "data", "items.parquet"))
    wanted = {}
    for row in items.itertuples():
        wanted[(row.question_id, row.turn, row.model_a, row.model_b)] = row.item_id

    # Pairwise, both orders.
    rows = [json.loads(line) for line in open(os.path.join(RAW, "gpt-4_pair.jsonl"))]
    grouped = defaultdict(list)
    for r in rows:
        lo, hi = sorted([r["model_1"], r["model_2"]])
        key = (r["question_id"], turn_of(r["judge"]), lo, hi)
        if key in wanted:
            grouped[key].append(r)

    n_pair = n_rerun = 0
    for key, runs in grouped.items():
        item_id = wanted[key]
        lo = key[2]
        # Deterministic ordering of repeat runs: oldest first, so sample 0 is stable.
        for sample, r in enumerate(sorted(runs, key=lambda x: x["tstamp"])):
            template = r["judge"][1]
            m1, m2 = r["model_1"], r["model_2"]
            for game, prompt_field, judgment_field, winner_field in (
                (1, "g1_user_prompt", "g1_judgment", "g1_winner"),
                (2, "g2_user_prompt", "g2_judgment", "g2_winner"),
            ):
                first, second = (m1, m2) if game == 1 else (m2, m1)
                order = "ab" if first == lo else "ba"
                store(
                    PAIR_JUDGE,
                    template,
                    r[prompt_field],
                    sample,
                    positional(r[winner_field], m1, m2, first),
                    r[judgment_field],
                    {"item_id": item_id, "order": order, "condition": "base"},
                    {"released_game": game},
                )
            n_pair += 1
            n_rerun += sample > 0

    # Single answer grading, turned into a pairwise verdict.
    single = [json.loads(line) for line in open(os.path.join(RAW, "gpt-4_single.jsonl"))]
    scores = {}
    graded_prompt = {}
    for r in single:
        k = (int(r["question_id"]), r["model"], int(r["turn"]))
        scores[k] = float(r["score"])
        judge_field = r["judge"] if isinstance(r["judge"], list) else json.loads(r["judge"].replace("'", '"'))
        graded_prompt[k] = (r["user_prompt"], r["judgment"], judge_field[1])

    n_single = 0
    for row in items.itertuples():
        ka = (row.question_id, row.model_a, row.turn)
        kb = (row.question_id, row.model_b, row.turn)
        if ka not in scores or kb not in scores:
            continue
        sa, sb = scores[ka], scores[kb]
        if sa < 0 or sb < 0:  # -1 marks a failed grade in the released file
            continue
        verdict = "tie" if sa == sb else ("first" if sa > sb else "second")
        prompt = "[[GRADE A]]\n" + graded_prompt[ka][0] + "\n\n[[GRADE B]]\n" + graded_prompt[kb][0]
        text = f"score_a={sa} score_b={sb}\n\nA: {graded_prompt[ka][1]}\n\nB: {graded_prompt[kb][1]}"
        store(
            SINGLE_JUDGE,
            graded_prompt[ka][2],
            prompt,
            0,
            verdict,
            text,
            {"item_id": row.item_id, "order": "ab", "condition": "base"},
            {"score_a": sa, "score_b": sb},
        )
        n_single += 1

    print(f"{PAIR_JUDGE}: {n_pair} comparisons cached in both orders ({n_rerun} are repeat runs)")
    print(f"{SINGLE_JUDGE}: {n_single} comparisons cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
