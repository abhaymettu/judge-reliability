"""Audit the raw data before anything is built on top of it.

Writes output/audit.md. Nothing downstream is allowed to assume a property that
is not checked here. The checks that matter most:

  - do the released GPT-4 judgments cover every human labelled comparison?
  - how many annotators per comparison, and how many comparisons have only one?
  - are there degenerate responses (empty, error strings) in the answer set?
  - do the two orders of each GPT-4 judgment actually exist?
"""

import collections
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "output")


def item_key(question_id, model_a, model_b, turn):
    """Identity of a comparison, independent of presentation order."""
    lo, hi = sorted([model_a, model_b])
    return (int(question_id), lo, hi, int(turn))


def turn_of(judge_field):
    return 2 if "multi-turn" in str(judge_field) else 1


def main():
    os.makedirs(OUT, exist_ok=True)
    lines = ["# Data audit", ""]
    problems = []

    human = pd.read_parquet(os.path.join(RAW, "human.parquet"))
    human["item"] = [
        item_key(q, a, b, t)
        for q, a, b, t in zip(human.question_id, human.model_a, human.model_b, human.turn)
    ]

    lines += [
        "## MT-Bench human judgments (lmsys/mt_bench_human_judgments, split `human`)",
        "",
        f"- {len(human)} individual votes",
        f"- {human.item.nunique()} distinct comparisons (question, model pair, turn)",
        f"- {human.judge.nunique()} annotators",
        f"- {human.question_id.nunique()} questions, {len(set(human.model_a) | set(human.model_b))} models",
        f"- turns: {dict(sorted(human.turn.value_counts().items()))}",
        f"- vote labels: {dict(human.winner.value_counts())}",
        "",
    ]

    per_item = collections.Counter(human.item)
    dist = collections.Counter(per_item.values())
    lines += [
        "Annotators per comparison:",
        "",
        "| annotators | comparisons |",
        "| --- | --- |",
    ]
    for k in sorted(dist):
        lines.append(f"| {k} | {dist[k]} |")
    singles = dist.get(1, 0)
    lines += [
        "",
        f"{singles} of {len(per_item)} comparisons ({100 * singles / len(per_item):.0f} percent) have a "
        "single annotator, so they contribute to judge agreement but cannot contribute to the "
        "human to human ceiling.",
        "",
    ]

    # Degenerate responses.
    def final_answer(conv, turn):
        return conv[2 * turn - 1]["content"]

    empty = 0
    for _, row in human.iterrows():
        for conv in (row.conversation_a, row.conversation_b):
            if not final_answer(conv, row.turn).strip():
                empty += 1
    lines += [f"- empty final responses in the human split: {empty}", ""]
    if empty:
        problems.append(f"{empty} empty responses found in the human split")

    # GPT-4 pairwise judgments, both orders.
    pair = [json.loads(line) for line in open(os.path.join(RAW, "gpt-4_pair.jsonl"))]
    pair_items = {
        item_key(r["question_id"], r["model_1"], r["model_2"], turn_of(r["judge"])): r for r in pair
    }
    human_items = set(human.item)
    covered = human_items & set(pair_items)
    g1 = collections.Counter(r["g1_winner"] for r in pair)
    g2 = collections.Counter(r["g2_winner"] for r in pair)
    missing_order = sum(1 for r in pair if not r.get("g1_judgment") or not r.get("g2_judgment"))
    lines += [
        "## Released GPT-4 pairwise judgments (LMSYS mt-bench Space, June 2023 revision)",
        "",
        f"- {len(pair)} judged comparisons, {len(pair_items)} distinct",
        f"- covers {len(covered)} of {len(human_items)} human labelled comparisons",
        f"- order 1 verdicts: {dict(g1)}",
        f"- order 2 verdicts: {dict(g2)}",
        f"- rows missing one of the two orders: {missing_order}",
        "",
    ]
    if len(covered) != len(human_items):
        problems.append(
            f"GPT-4 pairwise covers only {len(covered)}/{len(human_items)} human comparisons"
        )
    if missing_order:
        problems.append(f"{missing_order} GPT-4 rows are missing an order, position bias would be biased")

    # GPT-4 single answer grading.
    single = [json.loads(line) for line in open(os.path.join(RAW, "gpt-4_single.jsonl"))]
    graded = {(int(r["question_id"]), r["model"], int(r["turn"])) for r in single}
    needed = {
        (q, m, t)
        for q, a, b, t in zip(human.question_id, human.model_a, human.model_b, human.turn)
        for m in (a, b)
    }
    missing_models = collections.Counter(m for _, m, _ in (needed - graded))
    single_items = sum(
        1
        for k in human_items
        if (k[0], k[1], k[3]) in graded and (k[0], k[2], k[3]) in graded
    )
    lines += [
        "## Released GPT-4 single answer grades",
        "",
        f"- {len(single)} graded responses",
        f"- both responses graded for {single_items} of {len(human_items)} human comparisons",
        f"- models with no grades: {dict(missing_models)}",
        "",
    ]

    questions = [json.loads(line) for line in open(os.path.join(RAW, "question.jsonl"))]
    cats = collections.Counter(q["category"] for q in questions)
    lines += [
        "## Questions",
        "",
        f"- {len(questions)} questions across {len(cats)} categories: {dict(cats)}",
        "",
    ]

    lines += ["## Problems found", ""]
    lines += [f"- {p}" for p in problems] or ["- none"]
    lines.append("")

    with open(os.path.join(OUT, "audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[-8:]))
    print(f"wrote {os.path.join(OUT, 'audit.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
