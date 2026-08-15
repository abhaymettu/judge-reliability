"""Build the canonical comparison table: data/items.parquet.

One row per comparison, identified by (question_id, turn, model_lo, model_hi).
Presentation order is stripped out here and reintroduced deliberately later,
which is the only way position bias measurement can be trusted: if order lived
in the item identity, an order bug would be invisible.

`a` always means the alphabetically first model, never "the one shown first".
"""

import json
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")


def majority(votes):
    """Human majority label over a/b/tie. Ties in the vote count resolve to 'tie',
    which is the honest reading: the annotators did not agree on a winner."""
    counts = Counter(votes)
    top = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top)
    return winners[0] if len(winners) == 1 else "tie"


def main():
    human = pd.read_parquet(os.path.join(RAW, "human.parquet"))
    questions = {q["question_id"]: q for q in (json.loads(l) for l in open(os.path.join(RAW, "question.jsonl")))}

    items = {}
    for row in human.itertuples():
        lo, hi = sorted([row.model_a, row.model_b])
        key = (int(row.question_id), int(row.turn), lo, hi)
        flipped = row.model_a != lo
        conv_lo = row.conversation_b if flipped else row.conversation_a
        conv_hi = row.conversation_a if flipped else row.conversation_b
        vote = {"model_a": "a", "model_b": "b", "tie": "tie"}[row.winner]
        if flipped and vote in ("a", "b"):
            vote = "b" if vote == "a" else "a"

        entry = items.get(key)
        if entry is None:
            question = questions[int(row.question_id)]
            prior = [
                [conv_lo[2 * i]["content"], conv_lo[2 * i + 1]["content"], conv_hi[2 * i + 1]["content"]]
                for i in range(int(row.turn) - 1)
            ]
            entry = items[key] = {
                "item_id": f"q{row.question_id}-t{row.turn}-{lo}-{hi}",
                "question_id": int(row.question_id),
                "turn": int(row.turn),
                "category": question["category"],
                "model_a": lo,
                "model_b": hi,
                "question": conv_lo[2 * (int(row.turn) - 1)]["content"],
                "prior_turns": json.dumps(prior, ensure_ascii=False),
                "answer_a": conv_lo[2 * int(row.turn) - 1]["content"],
                "answer_b": conv_hi[2 * int(row.turn) - 1]["content"],
                "votes": [],
                "voters": [],
            }
        entry["votes"].append(vote)
        entry["voters"].append(row.judge)

    rows = []
    for entry in items.values():
        votes = entry.pop("votes")
        voters = entry.pop("voters")
        entry["n_votes"] = len(votes)
        entry["n_a"] = votes.count("a")
        entry["n_b"] = votes.count("b")
        entry["n_tie"] = votes.count("tie")
        entry["human_label"] = majority(votes)
        entry["votes"] = json.dumps(votes)
        entry["voters"] = json.dumps(voters)
        entry["len_a"] = len(entry["answer_a"])
        entry["len_b"] = len(entry["answer_b"])
        entry["words_a"] = len(entry["answer_a"].split())
        entry["words_b"] = len(entry["answer_b"].split())
        rows.append(entry)

    df = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
    out = os.path.join(HERE, "data", "items.parquet")
    df.to_parquet(out, index=False)

    print(f"{len(df)} comparisons -> {out}")
    print(f"human labels: {dict(df.human_label.value_counts())}")
    print(f"with 2+ annotators: {(df.n_votes >= 2).sum()}")
    print(f"turns: {dict(df.turn.value_counts())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
