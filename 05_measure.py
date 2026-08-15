"""Every number in the README. Reads the cache, writes output/metrics.json.

No network, no API key, no model. If this script needs either, it is a bug.

Scoring conventions, decided once and applied everywhere (see DECISIONS.md):

  no-ties setup   the primary one. Restricted to comparisons where the human
                  majority picked a winner and the judge picked a winner. This
                  is the setup the MT-Bench paper headlines and the one where a
                  judge, a human and a coin flip are all directly comparable.
  with-ties setup all comparisons, tie treated as a third label, exact match.

  Bootstrap resamples comparisons, never individual judgments, because the two
  presentation orders of one comparison are one draw from the world.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from harness import cache, judges, stats as hstats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

GPT4_PAIR = "gpt-4-0613-pair"
GPT4_SINGLE = "gpt-4-0613-single"
LOCAL = "qwen2.5-3b-4bit-local"

# USD per million tokens. gpt-4-0613 at the price OpenAI charged when these
# judgments were produced. The local judge is free at the point of use, so its
# cost is reported as wall clock instead.
PRICING = {GPT4_PAIR: (30.0, 60.0), GPT4_SINGLE: (30.0, 60.0)}

TIE = "tie"


def canonical(verdict, order):
    """Positional verdict -> which model won, independent of presentation order."""
    if verdict in (TIE, "error"):
        return verdict
    if order == "ab":
        return "a" if verdict == "first" else "b"
    return "b" if verdict == "first" else "a"


def load_verdicts(judge_id):
    records = cache.load_all(judge_id)
    if not records:
        return pd.DataFrame(
            columns=["item_id", "order", "template", "sample", "condition", "verdict", "choice"]
        )
    rows = []
    for r in records:
        rows.append(
            {
                "item_id": r["item_id"],
                "order": r["order"],
                "template": r["template"],
                "sample": r["sample"],
                "condition": r.get("condition", "base"),
                "verdict": r["verdict"],
                "choice": canonical(r["verdict"], r["order"]),
                "in_tokens": r["in_tokens"],
                "out_tokens": r["out_tokens"],
                "latency_s": r["latency_s"],
            }
        )
    return pd.DataFrame(rows)


def pick(table, template=None, sample=0, condition="base"):
    """item_id -> choice, for one slice of a judge's judgments, keyed by order."""
    sel = table[(table["sample"] == sample) & (table["condition"] == condition)]
    if template is not None:
        sel = sel[sel["template"] == template]
    return {(row.item_id, row.order): row.choice for row in sel.itertuples()}


def swap_average(choice_ab, choice_ba):
    """The standard mitigation: judge both orders, keep the verdict only if it
    survives the swap. Disagreement becomes a tie, which is the honest reading."""
    if choice_ab is None or choice_ba is None:
        return None
    return choice_ab if choice_ab == choice_ba else TIE


def score(items, choices, setup):
    """Per comparison hit and include arrays for one judge configuration."""
    hits, include = [], []
    for row in items.itertuples():
        got = choices.get(row.item_id)
        truth = row.human_label
        if setup == "no_ties":
            ok = got is not None and got not in (TIE, "error") and truth != TIE
            include.append(1.0 if ok else 0.0)
            hits.append(1.0 if ok and got == truth else 0.0)
        else:
            ok = got is not None and got != "error"
            include.append(1.0 if ok else 0.0)
            hits.append(1.0 if ok and got == truth else 0.0)
    return np.array(hits), np.array(include)


def agreement(items, choices, setup, label):
    point, lo, hi, n = hstats.bootstrap_ci(*score(items, choices, setup))
    return {"label": label, "setup": setup, "accuracy": point, "ci": [lo, hi], "n": n}


def human_ceiling(items, setup):
    """Pairwise agreement between annotators who judged the same comparison.

    Counted per comparison so the bootstrap can resample comparisons: a
    comparison with 4 annotators contributes 6 pairs to both numerator and
    denominator, and gets resampled as one unit.
    """
    hits, include = [], []
    for row in items.itertuples():
        votes = json.loads(row.votes)
        agree = total = 0
        for x, y in combinations(votes, 2):
            if setup == "no_ties" and (x == TIE or y == TIE):
                continue
            total += 1
            agree += x == y
        hits.append(float(agree))
        include.append(float(total))
    point, lo, hi, n = hstats.bootstrap_ci(hits, include)
    n_items = int(sum(1 for v in include if v > 0))
    return {
        "label": "human to human ceiling",
        "setup": setup,
        "accuracy": point,
        "ci": [lo, hi],
        "n": n,
        "n_comparisons": n_items,
    }


def human_vs_majority_ceiling(items, setup):
    """The other ceiling, and the fairer one.

    A judge is scored against the majority of several annotators, which is a
    denoised label. An individual annotator is not. So also score each annotator
    against the majority of the others, on comparisons with three or more votes.
    Anything else compares a judge on easy mode against humans on hard mode.
    """
    hits, include = [], []
    for row in items.itertuples():
        votes = json.loads(row.votes)
        if len(votes) < 3:
            hits.append(0.0)
            include.append(0.0)
            continue
        agree = total = 0
        for i, vote in enumerate(votes):
            rest = votes[:i] + votes[i + 1:]
            counts = Counter(rest)
            top = max(counts.values())
            winners = [k for k, v in counts.items() if v == top]
            label = winners[0] if len(winners) == 1 else TIE
            if setup == "no_ties" and (vote == TIE or label == TIE):
                continue
            total += 1
            agree += vote == label
        hits.append(float(agree))
        include.append(float(total))
    point, lo, hi, n = hstats.bootstrap_ci(hits, include)
    return {
        "label": "human ceiling, one annotator vs the majority of the others",
        "setup": setup,
        "accuracy": point,
        "ci": [lo, hi],
        "n": n,
        "n_comparisons": int(sum(1 for v in include if v > 0)),
    }


def length_choices(items):
    out = {}
    for row in items.itertuples():
        out[row.item_id] = canonical(judges.length_verdict(row.len_a, row.len_b), "ab")
    return out


def random_choices(items, seed=hstats.SEED):
    return {
        row.item_id: canonical(judges.random_verdict(f"{seed}-{row.item_id}"), "ab")
        for row in items.itertuples()
    }


def leaderboard(items, choices):
    """Win rate per model under one judge. Ties count half. This is the number a
    team would actually publish, which is why prompt sensitivity is measured here."""
    wins, games = Counter(), Counter()
    for row in items.itertuples():
        got = choices.get(row.item_id)
        if got is None or got == "error":
            continue
        games[row.model_a] += 1
        games[row.model_b] += 1
        if got == "a":
            wins[row.model_a] += 1
        elif got == "b":
            wins[row.model_b] += 1
        else:
            wins[row.model_a] += 0.5
            wins[row.model_b] += 0.5
    return {m: wins[m] / games[m] for m in games if games[m] >= 20}


def rank_of(board):
    return {m: i for i, m in enumerate(sorted(board, key=board.get, reverse=True))}


def kendall_tau(board_a, board_b):
    shared = sorted(set(board_a) & set(board_b))
    if len(shared) < 3:
        return None
    return float(stats.kendalltau([board_a[m] for m in shared], [board_b[m] for m in shared]).statistic)


def position_bias(items, table, template=None, sample=0):
    """Flip rate, direction, and what swap averaging costs and buys."""
    by_order = pick(table, template=template, sample=sample)
    ids = [row.item_id for row in items.itertuples()]
    both = [i for i in ids if (i, "ab") in by_order and (i, "ba") in by_order]
    if not both:
        return None

    flips = [by_order[(i, "ab")] != by_order[(i, "ba")] for i in both]
    flip_point, flip_lo, flip_hi, _ = hstats.bootstrap_ci([float(f) for f in flips])

    # Directional preference: how often the judge picks whichever answer was
    # shown first, counted over both presentations of every comparison.
    first_picks = decided = 0
    for i in both:
        for order in ("ab", "ba"):
            choice = by_order[(i, order)]
            if choice in ("a", "b"):
                decided += 1
                shown_first = "a" if order == "ab" else "b"
                first_picks += choice == shown_first
    return {
        "n_comparisons": len(both),
        "flip_rate": flip_point,
        "flip_ci": [flip_lo, flip_hi],
        "position_one_rate": first_picks / decided if decided else None,
        "n_decided_presentations": decided,
        "tie_rate_order_ab": float(np.mean([by_order[(i, "ab")] == TIE for i in both])),
        "tie_rate_order_ba": float(np.mean([by_order[(i, "ba")] == TIE for i in both])),
    }


def order_choices(table, template=None, sample=0, order="ab", condition="base"):
    by_order = pick(table, template=template, sample=sample, condition=condition)
    return {i: c for (i, o), c in by_order.items() if o == order}


def swapped_choices(table, template=None, sample=0, condition="base"):
    by_order = pick(table, template=template, sample=sample, condition=condition)
    ids = {i for i, _ in by_order}
    return {
        i: swap_average(by_order.get((i, "ab")), by_order.get((i, "ba")))
        for i in ids
        if by_order.get((i, "ab")) is not None and by_order.get((i, "ba")) is not None
    }


def majority_choices(table, template, n_samples, order="ab"):
    """Majority vote over repeated samples at the same order, ties on a split."""
    per_item = defaultdict(list)
    sel = table[(table["condition"] == "base") & (table["template"] == template) & (table["order"] == order)]
    for row in sel.itertuples():
        if row.sample < n_samples:
            per_item[row.item_id].append(row.choice)
    out = {}
    for item_id, votes in per_item.items():
        if len(votes) < n_samples:
            continue
        counts = Counter(votes)
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        out[item_id] = winners[0] if len(winners) == 1 else TIE
    return out


def verbosity(items, choices, label):
    """How much of the judge's behaviour is explained by which answer is longer."""
    signs, diffs, picked_longer, n = [], [], 0, 0
    for row in items.itertuples():
        got = choices.get(row.item_id)
        if got not in ("a", "b") or row.len_a == row.len_b:
            continue
        signs.append(1.0 if got == "a" else -1.0)
        diffs.append(float(row.len_a - row.len_b))
        longer = "a" if row.len_a > row.len_b else "b"
        picked_longer += got == longer
        n += 1
    if n < 10:
        return None
    r = stats.pearsonr(signs, diffs)
    hits = np.array([1.0] * picked_longer + [0.0] * (n - picked_longer))
    point, lo, hi, _ = hstats.bootstrap_ci(hits)
    return {
        "label": label,
        "n": n,
        "picks_longer_rate": point,
        "picks_longer_ci": [lo, hi],
        "corr_choice_lengthdiff": float(r.statistic),
        "corr_p": float(r.pvalue),
    }


def self_preference(items, choices, family_models, label):
    """Does the judge favour its own family more than humans do, on the same
    comparisons? Reported as a difference, with the sample size stated loudly."""
    rows = [
        row
        for row in items.itertuples()
        if (row.model_a in family_models) != (row.model_b in family_models)
        and choices.get(row.item_id) not in (None, "error")
    ]
    if len(rows) < 30:
        return {"label": label, "n": len(rows), "note": "too few comparisons to report"}

    def win_rate(getter):
        hits = []
        for row in rows:
            family_side = "a" if row.model_a in family_models else "b"
            got = getter(row)
            hits.append(1.0 if got == family_side else (0.5 if got == TIE else 0.0))
        return np.array(hits)

    judge_hits = win_rate(lambda row: choices[row.item_id])
    human_hits = win_rate(lambda row: row.human_label)
    ones = np.ones_like(judge_hits)
    diff, lo, hi = hstats.bootstrap_diff_ci(judge_hits, ones, human_hits, ones)
    return {
        "label": label,
        "n": len(rows),
        "judge_win_rate_for_family": float(judge_hits.mean()),
        "human_win_rate_for_family": float(human_hits.mean()),
        "difference": diff,
        "ci": [lo, hi],
    }


def cost_and_latency(judge_id, table):
    if table.empty:
        return None
    base = table[table["condition"] == "base"]
    in_tok, out_tok = base["in_tokens"].mean(), base["out_tokens"].mean()
    entry = {
        "judge_id": judge_id,
        "mean_input_tokens": float(in_tok),
        "mean_output_tokens": float(out_tok),
        "judgments_cached": int(len(table)),
    }
    latencies = base["latency_s"].dropna()
    if len(latencies):
        entry["median_latency_s"] = float(latencies.median())
        entry["mean_latency_s"] = float(latencies.mean())
    if judge_id in PRICING:
        pin, pout = PRICING[judge_id]
        # A deployed pairwise judge costs two calls per comparison if it swaps.
        per_call = in_tok / 1e6 * pin + out_tok / 1e6 * pout
        entry["usd_per_1000_judgments"] = float(per_call * 1000)
        entry["usd_per_1000_comparisons_swapped"] = float(per_call * 2000)
        entry["token_counts_estimated"] = True
    else:
        entry["usd_per_1000_judgments"] = 0.0
        entry["token_counts_estimated"] = False
    return entry


def main():
    os.makedirs(OUT, exist_ok=True)
    items = pd.read_parquet(os.path.join(HERE, "data", "items.parquet"))
    multi = items[items.n_votes >= 2]

    gpt4 = load_verdicts(GPT4_PAIR)
    gpt4_single = load_verdicts(GPT4_SINGLE)
    local = load_verdicts(LOCAL)

    # Only comparisons the local judge finished every condition on are analysed,
    # so a partial run cannot quietly bias a template comparison.
    local_ids = set()
    if not local.empty:
        base = local[local.condition == "base"]
        complete = base.groupby("item_id").apply(
            lambda g: g[g["sample"] == 0].groupby(["template", "order"]).size().shape[0], include_groups=False
        )
        n_conditions = base[base["sample"] == 0].groupby(["template", "order"]).size().shape[0]
        local_ids = set(complete[complete == n_conditions].index)
    if len(local_ids) < 50:
        local, local_ids = load_verdicts("__none__"), set()
    local_items = items[items.item_id.isin(local_ids)]

    metrics = {
        "dataset": {
            "comparisons": int(len(items)),
            "human_votes": int(items.n_votes.sum()),
            "annotators": len(set(v for row in items.itertuples() for v in json.loads(row.voters))),
            "questions": int(items.question_id.nunique()),
            "models": sorted(set(items.model_a) | set(items.model_b)),
            "comparisons_with_2plus_annotators": int(len(multi)),
            "human_label_counts": {k: int(v) for k, v in items.human_label.value_counts().items()},
            "local_judge_subsample": int(len(local_items)),
        }
    }

    # Judge configurations scored against the human majority.
    configs = {
        "gpt-4 pairwise, one order": order_choices(gpt4, order="ab"),
        "gpt-4 pairwise, swap averaged": swapped_choices(gpt4),
        "gpt-4 single answer grading": order_choices(gpt4_single, order="ab"),
        "length baseline": length_choices(items),
        "random baseline": random_choices(items),
    }
    if not local.empty:
        for template in sorted(local.template.unique()):
            configs[f"local 3B, {template} prompt, one order"] = order_choices(local, template=template)
        configs["local 3B, rubric prompt, swap averaged"] = swapped_choices(local, template="rubric")

    metrics["agreement"] = {}
    for setup in ("no_ties", "with_ties"):
        rows = [human_ceiling(multi, setup), human_vs_majority_ceiling(items, setup)]
        for label, choices in configs.items():
            scope = local_items if label.startswith("local") else items
            rows.append(agreement(scope, choices, setup, label))
        metrics["agreement"][setup] = rows

    # Same, restricted to the local judge's subsample, so the comparison between
    # the frontier judge and the local judge is on identical comparisons.
    metrics["agreement_on_local_subsample"] = [human_ceiling(local_items[local_items.n_votes >= 2], "no_ties")]
    for label, choices in configs.items():
        metrics["agreement_on_local_subsample"].append(
            agreement(local_items, choices, "no_ties", label)
        )

    # Kappa and alpha against the human majority.
    metrics["kappa"] = {}
    for label, choices in configs.items():
        scope = local_items if label.startswith("local") else items
        pairs = [
            (choices[row.item_id], row.human_label)
            for row in scope.itertuples()
            if choices.get(row.item_id) not in (None, "error")
        ]
        metrics["kappa"][label] = {
            "cohens_kappa": hstats.cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs]),
            "n": len(pairs),
        }

    # Krippendorff's alpha over the human annotators, and over the annotators
    # with the judge added as one more rater.
    voters = sorted({v for row in multi.itertuples() for v in json.loads(row.voters)})
    index = {v: i for i, v in enumerate(voters)}
    matrix = [[None] * len(multi) for _ in voters]
    for col, row in enumerate(multi.itertuples()):
        for voter, vote in zip(json.loads(row.voters), json.loads(row.votes)):
            matrix[index[voter]][col] = vote
    labels = ["a", "b", TIE]
    metrics["krippendorff"] = {
        "humans_only": hstats.krippendorff_alpha(matrix, labels),
        "n_raters": len(voters),
        "n_comparisons": int(len(multi)),
    }
    for label in ("gpt-4 pairwise, one order", "gpt-4 pairwise, swap averaged", "length baseline"):
        judge_row = [
            configs[label].get(row.item_id) if configs[label].get(row.item_id) in labels else None
            for row in multi.itertuples()
        ]
        metrics["krippendorff"][label] = hstats.krippendorff_alpha(matrix + [judge_row], labels)

    # Position bias.
    metrics["position_bias"] = {
        "gpt-4 pairwise": position_bias(items, gpt4),
        "gpt-4 single answer grading": {
            "note": "order free by construction: each response is graded on its own"
        },
    }
    if not local.empty:
        for template in sorted(local.template.unique()):
            metrics["position_bias"][f"local 3B, {template} prompt"] = position_bias(
                local_items, local, template=template
            )

    # GPT-4 was run twice, weeks apart, on 678 comparisons under an identical
    # prompt. That is a free measurement of run to run stability.
    rerun = pick(gpt4, sample=1)
    base = pick(gpt4, sample=0)
    shared = sorted(set(rerun) & set(base))
    if shared:
        same = [float(base[k] == rerun[k]) for k in shared]
        point, lo, hi, _ = hstats.bootstrap_ci(same)
        metrics["rerun_stability"] = {
            "judge": GPT4_PAIR,
            "n_judgments": len(shared),
            "same_verdict_rate": point,
            "ci": [lo, hi],
            "note": "same prompt, same model, two runs weeks apart",
        }

    # Verbosity.
    metrics["verbosity"] = {
        "human majority": verbosity(items, {r.item_id: r.human_label for r in items.itertuples()}, "human majority"),
        "gpt-4 pairwise, one order": verbosity(items, configs["gpt-4 pairwise, one order"], "gpt-4 pairwise, one order"),
        "gpt-4 pairwise, swap averaged": verbosity(items, configs["gpt-4 pairwise, swap averaged"], "gpt-4 pairwise, swap averaged"),
        "gpt-4 single answer grading": verbosity(items, configs["gpt-4 single answer grading"], "gpt-4 single answer grading"),
    }
    if not local.empty:
        metrics["verbosity"]["local 3B, rubric prompt"] = verbosity(
            local_items, configs["local 3B, rubric prompt, one order"], "local 3B, rubric prompt"
        )

    # Controlled padding test: same content, more words, does the verdict move?
    if not local.empty and (local.condition == "padded").any():
        padded = swapped_choices(local, template="rubric", condition="padded")
        unpadded = swapped_choices(local, template="rubric", condition="base")
        moved = []
        for row in local_items.itertuples():
            if row.item_id not in padded or row.item_id not in unpadded:
                continue
            padded_side = "a" if row.len_a <= row.len_b else "b"
            before, after = unpadded[row.item_id], padded[row.item_id]
            if before == padded_side:
                continue  # already preferred, nothing to move toward
            moved.append(float(after == padded_side))
        if moved:
            point, lo, hi, _ = hstats.bootstrap_ci(moved)
            metrics["padding_test"] = {
                "judge": "local 3B, rubric prompt, swap averaged",
                "n": len(moved),
                "moved_to_padded_rate": point,
                "ci": [lo, hi],
                "note": "shorter answer padded with content free filler to 1.5x the longer answer",
            }

    # Self preference.
    metrics["self_preference"] = {
        "gpt-4 judging gpt-4 responses": self_preference(
            items, configs["gpt-4 pairwise, swap averaged"], {"gpt-4"}, "gpt-4 judging gpt-4 responses"
        ),
        "gpt-4 judging claude-v1 responses (control)": self_preference(
            items, configs["gpt-4 pairwise, swap averaged"], {"claude-v1"}, "gpt-4 judging claude-v1 responses (control)"
        ),
        "note": "the qwen judge has no responses from its own family in this response set, so its self preference is not measurable here",
    }

    # Prompt sensitivity, read as what a team would actually ship: the leaderboard.
    human_board = leaderboard(items, {r.item_id: r.human_label for r in items.itertuples()})
    boards = {"human majority": human_board}
    for label, choices in configs.items():
        scope = local_items if label.startswith("local") else items
        boards[label] = leaderboard(scope, choices)
    metrics["leaderboards"] = {
        label: {
            "win_rates": board,
            "ranking": sorted(board, key=board.get, reverse=True),
            "kendall_tau_vs_human": kendall_tau(board, human_board),
            "top_model": max(board, key=board.get) if board else None,
        }
        for label, board in boards.items()
    }
    if not local.empty:
        local_human_board = leaderboard(local_items, {r.item_id: r.human_label for r in local_items.itertuples()})
        metrics["leaderboards"]["human majority, local subsample"] = {
            "win_rates": local_human_board,
            "ranking": sorted(local_human_board, key=local_human_board.get, reverse=True),
            "kendall_tau_vs_human": kendall_tau(local_human_board, human_board),
            "top_model": max(local_human_board, key=local_human_board.get),
        }

    # Mitigations, each measured against the same baseline on the same comparisons.
    metrics["mitigations"] = []

    def mitigation(name, scope, base_choices, new_choices):
        common = set(base_choices) & set(new_choices)
        sub = scope[scope.item_id.isin(common)]
        hb, ib = score(sub, base_choices, "no_ties")
        hn, iN = score(sub, new_choices, "no_ties")
        diff, lo, hi = hstats.bootstrap_diff_ci(hn, iN, hb, ib)
        return {
            "mitigation": name,
            "baseline_accuracy": hstats.ratio(hb, ib),
            "mitigated_accuracy": hstats.ratio(hn, iN),
            "delta": diff,
            "ci": [lo, hi],
            "n_comparisons": int(len(sub)),
            "coverage_before": float(ib.mean()),
            "coverage_after": float(iN.mean()),
        }

    metrics["mitigations"].append(
        mitigation(
            "gpt-4: swap averaging",
            items,
            order_choices(gpt4, order="ab"),
            swapped_choices(gpt4),
        )
    )
    if not local.empty:
        metrics["mitigations"] += [
            mitigation(
                "local 3B: rubric prompt instead of bare",
                local_items,
                order_choices(local, template="bare"),
                order_choices(local, template="rubric"),
            ),
            mitigation(
                "local 3B: chain of thought prompt instead of bare",
                local_items,
                order_choices(local, template="bare"),
                order_choices(local, template="cot"),
            ),
            mitigation(
                "local 3B: swap averaging",
                local_items,
                order_choices(local, template="rubric"),
                swapped_choices(local, template="rubric"),
            ),
            mitigation(
                "local 3B: majority vote over 3 samples",
                local_items,
                order_choices(local, template="rubric"),
                majority_choices(local, "rubric", 3),
            ),
        ]

    metrics["cost"] = [
        c
        for c in (
            cost_and_latency(GPT4_PAIR, gpt4),
            cost_and_latency(GPT4_SINGLE, gpt4_single),
            cost_and_latency(LOCAL, local),
        )
        if c
    ]

    path = os.path.join(OUT, "metrics.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=False)
    print(f"wrote {path}")
    for row in metrics["agreement"]["no_ties"]:
        print(f"  {row['label']:45} {row['accuracy']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]  n={row['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
