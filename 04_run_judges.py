"""Run a judge over the comparison set and fill data/judgments/.

This is the only step that calls a model. `make analysis` never runs it: the
cache is committed, so every number in the README reproduces without it.

The default judge is a local 7B through mlx-lm, chosen so the repo has a judge
that anyone can rerun end to end with no API key and no bill. Point it at an
API model with --backend/--model, see USAGE.md.

Conditions run per comparison:

    templates x {order ab, order ba}          prompt sensitivity, position bias
    default template, samples 1 and 2, t=0.7  majority vote mitigation
    default template, shorter answer padded   verbosity bias, quality held fixed

Sampling is a seeded, category and turn stratified subsample, because a local
model judging every comparison under every condition takes hours. The GPT-4
judge is not subsampled: its judgments were already paid for.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

from harness import cache, judges, prompts

HERE = os.path.dirname(os.path.abspath(__file__))

# Content free filler. It restates nothing and adds no information, so a judge
# that moves toward the padded answer is responding to length alone. A judge
# that moves away is penalising padding, which is correct behaviour.
FILLER = (
    "It is worth restating the above at greater length, since a fuller treatment "
    "can be useful. The response given addresses the question that was asked, and "
    "the considerations involved are the ones already set out. Different readers "
    "will weigh those considerations differently, and there is no single framing "
    "that suits every reader equally well. Taking the points in turn, each follows "
    "from what has already been said, and none of them introduces anything that "
    "was not already covered. In general terms, this is the shape of the answer."
)

SAMPLED_TEMPERATURE = 0.7


def pad(answer, target_chars):
    """Grow `answer` past `target_chars` with filler, without touching its content."""
    out = [answer]
    size = len(answer)
    while size < target_chars:
        out.append(FILLER)
        size += len(FILLER) + 2
    return "\n\n".join(out)


def subsample(items, n, seed):
    if n is None or n >= len(items):
        return items
    per_stratum = items.groupby(["category", "turn"], group_keys=False)
    frac = n / len(items)
    picked = per_stratum.apply(
        lambda g: g.sample(max(1, round(len(g) * frac)), random_state=seed)
    )
    return picked.sort_values("item_id").reset_index(drop=True)


def build_prompt(item, order, template, padded=False):
    """Render one judge prompt. `order` is the presentation order: 'ab' shows the
    alphabetically first model first, 'ba' shows it second."""
    answer_a, answer_b = item.answer_a, item.answer_b
    if padded:
        if len(answer_a) <= len(answer_b):
            answer_a = pad(answer_a, round(len(answer_b) * 1.5))
        else:
            answer_b = pad(answer_b, round(len(answer_a) * 1.5))
    first, second = (answer_a, answer_b) if order == "ab" else (answer_b, answer_a)
    prior = json.loads(item.prior_turns)
    if order == "ba":
        prior = [[user, b, a] for user, a, b in prior]
    return prompts.render(template, item.question, first, second, prior)


def jobs_for(item, templates, n_samples, do_padding):
    """Every (template, order, sample, condition, prompt) this item needs."""
    for template in templates:
        for order in ("ab", "ba"):
            yield template, order, 0, "base", build_prompt(item, order, template)
    for sample in range(1, n_samples):
        for order in ("ab", "ba"):
            yield prompts.DEFAULT_TEMPLATE, order, sample, "base", build_prompt(
                item, order, prompts.DEFAULT_TEMPLATE
            )
    if do_padding:
        for order in ("ab", "ba"):
            yield prompts.DEFAULT_TEMPLATE, order, 0, "padded", build_prompt(
                item, order, prompts.DEFAULT_TEMPLATE, padded=True
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-id", default="qwen2.5-7b-4bit-local")
    ap.add_argument("--backend", default="mlx", choices=sorted(judges.BACKENDS))
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--n", type=int, default=400, help="comparisons to judge, None for all")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--templates", default="bare,rubric,cot")
    ap.add_argument("--samples", type=int, default=3, help="samples for the majority vote mitigation")
    ap.add_argument("--no-padding", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many model calls")
    args = ap.parse_args()

    items = pd.read_parquet(os.path.join(HERE, "data", "items.parquet"))
    items = subsample(items, args.n, args.seed)
    templates = args.templates.split(",")

    sampled_ids = sorted(items.item_id)
    with open(os.path.join(HERE, "data", f"sample_{args.judge_id}.json"), "w") as fh:
        json.dump({"seed": args.seed, "n": len(sampled_ids), "item_ids": sampled_ids}, fh, indent=1)

    work = [
        (item, spec)
        for item in items.itertuples()
        for spec in jobs_for(item, templates, args.samples, not args.no_padding)
    ]
    print(f"{len(items)} comparisons, {len(work)} judge calls planned", flush=True)

    backend = None
    done = calls = 0
    started = time.perf_counter()
    for item, (template, order, sample, condition, prompt) in work:
        key = cache.cache_key(args.judge_id, template, prompt, sample)
        if cache.get(args.judge_id, key) is None:
            if backend is None:
                print(f"loading {args.model} on backend {args.backend}", flush=True)
                backend = judges.BACKENDS[args.backend](args.model)
            if args.limit is not None and calls >= args.limit:
                break
            calls += 1
        judges.run(
            backend,
            args.judge_id,
            template,
            prompt,
            sample=sample,
            temperature=SAMPLED_TEMPERATURE if sample > 0 else 0.0,
            extra={"item_id": item.item_id, "order": order, "condition": condition},
        )
        done += 1
        if done % 50 == 0:
            rate = calls / max(time.perf_counter() - started, 1e-9)
            print(
                f"{done}/{len(work)} done, {calls} model calls, {rate:.2f} calls/s",
                flush=True,
            )

    print(f"finished: {done} judgments, {calls} model calls, {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
