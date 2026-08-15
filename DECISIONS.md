# Analysis decisions

Written as the choices were made, with what each one costs.

## The ground truth is the MT-Bench human judgments, not a proxy

`lmsys/mt_bench_human_judgments`, CC BY 4.0. 3355 individual votes from 65
annotators over 1814 distinct comparisons. It is the only public preference set
of this size where the individual annotator votes survive, and without the
individual votes there is no human to human ceiling and therefore no way to say
whether a judge is good.

## The judge outputs are real, not simulated, and cost nothing to reproduce

LMSYS released the actual GPT-4 judgments alongside the benchmark, in both
presentation orders. That is what makes a no API key repo possible without
faking anything. The alternative, running a frontier judge here, would have cost
money and produced numbers nobody else could check.

## Pinned to the June 2023 revision of the mt-bench Space

The current revision of `lmsys/mt-bench` has judgments over a later, larger model
set that no longer includes `vicuna-13b-v1.2`. Against the human labels it covers
532 of 1814 comparisons. Revision `85425b615f50` covers 1814 of 1814. The audit
step checks this coverage and complains if it ever drops.

## Comparisons are identified without presentation order, and `a` is alphabetical

`a` means the alphabetically first model, never "the one shown first". Order is
introduced in exactly one place, `build_prompt` in `04_run_judges.py`, and is
mapped back in exactly one place, `canonical` in `05_measure.py`. If order lived
in the item identity, an order handling bug would be invisible: the flip rate
would come out at zero and look like a clean judge. `tests.py` fails if the two
orders are not genuine swaps of each other, including in earlier conversation
turns.

## Two scoring setups, and the primary one excludes ties

- **no-ties**, the headline: comparisons where the human majority named a winner
  and the judge named a winner. This is the setup the MT-Bench paper headlines,
  and it is the only one where a judge, a human and a coin flip are directly
  comparable, because the length and random baselines cannot say tie.
- **with-ties**: all comparisons, tie as a third label, exact match.

Excluding ties flatters any judge that abstains often, so the number of
comparisons each configuration is scored on is printed next to every accuracy in
the README, and the coverage is reported for every mitigation. GPT-4 single
answer grading is the case where this matters most: it scores highest and covers
least.

## Two human ceilings, because the obvious one is unfair to humans

A judge is scored against a majority of several annotators, which is a denoised
label. An individual annotator is not. So the README reports both:

- annotator against annotator, the raw pairwise agreement between two people;
- annotator against the majority of the other annotators, on comparisons with
  three or more votes, which is the like for like comparison to a judge.

The second is the higher bar and the one the headline uses. Reporting only the
first is how a judge ends up looking better than humans.

## The bootstrap resamples comparisons, never judgments

The two presentation orders of one comparison, and the several annotator votes on
it, are one draw from the world. Resampling judgments would treat them as
independent and shrink every interval. 10000 resamples, percentile intervals,
seed 20260815.

## The majority label breaks vote ties as "tie"

If the annotators split evenly between `a` and `b`, the label is `tie`, not a
coin flip. Those comparisons then drop out of the no-ties setup, which is the
honest treatment: nobody knows the answer.

## The local judge runs on a seeded subsample, and only complete comparisons count

A 3B judge under three templates, both orders, three samples and a padding
condition is 12 calls per comparison. 300 comparisons, stratified by category and
turn, is what fits in a few hours on a laptop. The measurement code drops any
comparison the judge has not finished every condition on, so a partial run cannot
bias the template comparison toward whichever template happened to run first.

## Cost is estimated from characters, and says so

The released judgment files carry no token usage, so input and output tokens are
estimated at 4 characters per token and every cached record carries
`token_counts_estimated: true`. Prices are gpt-4-0613 list prices, 30 and 60 USD
per million tokens. The local judge is free at the point of use, so it is
reported by wall clock instead.

## The 800 duplicated GPT-4 rows are kept as a rerun measurement

The released pair file judges 800 comparisons twice, weeks apart, under an
identical prompt. Sample 0 is the older run and is the one used everywhere else.
The pair is used once, to measure how often GPT-4 disagrees with itself. That
number turned out to be the most useful denominator in the repo: it is what the
position bias flip rate has to be compared against.

## Self preference is reported as a difference against humans, with the n stated

"GPT-4 prefers GPT-4" is easy to overclaim. The number reported is the gap
between the judge's win rate for the family and the human win rate for the same
family on the same comparisons, with a paired bootstrap interval, plus the same
measurement for a model family the judge has no stake in as a control.
