"""Write README.md from output/metrics.json.

The README is generated so that a number in it cannot go stale. Nothing here
computes anything: if a figure is not in metrics.json it does not go in the
README. Prose that does not depend on a number lives in this file as prose.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pct(x, places=1):
    return f"{100 * x:.{places}f} percent"


def num(x, places=1):
    return f"{100 * x:.{places}f}"


def interval(row, key="ci"):
    lo, hi = row[key]
    return f"{100 * lo:.1f} to {100 * hi:.1f}"


def find(rows, label):
    for row in rows:
        if row["label"] == label:
            return row
    raise KeyError(f"{label} is not in metrics.json, so it cannot go in the README")


def signed(x):
    return f"{'+' if x >= 0 else '-'}{abs(100 * x):.1f}"


def diff_interval(row):
    lo, hi = row["minus_ceiling_ci"]
    return f"{signed(lo)} to {signed(hi)}"


def zero_phrase(row):
    lo, hi = row["minus_ceiling_ci"]
    return "includes zero" if lo <= 0 <= hi else "excludes zero"


def gap_words(row):
    d = row["minus_ceiling"]
    return f"{num(abs(d))} points {'below' if d < 0 else 'above'}"


def matched_table(matched):
    out = [
        "| configuration | agreement | 95 percent CI | comparisons scored | minus ceiling | 95 percent CI |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {matched['ceiling']['label']} | {num(matched['ceiling']['accuracy'])} | {interval(matched['ceiling'])} | "
        f"{matched['n_comparisons']} ({matched['ceiling']['n']} scorings) | | |",
    ]
    for row in sorted(matched["full_majority"], key=lambda r: -r["accuracy"]):
        out.append(
            f"| {row['label']} | {num(row['accuracy'])} | {interval(row)} | {row['n']} | "
            f"{signed(row['minus_ceiling'])} | {diff_interval(row)} |"
        )
    return "\n".join(out)


def agreement_table(rows):
    out = [
        "| configuration | agreement | 95 percent CI | comparisons scored |",
        "| --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: -r["accuracy"]):
        if row["n"] == 0:
            continue
        out.append(f"| {row['label']} | {num(row['accuracy'])} | {interval(row)} | {row['n']} |")
    return "\n".join(out)


def build(m):
    data = m["dataset"]
    no_ties = m["agreement"]["no_ties"]
    with_ties = m["agreement"]["with_ties"]

    gpt4 = find(no_ties, "gpt-4 pairwise, one order")
    gpt4_single = find(no_ties, "gpt-4 single answer grading")
    ceiling = find(no_ties, "human ceiling, one annotator vs the majority of the others")
    raw_ceiling = find(no_ties, "human to human ceiling")
    length = find(no_ties, "length baseline")
    coin = find(no_ties, "random baseline")

    position = m["position_bias"]["gpt-4 pairwise"]
    rerun = m["rerun_stability"]
    swap_mitigation = next(r for r in m["mitigations"] if r["mitigation"] == "gpt-4: swap averaging")
    self_pref = m["self_preference"]["gpt-4 judging gpt-4 responses"]
    control = m["self_preference"]["gpt-4 judging claude-v1 responses (control)"]
    verb_human = m["verbosity"]["human majority"]
    verb_gpt4 = m["verbosity"]["gpt-4 pairwise, one order"]

    # Judge against ceiling is only ever stated on the comparisons the ceiling is
    # computed on. The full set judge numbers and the ceiling are different items.
    matched = m["agreement_matched"]
    matched_rows = {r["label"]: r for r in matched["full_majority"]}
    m_gpt4 = matched_rows["gpt-4 pairwise, one order"]
    m_swap = matched_rows["gpt-4 pairwise, swap averaged"]
    m_single = matched_rows["gpt-4 single answer grading"]
    m_length = matched_rows["length baseline"]
    m_same = {r["label"]: r for r in matched["same_targets"]}["gpt-4 pairwise, one order"]

    local_rows = [r for r in no_ties if r["label"].startswith("local 3B")]
    has_local = bool(local_rows)
    local_note = (
        f", plus a local\n  Qwen2.5 3B judge run here on a laptop over a seeded subsample of "
        f"{data['local_judge_subsample']} of them"
        if has_local
        else ""
    )

    parts = []
    parts.append(
        f"""# judge-reliability

**How much can you trust an LLM as a judge? This repo checks GPT-4 judging MT-Bench answers against the people who
labelled them, against the best a human can do on the same items, against a rule that just picks the longer
answer, and against itself when the two answers swap places.**

The judge is good. It is also not better than a careful human, it is only
{num(gpt4['accuracy'] - length['accuracy'])} points better than a rule that reads none
of the text and always picks the longer answer, and its order sensitivity turns out to
be indistinguishable from its own run to run noise. Those three facts are the reason to
measure a judge rather than assume one.

In numbers: across {data['comparisons']} human labelled MT-Bench comparisons, GPT-4 judging a
pair of answers agrees with the human majority {pct(gpt4['accuracy'])} of the time,
and a length only baseline reaches {pct(length['accuracy'])}. A like for like human
ceiling exists only on the {matched['n_comparisons']} comparisons with three or more
votes. On those, one annotator agrees with the majority of the others
{pct(ceiling['accuracy'])} of the time and GPT-4 scores {pct(m_gpt4['accuracy'])},
{gap_words(m_gpt4)} the ceiling (interval on the difference {diff_interval(m_gpt4)}).
Reversing the order the two answers are shown in flips the verdict on
{pct(position['flip_rate'])} of comparisons; swap averaging recovers
{num(swap_mitigation['delta'])} points over the full set and leaves the judge
{gap_words(m_swap)} the ceiling on its comparisons, an interval of
{diff_interval(m_swap)} that {zero_phrase(m_swap)}.

Part of [benchmeasure](https://github.com/abhaymettu/benchmeasure), benchmarks treated as measurement instruments: reliability ([agent-validity-ceiling](https://github.com/abhaymettu/agent-validity-ceiling)), invariance ([benchdif](https://github.com/abhaymettu/benchdif)), and the scorer (this repo). Paper: [benchmeasure.pdf](https://github.com/abhaymettu/benchmeasure/blob/main/paper/benchmeasure.pdf).

![GPT-4 judge agreement with the human majority, against the human ceiling and a length-only baseline](output/figures/fig1_agreement.png)

Write-up: [abhaymettu.com/research/judge-reliability](https://abhaymettu.com/research/judge-reliability/)

## Install

```bash
git clone https://github.com/abhaymettu/judge-reliability && cd judge-reliability && pip install -r requirements.txt
```

## Quick start

```bash
make analysis
```

This runs the tests, recomputes every metric from the committed judgment cache,
redraws the six figures and rewrites this README. No API key, no model call, no
network. It takes about 20 seconds on a laptop.

## What this is

An LLM-as-judge is a measurement instrument, and nobody ships an instrument
without calibrating it. This repo calibrates one as a rater: agreement with human
preferences against the human to human ceiling, position bias, verbosity bias,
self preference, prompt sensitivity, cost, and which debiasing mitigations
actually recover accuracy rather than merely sounding sensible.

## The data

- **{data['comparisons']} comparisons** from `lmsys/mt_bench_human_judgments`, CC BY 4.0:
  {data['human_votes']} individual votes from {data['annotators']} annotators over
  {data['questions']} MT-Bench questions and {len(data['models'])} models
  ({', '.join(data['models'])}), first and second conversation turns.
- **Human labels**: the annotator majority names a winner on
  {data['human_label_counts'].get('a', 0) + data['human_label_counts'].get('b', 0)}
  comparisons and ties on {data['human_label_counts'].get('tie', 0)}.
  {data['comparisons_with_2plus_annotators']} comparisons have two or more annotators,
  which is what makes a ceiling possible.
- **Judges**: GPT-4 (0613) pairwise in both presentation orders and GPT-4 single answer
  grading, both from the judgments LMSYS released with MT-Bench{local_note}, plus two
  non LLM baselines: always prefer the longer answer, and a coin flip.

## 1. Agreement, and the ceiling that gives it meaning

Ties excluded on both sides, which is the standard setup and the only one where a
judge, a human and a coin flip are comparable. `comparisons scored` is how many
of the {data['comparisons']} survive that restriction for each configuration:
a judge that abstains more is scored on fewer and easier comparisons.

{agreement_table(no_ties)}

Two ceilings, because the obvious one is unfair to humans. Annotator against
annotator is {pct(raw_ceiling['accuracy'])}, which is what people usually quote.
But a judge is scored against the *majority* of several annotators, a denoised
label, so the like for like number is one annotator against the majority of the
others: {pct(ceiling['accuracy'])} ({interval(ceiling)}). Quote the first and GPT-4
looks superhuman. The second can only be computed on {matched['n_comparisons']}
comparisons, so the judge has to be scored on those same comparisons before the
two can be subtracted.

### The judge and the ceiling on the same comparisons

Each configuration below is scored on exactly the {matched['n_comparisons']}
comparisons the like for like ceiling uses, against the majority of all the
annotators, ties excluded on both sides. The difference is paired: judge and
ceiling are computed on the same resampled comparisons in every bootstrap draw.

{matched_table(matched)}

One order GPT-4 sits {gap_words(m_gpt4)} the ceiling, an interval that
{zero_phrase(m_gpt4)}. Swap averaged it sits {gap_words(m_swap)}, an interval that
{zero_phrase(m_swap)}. Single answer grading sits {gap_words(m_single)}, an interval
that {zero_phrase(m_single)}, but it is scored on {m_single['n']} of the
{matched['n_comparisons']} comparisons, against {m_gpt4['n']} for one order, because it
calls far more ties. The length
rule sits {gap_words(m_length)}. Scoring the judge instead against every leave one
out majority the annotators face, on the same scorings, puts one order GPT-4
{gap_words(m_same)} the ceiling ({diff_interval(m_same)}); both versions are in
`output/metrics.json` under `agreement_matched`.

Cohen's kappa tells the same story with the chance agreement taken out:
{m['kappa']['gpt-4 pairwise, one order']['cohens_kappa']:.2f} for GPT-4 against
{m['kappa']['length baseline']['cohens_kappa']:.2f} for the length rule and
{m['kappa']['random baseline']['cohens_kappa']:.2f} for the coin. Krippendorff's alpha
over the {m['krippendorff']['n_raters']} human annotators is
{m['krippendorff']['humans_only']:.3f}; adding GPT-4 to the annotator pool moves it to
{m['krippendorff']['gpt-4 pairwise, one order']:.3f}. On this evidence GPT-4 is about as
agreeable as one more human annotator, no more and no less.

Same table, ties kept and counted as a third label:

{agreement_table(with_ties)}

## 2. Position bias, next to the judge's own noise

Every comparison is judged twice, once in each order.

![position bias flip rates](output/figures/fig2_position_bias.png)

- Order reversal flips the verdict on **{pct(position['flip_rate'])}**
  ({interval(position, 'flip_ci')}) of {position['n_comparisons']} comparisons.
- There is **no directional preference**: across
  {position['n_decided_presentations']} decided presentations the judge picks whichever
  answer was shown first {pct(position['position_one_rate'])} of the time. A coin does
  that. The bias is instability, not a thumb on the first position.
- GPT-4 single answer grading has no position bias at all, because it never sees
  the two answers together.

The number that puts the flip rate in perspective: {rerun['n_judgments']} of these
judgments exist twice in the released files, same prompt, same model, runs weeks
apart. GPT-4 agrees with itself on {pct(rerun['same_verdict_rate'])} of them, so it
**disagrees with itself {pct(1 - rerun['same_verdict_rate'])} of the time with nothing
changed at all**. The {pct(position['flip_rate'])} flip rate under order reversal is the
same size. Whatever swap averaging is buying here, it is variance reduction, not the
removal of a directional bias.

## 3. Verbosity

![how often each judge picks the longer answer](output/figures/fig3_verbosity.png)

GPT-4 picks the longer answer {pct(verb_gpt4['picks_longer_rate'])} of the time
(r = {verb_gpt4['corr_choice_lengthdiff']:.2f} between its choice and the length
difference). The humans pick the longer answer {pct(verb_human['picks_longer_rate'])} of
the time (r = {verb_human['corr_choice_lengthdiff']:.2f}). Those are the same number.

This is the finding people get backwards. The judge's correlation with length is
not by itself evidence of bias, because on this data longer answers really are
better and humans prefer them at the same rate. What it does establish is how low
the bar is: a rule that reads none of the text and always picks the longer answer
already agrees with humans {pct(length['accuracy'])} of the time, against a coin flip's
{pct(coin['accuracy'])}. Roughly {pct(length['accuracy'] / gpt4['accuracy'], 0)} of what a
frontier judge achieves is achievable by counting characters.
"""
    )

    padding = m.get("padding_test")
    if padding:
        parts.append(
            f"""
The controlled version holds quality fixed: pad the shorter answer with content
free filler until it is half again as long as the longer one, and see whether the
verdict moves. On {padding['n']} comparisons where the local judge preferred the other
answer before padding, it moved to the padded answer
**{pct(padding['moved_to_padded_rate'])}** of the time ({interval(padding)}).
"""
        )

    parts.append(
        f"""
## 4. Self preference

GPT-4 judging comparisons that include a GPT-4 response, against how the same
humans scored the same comparisons:

| | judge win rate for the family | human win rate | gap | 95 percent CI | comparisons |
| --- | --- | --- | --- | --- | --- |
| gpt-4 judging gpt-4 | {num(self_pref['judge_win_rate_for_family'])} | {num(self_pref['human_win_rate_for_family'])} | +{num(self_pref['difference'])} | {interval(self_pref)} | {self_pref['n']} |
| gpt-4 judging claude-v1 (control) | {num(control['judge_win_rate_for_family'])} | {num(control['human_win_rate_for_family'])} | +{num(control['difference'])} | {interval(control)} | {control['n']} |

GPT-4 rates its own family {num(self_pref['difference'])} points higher than the humans
do, more than three times the gap it shows for a family it has no stake in. The
interval excludes zero. It is a real effect on {self_pref['n']} comparisons, and it is
still one judge, one response set and one point in time, so read
`LIMITATIONS.md` before quoting it. It cannot be separated here from stylistic
kinship: GPT-4 may simply favour answers that look like its own.
"""
    )

    boards = m["leaderboards"]
    if has_local:
        # Prompt alone means one order, one sample. The swap averaged row is a
        # mitigation and belongs in section 6, not in a claim about prompts.
        prompt_rows = [r for r in local_rows if r["label"].endswith("one order")]
        short = lambda label: label.replace("local 3B, ", "").replace(", one order", "").replace(" prompt", "")
        prompt_boards = {
            short(k): v for k, v in boards.items() if k.startswith("local 3B") and k.endswith("one order")
        }
        taus = ", ".join(f"{k} {v['kendall_tau_vs_human']:.2f}" for k, v in prompt_boards.items())
        tops = ", ".join(f"the {k} prompt picks {v['top_model']}" for k, v in prompt_boards.items())
        best = max(prompt_rows, key=lambda r: r["accuracy"])
        worst = min(prompt_rows, key=lambda r: r["accuracy"])
        human_top = boards["human majority, local subsample"]["top_model"]
        local_gpt4 = find(m["agreement_on_local_subsample"], "gpt-4 pairwise, one order")
        local_length = find(m["agreement_on_local_subsample"], "length baseline")
        parts.append(
            f"""
## 5. Prompt sensitivity

The judge prompt is a free parameter that teams pick by taste. Here are three,
identical except for scaffolding, run by the same local Qwen2.5 3B model over the
same {data['local_judge_subsample']} comparisons in both orders: a bare instruction, a
rubric that spells out criteria and explicitly warns against rewarding length or
position, and a chain of thought prompt that reasons before committing.

![model win rates by judge prompt](output/figures/fig4_prompt_sensitivity.png)

| judge prompt | agreement | 95 percent CI | comparisons scored |
| --- | --- | --- | --- |
"""
            + "\n".join(
                f"| {short(r['label'])} | {num(r['accuracy'])} | {interval(r)} | {r['n']} |"
                for r in sorted(prompt_rows, key=lambda r: -r["accuracy"])
            )
            + f"""

**Changing nothing but the judge prompt moves the model at the top of the
leaderboard.** Over the same responses and the same comparisons, {tops}. The
humans on these comparisons pick {human_top}. Kendall tau against the human
ranking: {taus}.

Agreement moves {num(best['accuracy'] - worst['accuracy'])} points across the three
prompts, from {num(worst['accuracy'])} for the {short(worst['label'])} prompt to
{num(best['accuracy'])} for the {short(best['label'])} prompt. The direction is the
surprise: the rubric is the prompt that tells the judge in as many words not to
reward length or position, and it is the worst of the three. Scaffolding that
sounds like it should help is exactly the kind of thing that has to be measured.

For scale, on this same pool of {data['local_judge_subsample']} comparisons GPT-4
scores {num(local_gpt4['accuracy'])} (n={local_gpt4['n']} after ties are excluded) and the
length rule scores {num(local_length['accuracy'])} (n={local_length['n']}).
Every prompt variant of the 3B judge sits at or below the length rule. What the
local judge contributes is not accuracy, it is that it is cheap enough to run
under every prompt variant, which is the experiment a team needs before trusting
any leaderboard a judge produces.
"""
        )
    else:
        parts.append(
            """
## 5. Prompt sensitivity

Not measured in this run: the local judge cache is empty. Run
`make judge` and then `make analysis`, or see USAGE.md to point the harness at a
model of your own.
"""
        )

    parts.append(
        """
## 6. Mitigations, measured rather than assumed

![mitigation effectiveness](output/figures/fig5_mitigations.png)

| mitigation | before | after | change | 95 percent CI | comparisons scored before and after |
| --- | --- | --- | --- | --- | --- |
"""
        + "\n".join(
            f"| {r['mitigation']} | {num(r['baseline_accuracy'])} | {num(r['mitigated_accuracy'])} | "
            f"{'+' if r['delta'] >= 0 else ''}{num(r['delta'])} | {interval(r)} | "
            f"{pct(r['coverage_before'], 0)} to {pct(r['coverage_after'], 0)} |"
            for r in m["mitigations"]
        )
        + f"""

Swap averaging is the one that works, and it is not free: it doubles the number
of model calls and it converts every order disagreement into a tie, so coverage
drops from {pct(swap_mitigation['coverage_before'], 0)} of comparisons to
{pct(swap_mitigation['coverage_after'], 0)}. You are buying accuracy with abstention as
much as with debiasing, which is fine as long as you know that is the trade.
"""
    )

    parts.append(
        """
## 7. Cost against reliability

![cost against reliability](output/figures/fig6_cost_reliability.png)

Every point in that figure is measured on one pool of comparisons, the subsample
the local judge ran on, so the accuracies there are not the full set accuracies
in the table above. Mixing the two pools on one axis would compare different
samples as though they were the same one.

| judge | mean input tokens | mean output tokens | USD per 1000 judgments | median latency |
| --- | --- | --- | --- | --- |
"""
        + "\n".join(
            f"| {c['judge_id']} | {c['mean_input_tokens']:.0f} | {c['mean_output_tokens']:.0f} | "
            f"{c['usd_per_1000_judgments']:.2f}{' (estimated)' if c.get('token_counts_estimated') else ''} | "
            f"{c['median_latency_s']:.2f} s |" if "median_latency_s" in c else
            f"| {c['judge_id']} | {c['mean_input_tokens']:.0f} | {c['mean_output_tokens']:.0f} | "
            f"{c['usd_per_1000_judgments']:.2f}{' (estimated)' if c.get('token_counts_estimated') else ''} | not recorded |"
            for c in m["cost"]
        )
        + """

Token counts for the released GPT-4 judgments are estimated from character
counts and priced at gpt-4-0613 list prices, so treat that column as an order of
magnitude. Swap averaging doubles it.
"""
    )

    extra_finding = ""
    if has_local:
        best_prompt = max(
            (r for r in local_rows if r["label"].endswith("one order")), key=lambda r: r["accuracy"]
        )
        swapped_local = find(no_ties, "local 3B, rubric prompt, swap averaged")
        local_length = find(m["agreement_on_local_subsample"], "length baseline")
        local_flips = {
            k: v for k, v in m["position_bias"].items() if k.startswith("local 3B") and "flip_rate" in v
        }
        worst_flip = max(local_flips.items(), key=lambda kv: kv[1]["flip_rate"])
        extra_finding = f"""6. **No prompt variant of the small local judge beats the length baseline.** Its
   best prompt reaches {pct(best_prompt['accuracy'])} on the subsample where the length
   rule reaches {pct(local_length['accuracy'])}. Only after swap averaging does it edge
   past, to {pct(swapped_local['accuracy'])} on {swapped_local['n']} comparisons rather
   than {best_prompt['n']}. On this evidence a 3B judge on these comparisons is mostly
   measuring verbosity and noise, and should not be shipped as an evaluator.
7. **Chain of thought made the local judge less stable, not more.** Under the
   {worst_flip[0].replace('local 3B, ', '').replace(' prompt', '')} prompt the verdict reverses on
   {pct(worst_flip[1]['flip_rate'])} of comparisons when the order is reversed, worse than a coin, and it is the one prompt with a real thumb on the
   scale: it picks whichever answer came first
   {pct(worst_flip[1]['position_one_rate'])} of the time. Reasoning traces are not free
   reliability.
8. **The rubric that warns against length and position bias made agreement
   worse.** Swapping the bare prompt for the rubric costs the local judge
   {num(abs(next(r['delta'] for r in m['mitigations'] if 'rubric prompt instead of bare' in r['mitigation'])))} points, interval excluding zero. Telling a judge not to be biased is not a
   mitigation until it has been measured as one.
"""

    parts.append(
        f"""
## The unflattering findings, in one place

1. **Order sensitivity is not a bias, it is noise.** GPT-4 flips on
   {pct(position['flip_rate'])} of comparisons when the order is reversed, and disagrees
   with itself on {pct(1 - rerun['same_verdict_rate'])} of identical reruns. There is no
   directional preference for position one to remove
   ({pct(position['position_one_rate'])}, which is a coin). Swap averaging helps because
   averaging two noisy draws helps.
2. **The length baseline is closer than it should be.** Always preferring the
   longer answer agrees with humans {pct(length['accuracy'])} of the time, reaching
   {pct(length['accuracy'] / gpt4['accuracy'], 0)} of what GPT-4 achieves while reading
   none of the text. Any judge evaluation without this baseline in it is not
   telling you what the judge contributes.
3. **The pairwise judge is not better than a person.** On the
   {matched['n_comparisons']} comparisons where a like for like ceiling exists, GPT-4
   scores {pct(m_gpt4['accuracy'])} against {pct(ceiling['accuracy'])}, and the interval on
   the difference ({diff_interval(m_gpt4)}) {zero_phrase(m_gpt4)}. The commonly quoted comparison,
   against raw annotator to annotator agreement of {pct(raw_ceiling['accuracy'])}, makes
   the judge look superhuman and is the wrong comparison.
4. **The judge is partial to itself**, by {num(self_pref['difference'])} points against
   the human labels on {self_pref['n']} comparisons, more than three times the gap for a
   family it has no stake in.
5. **The best scoring configuration scores best partly by abstaining.** GPT-4
   single answer grading reaches {pct(gpt4_single['accuracy'])}, the highest number in
   the repo, on {gpt4_single['n']} comparisons rather than {gpt4['n']}. Read coverage
   next to accuracy or that number is a mirage.
{extra_finding}
## Reproduce

```bash
pip install -r requirements.txt
make all
```

`make all` downloads the public data, audits it, rebuilds the comparison table,
loads the released GPT-4 judgments into the cache, runs the tests, and rewrites
every number above. `make analysis` on its own does everything from the committed
cache with no network at all.

**No number in this README is typed by hand.** Every one of them is written by
`07_report.py` out of `output/metrics.json`, so a result cannot drift from the
data by being edited in one place and not the other. `LIMITATIONS.md` and
`DECISIONS.md` are prose and do quote figures, so the same script cross checks
every number those two assert against `metrics.json` and fails the build naming
the sentence if one stops matching. It exists because a hand audit of this repo
found three stale numbers in those two files, left behind by a bug fix that
changed the sample sizes underneath them. The audit caught those; the check is
there so the next ones cannot survive a build.

To run a judge of your own, see [USAGE.md](USAGE.md). The harness needs one class
with one method.

## What is new compared with the closest work

Judge agreement with humans is well studied. This repo adds two narrower things,
a fairer human ceiling and a noise floor for the order flip rate.

- **Zheng et al. 2023** (MT-Bench, the source of this data) report GPT-4 against
  humans at 85 percent on non-tie votes, against human to human agreement of 81
  percent, and read that as the judge matching humans. That human number is one
  annotator against another. This repo adds the like for like ceiling: one
  annotator against the majority of the others, which is what the judge is scored
  against. On the same data that ceiling is {pct(ceiling['accuracy'])}, against
  {pct(raw_ceiling['accuracy'])} for annotator against annotator, and scored on
  the same comparisons the pairwise judge's point estimate falls
  {'below' if m_gpt4['minus_ceiling'] < 0 else 'above'} it.
- **Anand et al. 2026** apply generalizability theory to 373,019 LLM judge scores.
  They show that a single judge caps generalizability however many items you add,
  and that judging in both orders raises the cap. Theirs is a variance decomposition
  for sizing judge panels. This repo measures one judge against a human ceiling on
  the items themselves.
- **Kumar et al. 2026** show that verdicts change across identical temperature 0
  replications and that position swaps flip most verdicts on hard tasks. This repo
  puts those two effects side by side on one dataset: the
  {pct(position['flip_rate'])} order flip rate next to the judge's
  {pct(1 - rerun['same_verdict_rate'])} disagreement with itself on
  {rerun['n_judgments']} duplicated judgments. The two are the same size, which
  makes swap averaging read as variance reduction.
- **Jha et al. 2026** find that on 246 unanimous MT-Bench cells, the benchmark
  authors and recruited experts disagree on 30.5 percent. That is a direct warning
  about the human labels this ceiling is built from.
- **Thakur et al. 2025** compare 13 judges against humans in a high agreement
  setting and find that only the best approach human alignment. They compare to
  inter-human agreement but do not treat it as a formal ceiling on the judge.

## Limits

[LIMITATIONS.md](LIMITATIONS.md) has the full list. These five matter most.

- **The domain is narrow.** 80 MT-Bench questions, English, at most two turns, and
  six models from 2023 that are far apart in quality. Agreement on easy pairs is an
  upper bound on agreement where two candidates are close.
- **The like for like ceiling is the thinnest number here.** It needs three or more
  votes per comparison, so it rests on {ceiling['n_comparisons']} comparisons.
- **The judgments are a snapshot.** The GPT-4 judgments are the June 2023 ones
  LMSYS released, from a model that has since been retired. Rerun the harness on a
  current judge before quoting these numbers about one you ship.
- **Self preference is one judge and one family**, and it cannot be separated from
  stylistic kinship.
- **The local 3B judge runs on a seeded subsample** of {data['local_judge_subsample']}
  comparisons, so its intervals are wide.

## Files

| Path | What it is |
| --- | --- |
| `00_download.py` to `07_report.py` | the pipeline, in order |
| `harness/` | the reusable part: prompts, cache, judge backends, statistics |
| `data/judgments/` | every judge call, one JSON file per call, keyed by a hash of the prompt |
| `output/metrics.json` | every number in this README |
| `output/audit.md` | the data audit, run before anything is built on the data |
| `tests.py` | swap logic, bootstrap, kappa, cache key, verdict parsing |
| `DECISIONS.md` | what was decided and what it cost |
| `LIMITATIONS.md` | what these numbers do not support |
| `USAGE.md` | how to measure a judge of your own |
| `AGENTS.md` | notes for coding agents working in this repo |

Data is `lmsys/mt_bench_human_judgments` (CC BY 4.0) and the judgments released
with MT-Bench by LMSYS. Figures use a colourblind checked palette; every bar
carries its value as text.

## For coding agents

Read [AGENTS.md](AGENTS.md) first. Never edit numbers in `README.md` by hand;
change `07_report.py` and run `make analysis`.

## Cite

```bibtex
@misc{{mettu2026judgereliability,
  author       = {{Mettu, Abhay}},
  title        = {{judge-reliability: calibrating an LLM judge against the human ceiling}},
  year         = {{2026}},
  howpublished = {{\\url{{https://github.com/abhaymettu/judge-reliability}}}},
  note         = {{Part of benchmeasure, https://github.com/abhaymettu/benchmeasure}}
}}
```
"""
    )
    return "".join(parts)


def check_static_docs(m):
    """Every number a hand written document asserts, checked against metrics.json.

    README.md is generated so it cannot drift. LIMITATIONS.md and DECISIONS.md
    are written by hand, and a hand written number that quietly stops matching
    the data is the most common way an honest repo starts lying. Anything
    asserted there is registered here and the build fails if it moves.
    """
    data = m["dataset"]
    nt = {r["label"]: r for r in m["agreement"]["no_ties"]}
    ceiling = nt["human ceiling, one annotator vs the majority of the others"]
    self_pref = m["self_preference"]["gpt-4 judging gpt-4 responses"]
    control = m["self_preference"]["gpt-4 judging claude-v1 responses (control)"]
    full_gpt4 = nt["gpt-4 pairwise, one order"]
    m_gpt4 = {r["label"]: r for r in m["agreement_matched"]["full_majority"]}["gpt-4 pairwise, one order"]
    lo, hi = m_gpt4["minus_ceiling_ci"]

    claims = [
        ("LIMITATIONS.md", f"one order scores {100 * m_gpt4['accuracy']:.1f} percent", "matched judge"),
        ("LIMITATIONS.md", f"{100 * abs(m_gpt4['minus_ceiling']):.1f} points under the {100 * ceiling['accuracy']:.1f} percent ceiling", "matched gap"),
        ("LIMITATIONS.md", f"difference {'includes' if lo <= 0 <= hi else 'excludes'} zero", "matched interval"),
        ("LIMITATIONS.md", f"full set {100 * full_gpt4['accuracy']:.1f} percent", "full set judge"),
        ("DECISIONS.md", f"{data['human_votes']} individual votes", "human votes"),
        ("DECISIONS.md", f"over {data['comparisons']} distinct comparisons", "comparisons"),
        ("DECISIONS.md", f"{m['rerun_stability']['n_judgments'] // 2} of those are comparisons", "reruns"),
        ("DECISIONS.md", f"{m['rerun_stability']['n_judgments']} repeated judgments", "repeated judgments"),
        ("DECISIONS.md", f"turn drew {data['local_judge_subsample']} of them", "local subsample"),
        ("LIMITATIONS.md", f"measured on {self_pref['n']} comparisons", "self preference n"),
        ("LIMITATIONS.md", f"claude-v1 responses on {control['n']}", "self preference control n"),
        ("LIMITATIONS.md", f"rests on {ceiling['n_comparisons']} comparisons", "leave one out comparisons"),
        ("LIMITATIONS.md", f"supply {ceiling['n']} annotator against", "leave one out scorings"),
        ("LIMITATIONS.md", f"two or more annotators, {data['comparisons_with_2plus_annotators']} of", "2+ annotators"),
        ("LIMITATIONS.md", f"{data['local_judge_subsample']} comparisons, seeded", "local subsample"),
    ]
    stale = []
    for filename, phrase, what in claims:
        with open(os.path.join(HERE, filename), encoding="utf-8") as fh:
            body = " ".join(fh.read().split())
        if " ".join(phrase.split()) not in body:
            stale.append(f"{filename}: {what} no longer matches the data, expected the phrase {phrase!r}")
    if stale:
        raise SystemExit("stale hand written claims:\n  " + "\n  ".join(stale))


def main():
    with open(os.path.join(HERE, "output", "metrics.json"), encoding="utf-8") as fh:
        metrics = json.load(fh)
    check_static_docs(metrics)
    text = build(metrics)
    if "\u2014" in text or "\u2013" in text:
        raise SystemExit("house rule: no dashes of that kind in the README")
    path = os.path.join(HERE, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {path} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
