# Limitations

What these numbers do not support, listed so nobody has to discover it the hard
way.

## The domain is narrow

MT-Bench is 80 questions across writing, roleplay, reasoning, math, coding,
extraction, STEM and humanities, ten each. Every number here is an average over
that mix. A judge that is reliable on writing and useless on math averages out to
respectable, and this repo does not break agreement down by category with enough
comparisons per cell to say much. It is also English only, single answer per
turn, and at most two turns deep. Nothing here transfers to long documents,
agentic traces, tool use, or safety judgements without being measured again.

## The response set is from 2023

The six models compared are `alpaca-13b`, `llama-13b`, `vicuna-13b-v1.2`,
`gpt-3.5-turbo`, `claude-v1` and `gpt-4`. They are far apart in quality, which
makes the comparisons easier than the ones a team actually cares about today,
where two candidate models differ by a little. Judge agreement measured on easy
pairs is an upper bound on judge agreement where it matters. The leaderboard
ranking being stable across judges here should be read in that light: separating
`llama-13b` from `gpt-4` is not a demanding test.

## Human labels are noisy, and that is the point but also a limit

Two annotators agree with each other on about four comparisons in five. Some of
that is genuine ambiguity and some is annotation error, and this data cannot tell
them apart. 853 of 1814 comparisons have a single annotator, so their majority
label is one person's opinion. Those comparisons still count toward judge
agreement, which adds noise in a direction that penalises every judge equally.
The ceiling is computed only on comparisons with two or more annotators, and the
leave one out ceiling only on those with three or more, which is 752 annotator
comparisons rather than 1814.

## Self preference is measured on one judge and one family

The gap for GPT-4 judging GPT-4 responses is measured on 563 comparisons, and
the control on claude-v1 responses on 586. That is enough to see the effect but
not enough to characterise it, and it comes from one judge model, one response
set and one point in time. It also cannot separate self preference from a real
quality advantage that the human annotators underweighted, or from stylistic
kinship: GPT-4 may favour answers that look like its own without any awareness of
whose they are. The local Qwen judge has no responses from its own family in this
response set, so its self preference is not measurable here at all.

## The judgments are cached, not live

The GPT-4 judgments are from June 2023 and the model behind that name has been
retired. They are reproducible forever, which is the trade being made, but they
describe a judge nobody can call any more. Every conclusion is about that
snapshot. Rerunning the harness against a current model is one command and no
code change, and it is the right thing to do before quoting these numbers about a
judge you are actually shipping.

## The local judge is measured on a subsample

300 comparisons, seeded and stratified by category and turn, out of 1814. Its
intervals are correspondingly wide and its per model win rates rest on roughly a
hundred comparisons each. The GPT-4 numbers on the same subsample are reported
next to it, so the comparison between the two judges is like for like even though
each judge's own interval is not.

## Pairwise preference is not the only judging mode, and it may not be yours

Everything here judges "which of these two is better". Teams also use single
answer grading on a rubric, absolute scoring against a reference, pass or fail
criteria, and rankings over more than two candidates. One of those, GPT-4 single
answer grading, is included and behaves differently enough to make the point: it
has no position bias at all, it scores highest of any configuration, and it
achieves that partly by declining to separate the two answers on a third of
comparisons. Its numbers are not comparable to the pairwise ones without reading
the coverage next to them.

## Ties are handled one way, and the choice matters

The headline setup drops comparisons where either side said tie. It is the
standard choice and it is what makes judges, humans and coin flips comparable,
but it rewards abstention. The with-ties numbers are in `output/metrics.json` and
in the README table for exactly this reason. Neither setup is the true one.

## The padding test shows one direction

Padding the shorter answer with content free filler tests whether length alone
moves a verdict. A judge that moves toward the padded answer is exhibiting
verbosity bias, which is unambiguous. A judge that does not may be robust, or may
simply be detecting the filler as filler, which is a different and easier task
than resisting genuinely well written length. The observational verbosity number
next to it, how often each judge picks the longer answer, has the opposite
problem: longer answers really are better on this dataset, so that correlation is
not evidence of bias on its own. Read the two together.

## Cost numbers are estimates

Token counts for the released judgments are estimated from character counts, at
list prices for a model that is no longer sold. Treat the cost axis as an order
of magnitude, not a quote.
