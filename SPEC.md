# judge-reliability

**Build this repo. This file is the whole brief. Read it fully, then start.**

## House rules (apply to every step)

- No em dashes anywhere, in code comments, README, or commit messages.
- Real public data with real human labels for the headline result.
- The honest result is the deliverable. If the judge agrees with humans no
  better than a length heuristic, the README leads with that.
- Every number in the README reproducible by one command, from cached model
  outputs, with no API key required to rerun the analysis.
- Python, pinned deps, `make all` regenerates everything.
- No Claude/Co-Authored-By trailers in commits.

## One-line pitch

How much can you trust an LLM as a judge? Agreement with human preferences,
position bias, verbosity bias and self preference, measured with confidence
intervals on public human labeled data.

## Why this repo exists

Teams ship LLM-as-judge evaluation without ever measuring the judge. This repo
treats the judge as an instrument and measures its reliability and its biases,
the same way you would validate any measurement instrument.

## Data

- **MT-Bench human judgments:** `lmsys/mt_bench_human_judgments` on Hugging
  Face. Roughly 3.3k human pairwise preferences over model responses, with
  multiple human annotators per pair. This is the ground truth.
- **Optional second source:** a sample of Chatbot Arena conversations with
  human votes, if the licensing and size are workable.
- Cache every judge call to `data/judgments/` keyed by a hash of the prompt.
  The repo must reproduce all analysis offline with zero API calls.

## Judges to evaluate

At least three, ideally spanning tiers and vendors, so the finding is about
judging and not about one model. For example a frontier model, a mid tier
model, and a small local model via Ollama. Plus two non LLM baselines:

- **Length heuristic:** always prefer the longer response.
- **Random:** coin flip.

The length baseline is essential. If a judge cannot clearly beat "prefer the
longer answer" then the judge is measuring verbosity, and that is a finding.

## Measurements

1. **Human agreement.** Accuracy against the human majority label, with
   bootstrap confidence intervals. Also report Cohen's kappa and Krippendorff's
   alpha, and critically report **human to human agreement** as the ceiling.
   A judge at 82 percent when humans agree with each other 81 percent of the
   time is a different story than 82 percent against a 95 percent ceiling.
2. **Position bias.** Run every pair in both orders (A,B) and (B,A). Report the
   flip rate, the directional preference for position one, and the fraction of
   pairs where the judge is self inconsistent. Report agreement before and
   after order randomization and swap averaging.
3. **Verbosity bias.** Correlation between the judge's choice and the response
   length difference. Then a controlled test: hold quality constant by padding
   a response with neutral filler and measure how often the judge flips.
4. **Self preference.** Where the response set includes outputs from a judge's
   own model family, measure whether the judge favors its own family relative
   to human labels. State the sample sizes; this is easy to overclaim.
5. **Prompt sensitivity.** Three judge prompt templates (bare, with rubric,
   with chain of thought). Report how much the measured "winner" changes as a
   function of judge prompt alone. This is the most damning and most useful
   plot in the repo.
6. **Cost and latency** per judgment per judge, so the reliability numbers can
   be read against price.
7. **Mitigations, measured not assumed.** Implement swap averaging, rubric
   prompting, and majority vote over three samples, and report how much each
   actually recovers. Report the ones that do not help.

## Deliverables

- `README.md` whose first two sentences carry the verdict. Example shape:
  "Across N human labeled pairs, judge X agrees with human majority Y percent
  of the time against a human to human ceiling of Z percent. The length only
  baseline reaches L percent. Position order alone flips the judge on P
  percent of pairs, and swap averaging recovers R of that."
- Hero figure: judge accuracy vs the human ceiling and the length baseline,
  with CIs.
- Figures: position bias flip rates, verbosity correlation, prompt template
  sensitivity, mitigation effectiveness, cost vs reliability Pareto.
- A short `USAGE.md` on how to point the harness at a new judge model, because
  the reusable harness is half the value of this repo.
- `LIMITATIONS.md`: MT-Bench domain coverage, human label noise, self
  preference sample size, the fact that pairwise preference is not the only
  judging mode.
- Tests: the swap logic (a test that fails if orders are not actually swapped),
  the bootstrap CI, kappa against a hand computed example, and the cache key.

## Acceptance criteria

- `make analysis` runs with no API keys from cached judgments.
- Human to human ceiling and the length baseline both appear in the README
  headline, not in an appendix.
- At least one negative or unflattering finding is stated plainly.

## Resume bullet this earns

"Built an evaluation harness measuring LLM-as-judge reliability against N
human labeled preference pairs: agreement versus the human to human ceiling,
position and verbosity bias, prompt sensitivity, and which debiasing
mitigations actually recover accuracy."
