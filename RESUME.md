# Framing

Built an evaluation harness that treats an LLM judge as a measurement instrument
and calibrates it against 1814 human labelled MT-Bench preference pairs:
agreement with the human majority against a like for like human to human ceiling,
position bias measured by judging every pair in both orders, verbosity bias
against a length only baseline and a controlled padding test, self preference,
prompt template sensitivity, and which debiasing mitigations actually recover
accuracy. The headline is a negative result held in view rather than buried: the
judge's order sensitivity is the same size as its disagreement with itself on
identical reruns, so swap averaging is variance reduction and not debiasing, and
a rule that always picks the longer answer already reaches 83 percent of the
frontier judge's agreement while reading none of the text. Every number
reproduces from a committed judgment cache with one command and no API key.

## Resume bullet

Built an evaluation harness measuring LLM-as-judge reliability against 1814 human
labelled preference pairs: agreement versus the human to human ceiling, position
and verbosity bias, prompt sensitivity, and which debiasing mitigations actually
recover accuracy.
