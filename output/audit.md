# Data audit

## MT-Bench human judgments (lmsys/mt_bench_human_judgments, split `human`)

- 3355 individual votes
- 1814 distinct comparisons (question, model pair, turn)
- 65 annotators
- 80 questions, 6 models
- turns: {1: 1689, 2: 1666}
- vote labels: {'model_a': np.int64(1293), 'model_b': np.int64(1282), 'tie': np.int64(780)}

Annotators per comparison:

| annotators | comparisons |
| --- | --- |
| 1 | 853 |
| 2 | 561 |
| 3 | 271 |
| 4 | 92 |
| 5 | 25 |
| 6 | 10 |
| 7 | 2 |

853 of 1814 comparisons (47 percent) have a single annotator, so they contribute to judge agreement but cannot contribute to the human to human ceiling.

- empty final responses in the human split: 9

## Released GPT-4 pairwise judgments (LMSYS mt-bench Space, June 2023 revision)

- 9280 judged comparisons, 8480 distinct
- covers 1814 of 1814 human labelled comparisons
- order 1 verdicts: {'model_2': 5776, 'model_1': 2213, 'tie': 1289, 'error': 2}
- order 2 verdicts: {'model_2': 5976, 'model_1': 1957, 'tie': 1344, 'error': 3}
- rows missing one of the two orders: 0

## Released GPT-4 single answer grades

- 5280 graded responses
- both responses graded for 1229 of 1814 human comparisons
- models with no grades: {'vicuna-13b-v1.2': 160}

## Questions

- 80 questions across 8 categories: {'writing': 10, 'roleplay': 10, 'reasoning': 10, 'math': 10, 'coding': 10, 'extraction': 10, 'stem': 10, 'humanities': 10}

## Problems found

- 9 empty responses found in the human split
