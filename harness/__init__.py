"""Reusable LLM-as-judge reliability harness.

Four pieces, deliberately small:

    prompts.py  judge prompt templates and verdict parsing
    cache.py    content addressed judgment cache, so nothing is ever paid for twice
    judges.py   judge backends, all behind one `judge(item, order, template)` call
    stats.py    bootstrap CIs, Cohen's kappa, Krippendorff's alpha, agreement setups

See USAGE.md for how to point this at a new judge model.
"""
