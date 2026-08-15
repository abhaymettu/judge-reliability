"""Judge prompt templates and verdict parsing.

Three templates, varying only in how much scaffolding the judge is given:

    bare    the minimum that can produce a verdict
    rubric  explicit criteria and an explicit warning about length and order
    cot     think first in a <reasoning> block, then commit to a verdict

They all end in the same output contract, [[A]] / [[B]] / [[C]], so the parser
is shared and template differences cannot leak in through parsing differences.
"""

import re

VERDICT_CONTRACT = (
    "Output your verdict on the last line in exactly this format, and nothing "
    'after it: "[[A]]" if assistant A is better, "[[B]]" if assistant B is '
    'better, or "[[C]]" for a tie.'
)

BARE = """You are comparing two AI assistant responses to a user question.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

Which response is better?
""" + VERDICT_CONTRACT

RUBRIC = """You are an impartial judge comparing two AI assistant responses to a user question.

Judge on these criteria, in this order of importance:
1. Helpfulness: does it actually answer what was asked?
2. Correctness: are the facts, reasoning and code right?
3. Relevance: does it stay on the question?
4. Depth and detail, only where the question calls for them.

Two rules you must follow:
- Length is not quality. A longer answer is not better for being longer. Penalise
  padding, repetition and filler.
- The order the two answers are presented in is arbitrary and carries no
  information. Do not let position influence your verdict.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

""" + VERDICT_CONTRACT

COT = """You are an impartial judge comparing two AI assistant responses to a user question.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]

First reason step by step inside a <reasoning> block: what the question actually
asks for, where each answer succeeds, where each fails. Close the block with
</reasoning>. Then give your verdict.

""" + VERDICT_CONTRACT

TEMPLATES = {"bare": BARE, "rubric": RUBRIC, "cot": COT}

# How many tokens each template needs to produce. cot has to think first.
MAX_TOKENS = {"bare": 24, "rubric": 24, "cot": 400}

DEFAULT_TEMPLATE = "rubric"

_MULTI_TURN_NOTE = (
    "This is a multi-turn conversation. Judge the assistants on the FINAL user "
    "question only, using the earlier turns as context.\n\n"
)


def render(template, question, answer_a, answer_b, prior_turns=None):
    """Fill a template. `prior_turns` is a list of (user, assistant_a, assistant_b)
    for turns before the one being judged; None or empty for single turn items."""
    body = TEMPLATES[template]
    if prior_turns:
        context = []
        for i, (user, a, b) in enumerate(prior_turns, start=1):
            context.append(
                f"[Turn {i} user question]\n{user}\n\n"
                f"[Turn {i} assistant A]\n{a}\n\n"
                f"[Turn {i} assistant B]\n{b}\n"
            )
        question = _MULTI_TURN_NOTE + "\n".join(context) + "\n[Final user question]\n" + question
    return body.format(question=question, answer_a=answer_a, answer_b=answer_b)


_VERDICT_RE = re.compile(r"\[\[([ABC])\]\]")


def parse_verdict(text):
    """Return 'first', 'second', 'tie' or 'error'.

    'first' and 'second' are positional, not identity. Mapping a positional
    verdict back to a model is the caller's job, and it is where order bugs
    hide, so the parser refuses to do it.
    """
    if not text:
        return "error"
    found = _VERDICT_RE.findall(text)
    if not found:
        # Some models drop the brackets. Accept a bare trailing A/B/C only if it
        # is unambiguous on the last non-empty line.
        for line in reversed(text.strip().splitlines()):
            line = line.strip().strip(".*_ ")
            if line in ("A", "B", "C"):
                found = [line]
                break
    if not found:
        return "error"
    return {"A": "first", "B": "second", "C": "tie"}[found[-1]]
