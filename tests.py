"""Tests for the parts that would fail silently.

Run with `python tests.py` or `make test`. No framework: assertions and a main.

The order swapping tests are the important ones. A judge harness that does not
really swap the two answers produces a position bias number of zero and looks
great, so these tests are written to fail if the swap is faked.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

from harness import cache, judges, prompts, stats

HERE = os.path.dirname(os.path.abspath(__file__))
FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  pass  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILURES.append(name)


# --------------------------------------------------------------------------
# Order swapping


def test_swap_actually_swaps():
    print("swap logic")
    item = type(
        "Item",
        (),
        {
            "question": "Q?",
            "answer_a": "ALPHA ANSWER",
            "answer_b": "BETA ANSWER",
            "prior_turns": "[]",
        },
    )()
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner", os.path.join(HERE, "04_run_judges.py"))
    runner = importlib.util.module_from_spec(spec)
    sys.argv = ["tests"]
    spec.loader.exec_module(runner)

    ab = runner.build_prompt(item, "ab", "bare")
    ba = runner.build_prompt(item, "ba", "bare")

    check("the two orders produce different prompts", ab != ba)
    check(
        "order ab puts answer A first",
        ab.index("ALPHA ANSWER") < ab.index("BETA ANSWER"),
    )
    check(
        "order ba puts answer B first",
        ba.index("BETA ANSWER") < ba.index("ALPHA ANSWER"),
    )
    # The strongest form: the two prompts must be each other's swap, so nothing
    # other than the answer positions changed.
    check(
        "the orders differ only by the swap",
        ab.replace("ALPHA ANSWER", "\x00").replace("BETA ANSWER", "ALPHA ANSWER").replace("\x00", "BETA ANSWER") == ba,
    )

    prior = json.dumps([["earlier question", "A turn one", "B turn one"]])
    item.prior_turns = prior
    ab2 = runner.build_prompt(item, "ab", "bare")
    ba2 = runner.build_prompt(item, "ba", "bare")
    check(
        "earlier turns are swapped too, not just the judged turn",
        ab2.index("A turn one") < ab2.index("B turn one")
        and ba2.index("B turn one") < ba2.index("A turn one"),
    )


def test_positional_verdicts_map_back_correctly():
    print("verdict mapping")
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure", os.path.join(HERE, "05_measure.py"))
    measure = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)

    check("ab + first -> a", measure.canonical("first", "ab") == "a")
    check("ab + second -> b", measure.canonical("second", "ab") == "b")
    check("ba + first -> b", measure.canonical("first", "ba") == "b")
    check("ba + second -> a", measure.canonical("second", "ba") == "a")
    check("ties survive both orders", measure.canonical("tie", "ba") == "tie")

    # A judge that always says "the first one" must come out as a perfect flip,
    # never as a consistent preference for one model.
    check(
        "always picking position one shows up as a flip",
        measure.canonical("first", "ab") != measure.canonical("first", "ba"),
    )
    check("swap averaging keeps an agreement", measure.swap_average("a", "a") == "a")
    check("swap averaging turns a disagreement into a tie", measure.swap_average("a", "b") == "tie")


def test_released_winner_field_is_resolved():
    print("released judgment parsing")
    import importlib.util

    spec = importlib.util.spec_from_file_location("cacher", os.path.join(HERE, "03_cache_released.py"))
    cacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cacher)

    # In the released files the winner is the literal string 'model_1', which
    # names a field, not a model. Reading it as a model name marks every row an
    # error and quietly deletes the judge.
    check(
        "model_1 winner shown first is 'first'",
        cacher.positional("model_1", "alpaca-13b", "gpt-4", "alpaca-13b") == "first",
    )
    check(
        "model_1 winner shown second is 'second'",
        cacher.positional("model_1", "alpaca-13b", "gpt-4", "gpt-4") == "second",
    )
    check("tie stays a tie", cacher.positional("tie", "a", "b", "a") == "tie")
    check("error stays an error", cacher.positional("error", "a", "b", "a") == "error")


# --------------------------------------------------------------------------
# Statistics


def test_bootstrap_ci():
    print("bootstrap CI")
    hits = np.array([1.0] * 70 + [0.0] * 30)
    point, lo, hi, n = stats.bootstrap_ci(hits, n_boot=4000, seed=1)
    check("point estimate is the sample mean", abs(point - 0.70) < 1e-12, f"got {point}")
    check("n is the number included", n == 100)
    check("interval brackets the estimate", lo < point < hi, f"[{lo}, {hi}]")
    # Normal approximation half width for p=0.7, n=100 is 0.0898.
    check("width is near the analytic half width", abs((hi - lo) / 2 - 0.0898) < 0.02, f"[{lo}, {hi}]")

    # An all-hits sample has a degenerate interval, which is correct, not a bug.
    point, lo, hi, _ = stats.bootstrap_ci(np.ones(50), n_boot=1000, seed=2)
    check("a perfect score gives a degenerate interval", point == 1.0 and lo == 1.0 and hi == 1.0)

    # Excluded items must not move the estimate.
    hits = np.array([1.0, 0.0, 0.0])
    include = np.array([1.0, 1.0, 0.0])
    check("excluded items are excluded", stats.ratio(hits, include) == 0.5)

    # Wider intervals for smaller samples.
    _, lo_small, hi_small, _ = stats.bootstrap_ci(np.array([1.0] * 7 + [0.0] * 3), n_boot=4000, seed=3)
    check("smaller samples give wider intervals", (hi_small - lo_small) > (0.0898 * 2))

    # The interval is reproducible.
    a = stats.bootstrap_ci(hits, include, n_boot=500, seed=7)
    b = stats.bootstrap_ci(hits, include, n_boot=500, seed=7)
    check("same seed gives the same interval", a == b)


def test_cohens_kappa():
    print("Cohen's kappa")
    # Hand computed. 2x2 table: both yes 20, both no 15, a yes b no 5, a no b yes 10.
    # observed = 35/50 = 0.7
    # p(a=yes)=25/50=0.5, p(b=yes)=30/50=0.6
    # expected = 0.5*0.6 + 0.5*0.4 = 0.30 + 0.20 = 0.5
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    a = ["yes"] * 20 + ["no"] * 15 + ["yes"] * 5 + ["no"] * 10
    b = ["yes"] * 20 + ["no"] * 15 + ["no"] * 5 + ["yes"] * 10
    check("matches the hand computed 0.4", abs(stats.cohens_kappa(a, b) - 0.4) < 1e-12, stats.cohens_kappa(a, b))

    check("perfect agreement is 1", abs(stats.cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) - 1.0) < 1e-12)
    # Chance level agreement is 0: two raters splitting 50/50 with no relation.
    x = ["a", "a", "b", "b"]
    y = ["a", "b", "a", "b"]
    check("chance agreement is 0", abs(stats.cohens_kappa(x, y)) < 1e-12, stats.cohens_kappa(x, y))
    check("worse than chance is negative", stats.cohens_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0)


def test_krippendorff():
    print("Krippendorff's alpha")
    matrix = [["a", "b", "a", "b"], ["a", "b", "a", "b"]]
    check("identical raters give alpha 1", abs(stats.krippendorff_alpha(matrix, ["a", "b"]) - 1.0) < 1e-9)
    matrix = [["a", "b", "a", "b"], ["b", "a", "b", "a"]]
    check("opposed raters give alpha below 0", stats.krippendorff_alpha(matrix, ["a", "b"]) < 0)
    matrix = [["a", "b", None, "b"], ["a", None, "a", "a"]]
    check("missing votes are tolerated", isinstance(stats.krippendorff_alpha(matrix, ["a", "b"]), float))


# --------------------------------------------------------------------------
# Cache


def test_cache_key():
    print("cache key")
    k = cache.cache_key("judge", "bare", "prompt text", 0)
    check("stable across calls", k == cache.cache_key("judge", "bare", "prompt text", 0))
    check("64 hex characters", len(k) == 64 and all(c in "0123456789abcdef" for c in k))
    check("prompt changes the key", k != cache.cache_key("judge", "bare", "prompt texts", 0))
    check("judge changes the key", k != cache.cache_key("other", "bare", "prompt text", 0))
    check("template changes the key", k != cache.cache_key("judge", "cot", "prompt text", 0))
    check("sample index changes the key", k != cache.cache_key("judge", "bare", "prompt text", 1))
    check(
        "a one character prompt change changes the key",
        cache.cache_key("j", "t", "A first, B second", 0) != cache.cache_key("j", "t", "B first, A second", 0),
    )

    root = tempfile.mkdtemp()
    try:
        check("a miss returns None", cache.get("judge", k, root) is None)
        cache.put("judge", k, {"verdict": "first"}, root)
        check("a hit returns the record", cache.get("judge", k, root)["verdict"] == "first")
        check("load_all finds it", len(cache.load_all("judge", root)) == 1)
        check("an unknown judge loads empty", cache.load_all("nobody", root) == [])
    finally:
        shutil.rmtree(root)

    # When a comparison's two responses are byte identical, both orders render
    # the same prompt and land on the same cache key. The answer really is the
    # same, but both contexts have to survive or the comparison silently
    # disappears from the analysis. This regression cost 28 comparisons once.
    root = tempfile.mkdtemp()
    try:
        class OneAnswer:
            def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
                return {"text": "[[C]]", "in_tokens": 1, "out_tokens": 1, "latency_s": 0.0}

        for order in ("ab", "ba"):
            record = judges.run(
                OneAnswer(), "collide", "bare", "identical prompt", cache_root=root,
                extra={"item_id": "q1-t1-x-y", "order": order, "condition": "base"},
            )
        contexts = record["contexts"]
        check("both orders are remembered on one cached answer", len(contexts) == 2, contexts)
        check(
            "both orders name the same comparison",
            {c["order"] for c in contexts} == {"ab", "ba"} and {c["item_id"] for c in contexts} == {"q1-t1-x-y"},
        )
        check("the collision stores one file, not two", len(cache.load_all("collide", root)) == 1)
    finally:
        shutil.rmtree(root)

    # A judge is never called twice for the same prompt.
    calls = []

    class CountingBackend:
        def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
            calls.append(prompt)
            return {"text": "[[A]]", "in_tokens": 1, "out_tokens": 1, "latency_s": 0.0}

    root = tempfile.mkdtemp()
    try:
        for _ in range(3):
            record = judges.run(CountingBackend(), "counting", "bare", "same prompt", cache_root=root)
        check("the backend is called once for three identical requests", len(calls) == 1, f"{len(calls)} calls")
        check("the cached verdict is parsed", record["verdict"] == "first")
    finally:
        shutil.rmtree(root)


def test_verdict_parsing():
    print("verdict parsing")
    check("plain A", prompts.parse_verdict("[[A]]") == "first")
    check("plain B", prompts.parse_verdict("Some reasoning.\n\n[[B]]") == "second")
    check("tie", prompts.parse_verdict("[[C]]") == "tie")
    check("the last verdict wins", prompts.parse_verdict("maybe [[A]] but really [[B]]") == "second")
    check("no verdict is an error", prompts.parse_verdict("I cannot decide.") == "error")
    check("empty is an error", prompts.parse_verdict("") == "error")
    check("a bare trailing letter is accepted", prompts.parse_verdict("Verdict:\nB") == "second")
    check("a letter mid sentence is not a verdict", prompts.parse_verdict("Assistant A wrote more.") == "error")
    check("every template shares the output contract", all(
        prompts.VERDICT_CONTRACT in body for body in prompts.TEMPLATES.values()
    ))


def test_baselines():
    print("baselines")
    check("longer wins", judges.length_verdict(100, 50) == "first")
    check("shorter loses", judges.length_verdict(50, 100) == "second")
    check("equal is a tie", judges.length_verdict(50, 50) == "tie")
    draws = [judges.random_verdict(f"seed-{i}") for i in range(2000)]
    rate = draws.count("first") / len(draws)
    check("the coin is fair", 0.45 < rate < 0.55, f"{rate}")
    check("the coin never ties", "tie" not in draws)
    check("the coin is reproducible", judges.random_verdict("x") == judges.random_verdict("x"))


def test_padding_is_content_free():
    print("padding")
    import importlib.util

    spec = importlib.util.spec_from_file_location("runner", os.path.join(HERE, "04_run_judges.py"))
    runner = importlib.util.module_from_spec(spec)
    sys.argv = ["tests"]
    spec.loader.exec_module(runner)

    padded = runner.pad("short answer", 500)
    check("padding grows the answer past the target", len(padded) >= 500)
    check("padding keeps the original text intact", padded.startswith("short answer"))
    check("padding adds only the filler", padded.replace(runner.FILLER, "").strip() == "short answer")


def main():
    for test in (
        test_swap_actually_swaps,
        test_positional_verdicts_map_back_correctly,
        test_released_winner_field_is_resolved,
        test_bootstrap_ci,
        test_cohens_kappa,
        test_krippendorff,
        test_cache_key,
        test_verdict_parsing,
        test_baselines,
        test_padding_is_content_free,
    ):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
