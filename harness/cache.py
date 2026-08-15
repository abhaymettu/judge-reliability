"""Content addressed cache for judge calls.

The key is a SHA-256 over everything that could change the answer: judge id,
template, sample index, and the exact prompt text. Change the prompt by one
character and you get a new key, so a stale answer can never be served for a
changed prompt. Nothing else goes into the key, so re-running the pipeline on
the same inputs costs nothing.

One JSON file per judgment under data/judgments/<judge_id>/<hash>.json.
"""

import hashlib
import json
import os

CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "judgments")


def cache_key(judge_id, template, prompt, sample=0):
    payload = json.dumps(
        {"judge_id": judge_id, "template": template, "sample": sample, "prompt": prompt},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path(judge_id, key, root=None):
    return os.path.join(root or CACHE_ROOT, judge_id, key + ".json")


def get(judge_id, key, root=None):
    try:
        with open(_path(judge_id, key, root), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def add_context(record, context):
    """Attach one (item, order, condition) that this judgment answers.

    A single cached answer can serve more than one context. It happens whenever
    the two orders of a comparison render to the same prompt, which is exactly
    when the two responses are byte identical. Keying on the prompt is still
    right, the answer really is the same, but the record has to remember both
    contexts or the comparison quietly disappears from the analysis.
    """
    contexts = record.setdefault("contexts", [])
    if context not in contexts:
        contexts.append(context)
        contexts.sort(key=lambda c: (c.get("item_id", ""), c.get("order", ""), c.get("condition", "")))
        return True
    return False


def put(judge_id, key, record, root=None):
    path = _path(judge_id, key, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, sort_keys=True)
    os.replace(tmp, path)


def load_all(judge_id, root=None):
    """Every cached record for a judge, as a list."""
    directory = os.path.join(root or CACHE_ROOT, judge_id)
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            with open(os.path.join(directory, name), encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out
