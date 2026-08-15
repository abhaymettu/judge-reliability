"""Judge backends.

Every backend exposes the same call:

    backend.complete(prompt, max_tokens, temperature, seed) -> dict

with keys text, in_tokens, out_tokens, latency_s. `run` wraps a backend with
the cache and the verdict parser, so callers never touch either.

Adding a judge means adding one class with one method. See USAGE.md.
"""

import json
import os
import random
import time
import urllib.request

from . import cache, prompts


class MLXBackend:
    """A local model through mlx-lm. No API key, no network at judge time."""

    def __init__(self, model_path="mlx-community/Qwen2.5-7B-Instruct-4bit"):
        from mlx_lm import load

        self.model_path = model_path
        self.model, self.tokenizer = load(model_path)

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        in_tokens = len(self.tokenizer.encode(chat))
        sampler = make_sampler(temp=temperature)
        start = time.perf_counter()
        text = generate(
            self.model,
            self.tokenizer,
            prompt=chat,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        latency = time.perf_counter() - start
        return {
            "text": text,
            "in_tokens": in_tokens,
            "out_tokens": len(self.tokenizer.encode(text)),
            "latency_s": latency,
        }


class _HTTPBackend:
    url = None
    env_key = None

    def _post(self, payload, headers):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        return data, time.perf_counter() - start


class AnthropicBackend(_HTTPBackend):
    url = "https://api.anthropic.com/v1/messages"
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self, model="claude-sonnet-5"):
        self.model = model

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        data, latency = self._post(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "content-type": "application/json",
                "x-api-key": os.environ[self.env_key],
                "anthropic-version": "2023-06-01",
            },
        )
        return {
            "text": "".join(b.get("text", "") for b in data["content"]),
            "in_tokens": data["usage"]["input_tokens"],
            "out_tokens": data["usage"]["output_tokens"],
            "latency_s": latency,
        }


class OpenAIBackend(_HTTPBackend):
    url = "https://api.openai.com/v1/chat/completions"
    env_key = "OPENAI_API_KEY"

    def __init__(self, model="gpt-4o"):
        self.model = model

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        data, latency = self._post(
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "seed": seed,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "content-type": "application/json",
                "authorization": "Bearer " + os.environ[self.env_key],
            },
        )
        return {
            "text": data["choices"][0]["message"]["content"],
            "in_tokens": data["usage"]["prompt_tokens"],
            "out_tokens": data["usage"]["completion_tokens"],
            "latency_s": latency,
        }


class OllamaBackend(_HTTPBackend):
    url = "http://localhost:11434/api/chat"

    def __init__(self, model="llama3.1:8b"):
        self.model = model

    def complete(self, prompt, max_tokens, temperature=0.0, seed=0):
        data, latency = self._post(
            {
                "model": self.model,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
                "messages": [{"role": "user", "content": prompt}],
            },
            {"content-type": "application/json"},
        )
        return {
            "text": data["message"]["content"],
            "in_tokens": data.get("prompt_eval_count", 0),
            "out_tokens": data.get("eval_count", 0),
            "latency_s": latency,
        }


BACKENDS = {
    "mlx": MLXBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def run(backend, judge_id, template, prompt, sample=0, temperature=0.0, cache_root=None, extra=None):
    """Cached judge call. Returns the stored record, calling the model only on a miss.

    `extra` is bookkeeping stored alongside the answer (item id, order, condition).
    It is deliberately not part of the cache key: the prompt already determines
    the answer, and putting labels in the key would silently duplicate work.
    """
    key = cache.cache_key(judge_id, template, prompt, sample)
    hit = cache.get(judge_id, key, cache_root)
    if hit is not None:
        return hit
    out = backend.complete(
        prompt, prompts.MAX_TOKENS[template], temperature=temperature, seed=1000 + sample
    )
    record = {
        "key": key,
        "judge_id": judge_id,
        "template": template,
        "sample": sample,
        "verdict": prompts.parse_verdict(out["text"]),
        "text": out["text"],
        "in_tokens": out["in_tokens"],
        "out_tokens": out["out_tokens"],
        "latency_s": out["latency_s"],
    }
    record.update(extra or {})
    cache.put(judge_id, key, record, cache_root)
    return record


# Non LLM baselines. They take the same positional view of an item as an LLM
# judge does, so the same scoring code handles all of them.


def length_verdict(len_first, len_second):
    """Always prefer the longer response. The bar every judge has to clear."""
    if len_first == len_second:
        return "tie"
    return "first" if len_first > len_second else "second"


def random_verdict(rng_seed):
    """Coin flip, never a tie. Seeded so the repo reproduces."""
    return "first" if random.Random(rng_seed).random() < 0.5 else "second"
