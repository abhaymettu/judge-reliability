"""Fetch the public source data into data/raw/.

Network, but no API key and no model calls. Everything here is public and
licensed for reuse: MT-Bench human judgments are CC BY 4.0, the MT-Bench
questions and released GPT-4 judgments come from the LMSYS mt-bench Space.

Skips anything already downloaded, so it is safe to re-run.
"""

import os
import sys
import urllib.request

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

HUMAN = "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data"

# Pinned to the June 2023 revision of the Space. Later revisions replaced the
# judgment files with a run over a different, larger model set that does not
# include vicuna-13b-v1.2, so they no longer line up with the human labels.
# See DECISIONS.md.
SPACE_REV = "85425b615f50"
SPACE = f"https://huggingface.co/spaces/lmsys/mt-bench/resolve/{SPACE_REV}/data/mt_bench"

FILES = {
    "human.parquet": f"{HUMAN}/human-00000-of-00001-25f4910818759289.parquet",
    "gpt4_pair.parquet": f"{HUMAN}/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
    "question.jsonl": f"{SPACE}/question.jsonl",
    "gpt-4_pair.jsonl": f"{SPACE}/model_judgment/gpt-4_pair.jsonl",
    "gpt-4_single.jsonl": f"{SPACE}/model_judgment/gpt-4_single.jsonl",
}


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(RAW, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"have  {name}")
            continue
        print(f"get   {name}", flush=True)
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    total = sum(os.path.getsize(os.path.join(RAW, n)) for n in FILES)
    print(f"data/raw ready, {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
