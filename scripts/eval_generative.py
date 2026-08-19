#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Score multiple-choice benchmarks by *generation*, so any served stack can run them.

The usual harness scores multiple choice by log-likelihood: it asks the server
for the logprob of each option and picks the largest. That needs prompt
logprobs, which vLLM exposes through ``echo`` and llama-server does not -- its
``/v1/completions`` ignores ``echo`` and returns a logprob only for the token it
generated. So the standard comparison cannot be run against a GGUF at all.

Generating the answer instead needs nothing but text completion, which every
stack has. It measures something slightly different -- whether the model *says*
the right letter, not whether it ranks it highest -- and it is the more honest
question when the thing being compared is two deployments rather than two sets
of weights. Both sides get the same prompt, the same parse, the same scoring.

Comparability is the whole point, so the prompt is fixed here rather than taken
from a task config: five-shot from the benchmark's own dev split where one
exists, the question, its options as ``A.``-``D.``, and ``Answer:``. The reply is
capped at a few tokens and the first A-D character in it is the answer. A reply
with no letter counts as wrong, and is reported separately -- a stack that
rambles instead of answering is failing at the task, but for a different reason
than one that answers wrongly, and the two should not be silently merged.

Usage::

    eval_generative.py --url http://192.168.100.2:8000/v1 --model dsv4-eval \\
        --tasks mmlu --limit 20 --out artifacts/gen-mmlu-reap.json
    eval_generative.py --url http://192.168.100.2:8080/v1 --model x \\
        --tasks global_mmlu:ja,global_mmlu:fr --limit 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LETTERS = "ABCD"
FIRST_LETTER = re.compile(r"\b([A-D])\b")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def complete(url: str, model: str, prompt: str, timeout: float, max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "stop": ["\n\n"]}).encode()
    req = urllib.request.Request(f"{url}/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["choices"][0]["text"]
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def render(row: dict, shots: list[dict]) -> str:
    def one(r: dict, with_answer: bool) -> str:
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
        tail = f" {LETTERS[r['answer']]}" if with_answer else ""
        return f"{r['question'].strip()}\n{opts}\nAnswer:{tail}"
    parts = [one(s, True) for s in shots] + [one(row, False)]
    return "\n\n".join(parts)


def load_task(spec: str, limit: int) -> tuple[list[dict], list[dict]]:
    """Return (rows, few-shot examples). Rows carry question/choices/answer."""
    from datasets import load_dataset

    if spec == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        dev = load_dataset("cais/mmlu", "all", split="dev")
        rows = [{"question": r["question"], "choices": r["choices"],
                 "answer": int(r["answer"]), "group": r["subject"]}
                for r in ds.shuffle(seed=0).select(range(min(limit, len(ds))))]
        shots = [{"question": r["question"], "choices": r["choices"],
                  "answer": int(r["answer"])} for r in dev.select(range(5))]
        return rows, shots

    if spec.startswith("global_mmlu:"):
        lang = spec.split(":", 1)[1]
        ds = load_dataset("CohereForAI/global-mmlu-lite", lang, split="test")
        rows = []
        for r in ds.shuffle(seed=0).select(range(min(limit, len(ds)))):
            choices = [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]
            rows.append({"question": r["question"], "choices": choices,
                         "answer": LETTERS.index(r["answer"].strip().upper()),
                         "group": lang})
        # global-mmlu-lite has no dev split; the first few test rows serve as
        # shots and are then excluded from scoring, so nothing is scored twice.
        shots, rows = rows[:5], rows[5:]
        return rows, shots

    raise SystemExit(f"unknown task {spec!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="OpenAI-compatible base, e.g. .../v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True,
                    help="comma-separated: mmlu, global_mmlu:ja, global_mmlu:fr, ...")
    ap.add_argument("--limit", type=int, default=100, help="rows per task")
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    url = args.url.rstrip("/")
    results = {}
    for spec in args.tasks.split(","):
        spec = spec.strip()
        rows, shots = load_task(spec, args.limit + 5)
        log(f"{spec}: {len(rows)} question(s), {len(shots)} shot(s)")

        def ask(row: dict) -> tuple[str | None, int]:
            text = complete(url, args.model, render(row, shots), args.timeout,
                            args.max_tokens)
            m = FIRST_LETTER.search(text.upper())
            return (m.group(1) if m else None), row["answer"]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            out = list(pool.map(ask, rows))

        by_group: dict[str, list[int]] = defaultdict(list)
        correct = unparsed = 0
        for (got, want), row in zip(out, rows):
            hit = got is not None and LETTERS.index(got) == want
            correct += hit
            unparsed += got is None
            by_group[row["group"]].append(int(hit))
        n = len(rows)
        acc = correct / n if n else 0.0
        log(f"{spec}: {acc:.2%} ({correct}/{n}), {unparsed} unparseable, "
            f"{(time.perf_counter()-t0)/60:.1f} min")
        results[spec] = {"accuracy": acc, "correct": correct, "n": n,
                         "unparseable": unparsed,
                         "by_group": {k: sum(v) / len(v) for k, v in sorted(by_group.items())}}

    print()
    for spec, r in results.items():
        print(f"{spec:<22} {r['accuracy']:>7.2%}  ({r['correct']}/{r['n']}, "
              f"{r['unparseable']} unparseable)")
    if args.out:
        args.out.write_text(json.dumps(
            {"url": url, "model": args.model, "limit": args.limit,
             "max_tokens": args.max_tokens, "results": results}, indent=1))
        log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
