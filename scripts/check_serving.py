#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Ask a served checkpoint whether it works, and whether speculation is paying.

Two questions, and they fail differently:

* **Does it generate at all?** A pruned MoE that loads can still be broken --
  the router can point at experts that are no longer there, the hash tables can
  send tokens off the end of the expert range. That shows up as garbage or a
  500, not as a load error.
* **Is the MTP block being used?** vLLM only speculates when it is configured
  to, so "no acceptance" has two causes worth separating: not enabled, or
  enabled and rejected. ``vllm:spec_decode_num_draft_tokens`` distinguishes
  them -- zero drafts means it never tried.

The acceptance rate is the number that matters for this checkpoint in
particular. Its MTP block was pruned with another layer's retained set (REAP
never scored the MTP experts, see carry_mtp.py), so a draft head that proposes
tokens the main model rejects is the predicted way for that shortcut to show up.

Usage::

    check_serving.py --url http://192.168.100.2:8000 --model dsv4-ref
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    ("factual", "The capital of France is"),
    ("code", "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"),
    ("japanese", "日本の首都は"),
    ("repetition", "Count from one to ten: one, two,"),
]

# The prompt the published DSpark figures were measured with, verbatim, so our
# acceptance can be put beside theirs (0.825 code / 0.338 prose on the unpruned
# drafter) instead of beside a different workload. From bench/codebench.py in
# jacklarmer/deepseek-v4-flash-0731-sm120 (MIT); they run it at 512 tokens,
# temperature 0, through /v1/chat/completions, and note that acceptance is
# strongly workload-dependent, which is exactly why the prompt has to match.
BENCH_PROMPTS = [
    ("code-512", 512,
     "Write a complete, production-quality Python module that implements a "
     "thread-safe LRU cache with TTL expiry. Include: the full class with type "
     "hints, __getitem__/__setitem__/__delitem__, a background sweeper thread, "
     "explicit locking, a stats() method returning hits/misses/evictions, and "
     "pytest unit tests covering eviction order, TTL expiry, and concurrent "
     "access. Output only code."),
    ("prose-512", 512,
     "Explain, in careful prose and without code, why speculative decoding "
     "speeds up autoregressive generation, what determines the acceptance rate, "
     "and when it stops paying off."),
]


def call(url: str, path: str, payload: dict | None = None, timeout: float = 300):
    target = f"{url.rstrip('/')}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        target, data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body) if body.startswith(("{", "[")) else body


def metrics(url: str) -> dict[str, float]:
    """The counters vLLM exposes, flattened to {name: summed value}."""
    try:
        raw = call(url, "/metrics", timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"  (no /metrics: {e})")
        return {}
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{")[0]
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


def position_counts(url: str) -> dict[int, float]:
    """Accepted-token counts keyed by position within the draft.

    These are counters, not gauges: read them before and after and subtract, or
    a second run on the same server reports the first run's numbers added in.
    """
    try:
        raw = call(url, "/metrics", timeout=30)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, float] = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            continue
        match = re.search(r'position="(\d+)"', line)
        if match:
            out[int(match.group(1))] = float(line.rsplit(" ", 1)[1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.100.2:8000")
    ap.add_argument("--model", default="dsv4-ref")
    ap.add_argument("--tokens", type=int, default=64, help="tokens per prompt")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--chat", action="store_true", default=None,
                    help="use /v1/chat/completions, so the model's chat template "
                         "is applied (default: on with --bench, off otherwise)")
    ap.add_argument("--bench", action="store_true",
                    help="run the published DSpark benchmark prompts instead: code "
                         "and prose at 512 tokens, the workloads the 0.825/0.338 "
                         "acceptance figures were measured on")
    args = ap.parse_args()

    if args.chat is None:
        args.chat = args.bench
    prompts = ([(label, text, tokens) for label, tokens, text in BENCH_PROMPTS]
               if args.bench else
               [(label, text, args.tokens) for label, text in PROMPTS])

    try:
        models = call(args.url, "/v1/models", timeout=30)
        served = [m["id"] for m in models.get("data", [])]
        print(f"served models: {served}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot reach {args.url}: {e}", file=sys.stderr)
        return 1

    before = metrics(args.url)
    before_pos = position_counts(args.url)
    print()

    failures = 0
    total_tokens = total_time = 0.0
    for label, prompt, max_tokens in prompts:
        # DeepSeek-V4-Flash is a thinking model: hit /v1/completions with a raw
        # string and it answers as if it were already inside a thought block,
        # emitting "</think>" and then looping. The loop is trivially
        # predictable, so acceptance measured that way reads *higher* while the
        # output is garbage. The published harness uses chat completions, and so
        # must anything compared against it.
        if args.chat:
            path = "/v1/chat/completions"
            payload = {"model": args.model,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": args.temperature}
        else:
            path = "/v1/completions"
            payload = {"model": args.model, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": args.temperature}
        # Acceptance is strongly prompt-dependent -- published figures for this
        # model quote code and prose separately (0.825 / 0.338), so a single
        # mixed number cannot be compared against them.
        spec_before = metrics(args.url)
        t0 = time.perf_counter()
        try:
            result = call(args.url, path, payload)
        except urllib.error.HTTPError as e:
            print(f"[{label}] HTTP {e.code}: {e.read().decode()[:200]}")
            failures += 1
            continue
        elapsed = time.perf_counter() - t0
        spec_after = metrics(args.url)
        drafted = (spec_after.get("vllm:spec_decode_num_draft_tokens_total", 0)
                   - spec_before.get("vllm:spec_decode_num_draft_tokens_total", 0))
        took = (spec_after.get("vllm:spec_decode_num_accepted_tokens_total", 0)
                - spec_before.get("vllm:spec_decode_num_accepted_tokens_total", 0))
        choice = result["choices"][0]
        text = choice.get("text") or choice.get("message", {}).get("content", "")
        used = result.get("usage", {}).get("completion_tokens", 0)
        total_tokens += used
        total_time += elapsed
        rate = f", {took / drafted:.1%} accepted" if drafted else ""
        print(f"[{label}] {used} tokens in {elapsed:.1f}s "
              f"({used / elapsed if elapsed else 0:.1f} tok/s{rate})")
        print(f"    {prompt!r} -> {text[:160]!r}")
        if not text.strip():
            print("    EMPTY OUTPUT")
            failures += 1

    print(f"\noverall: {total_tokens:.0f} tokens in {total_time:.1f}s "
          f"({total_tokens / total_time if total_time else 0:.1f} tok/s)")

    # The counters that carry values are the ``_total`` ones; the bare names
    # exist only as ``_created`` timestamps, and reading those reports a live
    # drafter as "never drafted".
    def delta(name: str) -> float:
        return after.get(name, 0) - before.get(name, 0)

    after = metrics(args.url)
    drafts = delta("vllm:spec_decode_num_draft_tokens_total")
    accepted = delta("vllm:spec_decode_num_accepted_tokens_total")
    batches = delta("vllm:spec_decode_num_drafts_total")

    print("\nspeculative decoding:")
    if not any(k.startswith("vllm:spec_decode") for k in after):
        print("  the server exposes no spec_decode counters -- not enabled")
    elif drafts <= 0:
        print(f"  enabled but never drafted (drafts={drafts:.0f}) -- the draft "
              "head is not being used")
    else:
        print(f"  drafted {drafts:.0f} token(s) over {batches:.0f} step(s), "
              f"accepted {accepted:.0f} -- {accepted / drafts:.1%} acceptance, "
              f"{accepted / batches:.2f} token(s) per step")
        # Where in the draft the head stops being right is more useful than the
        # mean: a head that only ever lands its first token is a different
        # problem from one that decays gently. Rejection is terminal, so each
        # position's rate is conditional on the one before it.
        per_pos = {p: c - before_pos.get(p, 0) for p, c in
                   position_counts(args.url).items()}
        if per_pos:
            reached = batches
            parts = []
            for position in sorted(per_pos):
                count = per_pos[position]
                parts.append(f"{position}:{count:.0f}"
                             + (f" ({count / reached:.0%})" if reached else ""))
                reached = count
            print("  accepted by draft position: " + "  ".join(parts))

    if failures:
        print(f"\nFAIL: {failures} prompt(s) did not produce usable output")
        return 1
    print("\nall prompts produced output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
