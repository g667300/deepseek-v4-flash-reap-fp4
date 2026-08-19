#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Push the REAP calibration tokens through a served checkpoint.

The main stack's experts were chosen from saliency measured over
``artifacts/calib.pt`` -- 512 x 2048 tokens, seven sources. Measuring the
drafter's expert usage on anything else would pick its experts from a different
distribution than the one the model around it was pruned for, and the two
decisions would not be about the same data. So this replays the calibration
tokens themselves.

They are sent as **token ids**, not text: vLLM's completions API accepts a list
of ints as ``prompt``, which skips a detokenize/retokenize round trip that would
not reproduce the calibration set exactly.

Prompt tokens alone would only exercise prefill, and the drafter only runs
during decode -- so each sample is sent as a prefix plus a short greedy
continuation, which is what puts the draft head to work on this distribution.

Pair it with ``moe_usage_probe.py`` mounted into the server as
``sitecustomize.py``; this script only drives traffic, the probe does the
counting.

Usage::

    replay_calibration.py --url http://192.168.100.2:8000 --model dsv4-probe \\
        --samples 128 --prefix 1024 --generate 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def post(url: str, path: str, payload: dict, timeout: float = 600) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://192.168.100.2:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=Path, default=REPO / "artifacts/calib.pt")
    ap.add_argument("--samples", type=int, default=128,
                    help="calibration samples to replay (default: 128 of 512)")
    ap.add_argument("--prefix", type=int, default=1024,
                    help="prompt tokens taken from the head of each sample")
    ap.add_argument("--generate", type=int, default=64,
                    help="tokens to generate per sample -- this is the part the "
                         "draft head actually runs on")
    args = ap.parse_args()

    import torch

    blob = torch.load(args.tokens, weights_only=True)
    samples = [torch.as_tensor(t, dtype=torch.long).tolist() for t in blob]
    samples = samples[: args.samples]
    print(f"replaying {len(samples)} of {len(blob)} calibration samples, "
          f"{args.prefix} prompt tokens + {args.generate} generated")

    started = time.perf_counter()
    generated = 0
    for i, ids in enumerate(samples, start=1):
        payload = {"model": args.model, "prompt": ids[: args.prefix],
                   "max_tokens": args.generate, "temperature": 0.0}
        try:
            result = post(args.url, "/v1/completions", payload)
        except urllib.error.HTTPError as e:
            print(f"FAIL on sample {i}: HTTP {e.code} {e.read().decode()[:200]}",
                  file=sys.stderr)
            return 1
        generated += result.get("usage", {}).get("completion_tokens", 0)
        if i % 16 == 0 or i == len(samples):
            elapsed = time.perf_counter() - started
            print(f"  [{i}/{len(samples)}] {generated:,} tokens generated, "
                  f"{elapsed / 60:.1f} min, {generated / elapsed:.1f} tok/s", flush=True)

    print(f"done: {generated:,} tokens over {(time.perf_counter() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
