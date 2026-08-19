#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Recover which source each calibration sample came from.

``build_calibration.py`` collects each source in turn, concatenates the results,
shuffles with a seeded permutation and writes **only** the token ids -- the
source labels are dropped. ``reap_reference.py --sample-sources`` needs them
back, because a saliency split by source is what turns one hour of calibration
into every mixture weighting instead of one.

The order is reconstructible rather than guessable: the per-source sample counts
come from the mixture's weights by the same largest-remainder allocation the
builder uses, the sources are concatenated in file order, and the shuffle is
``torch.randperm`` under ``mix["seed"]`` (0 by default). This script redoes that
and inverts it.

**It refuses rather than guesses.** A source that came up short of its requested
count shifts every label after it, and nothing in ``calib.pt`` would reveal that
-- the file is just a list of equal-length id lists. So the reconstruction is
only accepted when the allocation sums exactly to the number of samples present;
otherwise pass ``--counts`` with the real per-source numbers from the build log
(the ``name  N samples`` column), which is the one place they were printed.

Usage::

    label_calibration.py --mix calib/mix-dsv4.json --tokens artifacts/calib.pt \\
        --out artifacts/calib-sources.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def allocate(sources: list[dict], total: int) -> list[int]:
    """Largest-remainder allocation, the same one build_calibration.py uses.

    Plain rounding drifts by a sample or two per source, which would misalign
    every label after the first drift.
    """
    wsum = sum(s["weight"] for s in sources)
    exact = [total * s["weight"] / wsum for s in sources]
    n = [int(e) for e in exact]
    short = total - sum(n)
    order = sorted(range(len(sources)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for i in range(short):
        n[order[i % len(order)]] += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=Path, default=REPO / "calib/mix-dsv4.json")
    ap.add_argument("--tokens", type=Path, default=REPO / "artifacts/calib.pt")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/calib-sources.json")
    ap.add_argument("--counts", type=str,
                    help="override the allocation, e.g. 'ja-wiki=97,en-chat=102,...' "
                         "-- use the sample counts the build log printed when any "
                         "source came up short of its request")
    args = ap.parse_args()

    mix = json.loads(args.mix.read_text())
    sources = mix["sources"]
    samples = torch.load(args.tokens, weights_only=False)
    total = len(samples)

    if args.counts:
        override = dict(kv.split("=") for kv in args.counts.split(","))
        counts = [int(override[s["name"]]) for s in sources]
    else:
        counts = allocate(sources, total)

    if sum(counts) != total:
        print(f"FAIL: allocation sums to {sum(counts)} but {args.tokens} holds "
              f"{total} samples. A source came up short at build time; pass "
              f"--counts with the numbers from the build log.", file=sys.stderr)
        return 1

    ordered: list[str] = []
    for spec, n in zip(sources, counts):
        ordered.extend([spec["name"]] * n)

    seed = mix.get("seed", 0)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=g).tolist()
    labels = [ordered[i] for i in order]

    by_name = {s["name"]: s for s in sources}
    got = Counter(labels)
    print(f"{args.tokens.name}: {total} samples, seed {seed}")
    print(f"{'source':<18} {'lang':>5} {'samples':>8} {'share':>7} {'mix weight':>11}")
    wsum = sum(s["weight"] for s in sources)
    for spec in sources:
        n = got[spec["name"]]
        print(f"{spec['name']:<18} {spec.get('lang','?'):>5} {n:>8} "
              f"{n/total:>6.1%} {spec['weight']/wsum:>10.1%}")

    args.out.write_text(json.dumps({
        "mix": str(args.mix), "tokens": str(args.tokens), "seed": seed,
        "counts": {s["name"]: c for s, c in zip(sources, counts)},
        "lang": {s["name"]: s.get("lang", "?") for s in sources},
        "sample_sources": labels,
    }, indent=1))
    print(f"\nwrote {args.out}")
    print("Note: this reproduces the builder's ordering; it cannot detect a source "
          "that silently came up short. Cross-check the shares above against the "
          "build log before trusting a re-weighting built on them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
