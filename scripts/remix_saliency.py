#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Re-weight a per-source saliency file into any other calibration mixture.

REAP's saliency is what the calibration set made of each expert, so "what if the
mixture leaned towards code?" is normally another hour of calibration per
question. It does not have to be. ``sum_saliency`` is a plain sum of
``g_j * ||f_j||_2`` over the tokens routed to expert ``j`` and ``count`` is how
many there were, so both are additive over disjoint subsets of the calibration
set. Split them by source once -- ``reap_reference.py --sample-sources`` -- and
every mixture is exact arithmetic on what was already measured::

    mean_w(j) = (sum_s w_s * sum_s(j)) / (sum_s w_s * count_s(j))

The weights are relative shares of *tokens per source*, matching how the mixture
file is interpreted, so ``--weights code=1`` means "code only" and
``--weights code=2,ja=1`` means twice as much code as Japanese.

What this does **not** do is invent evidence. An expert that the code samples
never routed to has ``count = 0`` there, and no weighting makes it visible; a
mixture that leans hard on one source ranks the rest of the experts on very few
tokens. ``--report`` prints how many experts each mixture actually measured, so
a selection resting on twelve tokens is visible rather than implied.

Usage::

    remix_saliency.py --saliency artifacts/saliency.json \\
        --weights code=0.5,ja=0.25,en=0.25 --experts 128 \\
        --out artifacts/retained-code.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_weights(spec: str) -> dict[str, float]:
    out = {}
    for item in spec.split(","):
        if "=" not in item:
            raise SystemExit(f"bad --weights entry {item!r}, expected name=number")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saliency", type=Path, default=REPO / "artifacts/saliency.json")
    ap.add_argument("--sources", type=Path, default=REPO / "artifacts/calib-sources.json",
                    help="used only to map source names to languages, so weights can "
                         "be given per language as well as per source")
    ap.add_argument("--weights", required=True,
                    help="relative token shares, e.g. 'code=2,ja=1,en=1'. Names may be "
                         "sources or languages; unnamed ones get weight 0")
    ap.add_argument("--experts", type=int, default=128, help="how many to keep per layer")
    ap.add_argument("--out", type=Path, help="write retained sets here")
    ap.add_argument("--compare", type=Path, default=REPO / "artifacts/retained-sets.json",
                    help="report agreement against this existing selection")
    ap.add_argument("--report", action="store_true", default=True)
    args = ap.parse_args()

    doc = json.loads(args.saliency.read_text())
    layers = doc["layers"]
    probe = next(iter(layers.values()))
    if "by_source" not in probe:
        print(f"FAIL: {args.saliency} has no per-source split. It was written before "
              f"--sample-sources existed, and the split cannot be recovered from the "
              f"totals. Re-run the calibration with:\n"
              f"  reap_reference.py --sample-sources artifacts/calib-sources.json ...",
              file=sys.stderr)
        return 1

    lang = {}
    if args.sources.exists():
        lang = json.loads(args.sources.read_text()).get("lang", {})

    want = parse_weights(args.weights)
    names = sorted(probe["by_source"])
    weight = {n: want.get(n, want.get(lang.get(n, ""), 0.0)) for n in names}
    if not any(weight.values()):
        print(f"FAIL: none of {sorted(want)} matched a source {names} or language "
              f"{sorted(set(lang.values()))}", file=sys.stderr)
        return 1
    total = sum(weight.values())
    print("mixture:")
    for n in names:
        print(f"  {n:<18} {lang.get(n,'?'):>5} {weight[n]/total:>7.1%}")

    retained, thin = {}, []
    for name in sorted(layers):
        split = layers[name]["by_source"]
        n_exp = len(layers[name]["mean"])
        num = [0.0] * n_exp
        den = [0.0] * n_exp
        for src, w in weight.items():
            if not w or src not in split:
                continue
            s, c = split[src]["sum_saliency"], split[src]["count"]
            for j in range(n_exp):
                num[j] += w * s[j]
                den[j] += w * c[j]
        mean = [num[j] / den[j] if den[j] > 0 else 0.0 for j in range(n_exp)]
        keep = sorted(sorted(range(n_exp), key=lambda j: -mean[j])[: args.experts])
        retained[name] = keep
        measured = sum(1 for j in keep if den[j] > 0)
        if measured < args.experts:
            thin.append((name, args.experts - measured))

    if args.compare and args.compare.exists():
        base = json.loads(args.compare.read_text())
        shared = [len(set(retained[k]) & set(base[k])) for k in retained if k in base]
        if shared:
            print(f"\nagreement with {args.compare.name}: "
                  f"{sum(shared)/len(shared):.1f}/{args.experts} per layer "
                  f"({sum(shared)/len(shared)/args.experts:.1%}), "
                  f"range {min(shared)}-{max(shared)}")

    if thin:
        worst = sorted(thin, key=lambda t: -t[1])[:3]
        print(f"\nWARNING: {len(thin)} layer(s) keep experts this mixture never routed "
              f"to (worst: " + ", ".join(f"{n} {k} unmeasured" for n, k in worst) + ")")

    if args.out:
        args.out.write_text(json.dumps(retained, indent=1, sort_keys=True))
        print(f"\nwrote {args.out} ({len(retained)} layers x {args.experts} experts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
