#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Make an MTP-free clone of a finished checkpoint, ready for ``carry_mtp.py``.

Trying a different draft-head selection means rebuilding only the 5.7 GB of
``mtp.*`` tensors, not the 82 GB underneath them -- so the clone hardlinks every
shard that holds no MTP tensor and rewrites the index without the MTP entries.
The shards are then provably the same bytes as the original rather than merely
equal, and the variant costs what its own MTP block costs.

**The metadata must not be hardlinked.** ``model.safetensors.index.json`` and
``config.json`` are small enough to look harmless, but Python's ``write_text``
truncates the *inode*: editing a hardlinked index rewrites the original
checkpoint's index too. That is not hypothetical here -- it cost
``artifacts/dsv4-reap50`` its MTP entries and its ``_mtp_pruned_with_layer`` on
2026-08-16, and the copy had to be restored from the Spark. So every file this
script writes is unlinked first and written fresh; only the shards are shared.

Usage::

    new_mtp_variant.py --src artifacts/dsv4-reap50 --dst artifacts/dsv4-reap50-salmean
    carry_mtp.py --dst artifacts/dsv4-reap50-salmean --score saliency ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX = "model.safetensors.index.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO / "artifacts/dsv4-reap50",
                    help="finished checkpoint to clone (default: artifacts/dsv4-reap50)")
    ap.add_argument("--dst", type=Path, required=True,
                    help="variant to create; must not exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (args.src / INDEX).exists():
        print(f"FAIL: no index at {args.src / INDEX}", file=sys.stderr)
        return 1
    if args.dst.exists():
        print(f"FAIL: {args.dst} exists; remove it first", file=sys.stderr)
        return 1

    index = json.loads((args.src / INDEX).read_text())
    weight_map: dict[str, str] = index["weight_map"]

    mtp_shards = {s for n, s in weight_map.items() if n.startswith("mtp.")}
    kept_map = {n: s for n, s in weight_map.items() if not n.startswith("mtp.")}
    kept_shards = sorted(set(kept_map.values()))
    # A shard holding both MTP and non-MTP tensors could not be shared, and
    # carry_mtp.py always writes MTP into files of its own -- so say so rather
    # than silently dropping tensors.
    mixed = sorted(mtp_shards & set(kept_shards))
    if mixed:
        print(f"FAIL: {len(mixed)} shard(s) mix MTP and non-MTP tensors, "
              f"e.g. {mixed[0]}", file=sys.stderr)
        return 1

    others = sorted(p.name for p in args.src.iterdir()
                    if p.is_file() and p.name != INDEX
                    and not p.name.endswith(".safetensors"))

    print(f"{args.src} -> {args.dst}")
    print(f"  link {len(kept_shards)} shard(s), drop {len(mtp_shards)} MTP shard(s)")
    print(f"  copy {len(others)} metadata file(s) + a rewritten index "
          f"({len(weight_map) - len(kept_map)} mtp entries removed)")
    if args.dry_run:
        return 0

    args.dst.mkdir(parents=True)
    for name in kept_shards:
        os.link(args.src / name, args.dst / name)
    for name in others:
        # copy, never link: config.json is edited in place by carry_mtp.py
        shutil.copy2(args.src / name, args.dst / name)

    total = sum((args.src / n).stat().st_size for n in kept_shards)
    out = dict(index)
    out["weight_map"] = kept_map
    out["metadata"] = {**index.get("metadata", {}), "total_size": total}
    path = args.dst / INDEX
    path.unlink(missing_ok=True)
    path.write_text(json.dumps(out, indent=2))

    print(f"  wrote {path} ({len(kept_map):,} tensors, {total / 1e9:.1f} GB shared)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
