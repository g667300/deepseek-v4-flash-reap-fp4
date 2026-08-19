#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Build a pruned checkpoint at any expert count, straight from the reference.

The full pipeline costs an hour of calibration, a 272 GB BF16 intermediate and
28 minutes of re-quantization. **None of it is needed to change the sparsity**,
because REAP does not modify the experts it keeps. ``prune_moe_layer`` rebuilds
the ``ModuleList`` from the survivors -- ``((str(i), experts[pos]) for i, pos in
enumerate(retained))`` -- and slices the router. No merging, no rescaling. So
expert slot *i* of the pruned model *is* reference expert ``retained[i]``, and
the quantized bytes for it are already sitting in the reference checkpoint.

Verified rather than assumed: against the finished 128-expert build, every
expert tensor dequantizes to exactly the same values as the reference expert its
retained set names (max |diff| 0). About half the raw bytes differ, and only in
one way -- FP4 nibble ``8`` against ``0``, negative zero against positive zero,
which E2M1 gives the same value. Step 1 dropped the sign when it widened to
BF16 and step 3 never put it back. Copying the reference bytes therefore
preserves *more* of the original than the pipeline does.

What still has to be computed is small:

* ``ffn.gate.weight`` and ``ffn.gate.bias`` -- row selection, and the bias comes
  from the reference's fp32 rather than a bf16 round trip;
* ``ffn.gate.tid2eid`` on the three hash-routed layers -- the frozen table names
  experts by index, so every dropped one is remapped onto a survivor by
  ``build_value_map``, which needs the *unpruned* router rows and so is computed
  here from the reference exactly as the modifier computes it mid-run.

Everything else is copied byte for byte. ``mtp.*`` is skipped; ``carry_mtp.py``
adds the draft head, and its expert choice is a separate measurement.

Usage::

    build_pruned.py --experts 160 --out artifacts/dsv4-reap160
    build_pruned.py --retained artifacts/retained-sets.json --out artifacts/rebuilt-128
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv4_patches import build_value_map  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INDEX = "model.safetensors.index.json"
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)$")
LAYER_KEY = re.compile(r"(\d+)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def retained_from_saliency(path: Path, keep_n: int) -> dict[int, list[int]]:
    """Top-``keep_n`` per layer by REAP's own selection statistic.

    ``REAPSaliencyTracker.compute_retained_experts`` reads ``mean_saliency``, so
    this is a topk over the same numbers the run itself selected on -- which is
    why a different sparsity needs no recalibration at all.
    """
    layers = json.loads(path.read_text())["layers"]
    out = {}
    for name, data in layers.items():
        mean = data["mean"]
        idx = int(LAYER_KEY.search(name).group())
        out[idx] = sorted(sorted(range(len(mean)), key=lambda j: -mean[j])[:keep_n])
    return out


def retained_from_file(path: Path) -> dict[int, list[int]]:
    doc = json.loads(path.read_text())
    return {int(LAYER_KEY.search(k).group()): v for k, v in doc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path,
                    default=REPO / "models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--saliency", type=Path, default=REPO / "artifacts/saliency.json")
    ap.add_argument("--retained", type=Path,
                    help="use these retained sets instead of a topk over the saliency")
    ap.add_argument("--experts", type=int,
                    help="how many experts to keep per layer (with --saliency)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hash-remap", default="balanced", choices=("balanced", "nearest"))
    ap.add_argument("--shard-size", type=float, default=4.0, help="GB per shard")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.retained and not args.experts:
        print("FAIL: pass --experts N or --retained FILE", file=sys.stderr)
        return 1
    if args.out.exists():
        print(f"FAIL: {args.out} exists", file=sys.stderr)
        return 1

    keeps = (retained_from_file(args.retained) if args.retained
             else retained_from_saliency(args.saliency, args.experts))
    sizes = {len(v) for v in keeps.values()}
    if len(sizes) != 1:
        print(f"FAIL: retained sets differ in size: {sorted(sizes)}", file=sys.stderr)
        return 1
    keep_n = sizes.pop()
    log(f"{len(keeps)} layer(s), keeping {keep_n} expert(s) each")

    index = json.loads((args.reference / INDEX).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    config = json.loads((args.reference / "config.json").read_text())
    n_orig = config["n_routed_experts"]

    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(args.reference / shard), framework="pt")
        return handles[shard].get_tensor(name)

    # Plan: what is copied, what is rebuilt, what is dropped.
    copied, rebuilt, dropped = [], [], 0
    slot_of: dict[int, dict[int, int]] = {L: {e: i for i, e in enumerate(v)}
                                          for L, v in keeps.items()}
    for name in weight_map:
        if name.startswith("mtp."):
            continue
        m = EXPERT.match(name)
        if m:
            L, e = int(m.group(1)), int(m.group(2))
            if L in slot_of and e in slot_of[L]:
                copied.append(name)
            else:
                dropped += 1
            continue
        if name.endswith("ffn.gate.weight") or name.endswith("ffn.gate.bias") \
                or name.endswith("ffn.gate.tid2eid"):
            L = int(LAYER_KEY.search(name).group())
            (rebuilt if L in keeps else copied).append(name)
            continue
        copied.append(name)

    log(f"copy {len(copied):,} tensor(s) byte for byte, rebuild {len(rebuilt)}, "
        f"drop {dropped:,}")
    if args.dry_run:
        return 0

    args.out.mkdir(parents=True)
    out_map: dict[str, str] = {}
    shard_bytes = int(args.shard_size * 1e9)
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    shard_no = 0
    total = 0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_no, total
        if not pending:
            return
        shard_no += 1
        path = args.out / f"model-{shard_no:05d}.part"
        save_file(pending, str(path), metadata={"format": "pt"})
        for k in pending:
            out_map[k] = path.name
        total += pending_bytes
        log(f"  {path.name}: {len(pending):,} tensor(s), {pending_bytes/1e9:.1f} GB")
        pending, pending_bytes = {}, 0

    def add(name: str, tensor: torch.Tensor) -> None:
        nonlocal pending_bytes
        pending[name] = tensor
        pending_bytes += tensor.numel() * tensor.element_size()
        if pending_bytes >= shard_bytes:
            flush()

    # The hash table's remap needs the *unpruned* router rows, which is why it is
    # computed from the reference here rather than read off a pruned checkpoint.
    for name in rebuilt:
        L = int(LAYER_KEY.search(name).group())
        keep = keeps[L]
        if name.endswith("tid2eid"):
            vmap = torch.tensor(
                build_value_map(get(f"layers.{L}.ffn.gate.weight").float(),
                                keep, args.hash_remap), dtype=torch.long)
            table = get(name)
            add(name, vmap[table.cpu().long()].to(table.dtype))
        else:
            idx = torch.tensor(keep, dtype=torch.long)
            add(name, get(name).index_select(0, idx).contiguous())

    for name in copied:
        m = EXPERT.match(name)
        if m:
            L, e, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            add(f"layers.{L}.ffn.experts.{slot_of[L][e]}.{rest}", get(name))
        else:
            add(name, get(name))
    flush()

    for path in sorted(args.out.glob("model-*.part")):
        final = path.with_name(f"{path.stem}-of-{shard_no:05d}.safetensors")
        path.rename(final)
        for k, v in out_map.items():
            if v == path.name:
                out_map[k] = final.name

    (args.out / INDEX).write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": out_map}, indent=2))

    config["n_routed_experts"] = keep_n
    config["_pruned_from"] = str(args.reference)
    config["_built_by"] = "build_pruned.py (byte copy from the reference)"
    config["_hash_remap"] = args.hash_remap
    (args.out / "config.json").write_text(json.dumps(config, indent=2))

    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "LICENSE", ".gitattributes"):
        src = args.reference / extra
        if src.exists():
            (args.out / extra).write_bytes(src.read_bytes())

    log(f"wrote {len(out_map):,} tensor(s), {total/1e9:.1f} GB, {shard_no} shard(s) "
        f"-> {args.out}")
    log("next: carry_mtp.py to add the draft head")
    return 0


if __name__ == "__main__":
    sys.exit(main())
