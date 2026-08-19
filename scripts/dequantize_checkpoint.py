#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Dequantize the DeepSeek-V4 checkpoint to BF16, leaving every name alone.

This is the price of admission for the reference REAP path: llm-compressor
reads the model through transformers, transformers reads it through
compressed-tensors, and compressed-tensors does not know DeepSeek's FP4/FP8
format. So the whole thing is expanded to BF16 first -- 166.9 GB in, **568.7
GB out** with the MTP block excluded.

Names are *not* touched. transformers 5.14 ships a conversion mapping for
``deepseek_v4`` that reads DeepSeek's native spellings directly, so the only
change is that each ``X.weight`` / ``X.scale`` pair becomes a plain BF16
``X.weight``.

**Shard size is a real dial, not a formality.** The offload cache that the
REAP run works under writes one file per offloaded module holding the tensors
of the *shard* that module came from, so the offload footprint scales with
shard size. The source's 48 shards would come out at 11.9 GB each; the default
here is 4 GB, which cuts that footprint by about a third for no cost beyond
more files.

``mtp.*`` is dropped by default. transformers ignores those tensors on load
(``_keys_to_ignore_on_load_unexpected``), so carrying them through this
expansion would cost 39.7 GB to produce something no later stage can read.
``carry_mtp.py`` takes the block from the original shards instead, still
quantized.

Usage::

    dequantize_checkpoint.py --src models/DeepSeek-V4-Flash-0731 \\
        --dst artifacts/dsv4-bf16 --workers 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import dequantize_linear  # noqa: E402

INDEX_NAME = "model.safetensors.index.json"
# Advertised safetensors dtypes and their byte width. FP4 arrives as I8 with
# two values packed per byte, which is why the element count is doubled below.
DTYPE_BYTES = {
    "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F64": 8,
}
PACKED = ("U8", "I8")


def read_header(path: Path) -> dict:
    """The safetensors header, without touching the tensor data."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def bf16_bytes(meta: dict) -> int:
    numel = 1
    for dim in meta["shape"]:
        numel *= dim
    if meta["dtype"] in PACKED:
        numel *= 2      # two FP4 nibbles per byte
    return numel * 2


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def plan(src: Path, keep_mtp: bool, shard_bytes: int) -> tuple[list[list[str]], dict, int]:
    """Group every surviving tensor into output shards of about ``shard_bytes``.

    Returns ``(shards, source_shard_of_name, total_bytes)``. Tensors are kept
    in the source's own order so an output shard draws from as few input
    shards as possible, which keeps each worker's reads sequential.
    """
    weight_map: dict[str, str] = json.loads((src / INDEX_NAME).read_text())["weight_map"]

    headers: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        headers[shard] = read_header(src / shard)

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    total = 0

    for shard in sorted(set(weight_map.values())):
        for name, meta in headers[shard].items():
            if name.endswith(".scale"):
                continue        # consumed together with its .weight
            if not keep_mtp and name.startswith("mtp."):
                continue
            size = bf16_bytes(meta)
            if current and current_bytes + size > shard_bytes:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
            total += size

    if current:
        shards.append(current)
    return shards, weight_map, total


# --------------------------------------------------------------------------
# the work
# --------------------------------------------------------------------------


def build_shard(
    src_dir: str, dst_path: str, names: list[str], weight_map_json: str
) -> tuple[str, dict[str, int], int]:
    """Dequantize one output shard. Returns (filename, {name: nbytes}, numel)."""
    src = Path(src_dir)
    weight_map: dict[str, str] = json.loads(weight_map_json)
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = weight_map[name]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(src / shard), framework="pt")
            handles[shard] = handle
        return handle.get_tensor(name)

    tensors: dict[str, torch.Tensor] = {}
    for name in names:
        if not name.endswith(".weight"):
            tensors[name] = get(name)       # biases, norms, tid2eid, hc params
            continue
        prefix = name[: -len(".weight")]
        raw = {name: get(name)}
        scale = f"{prefix}.scale"
        if scale in weight_map:
            raw[scale] = get(scale)
        tensors[name] = dequantize_linear(
            raw, prefix, scheme="deepseek", dtype=torch.bfloat16
        )

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dst_path, metadata={"format": "pt"})

    sizes = {n: t.numel() * t.element_size() for n, t in tensors.items()}
    return Path(dst_path).name, sizes, sum(t.numel() for t in tensors.values())


def patch_config(src: Path, dst: Path, kept_mtp: bool) -> None:
    """Strip the quantization declaration; the weights are plain BF16 now."""
    config = json.loads((src / "config.json").read_text())
    config.pop("expert_dtype", None)        # "fp4": what the experts *were*
    config.pop("scale_fmt", None)           # "ue8m0": how the scales were stored
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    if not kept_mtp:
        # The blocks are gone from this copy. carry_mtp.py restores the count
        # when it puts them back into the final checkpoint.
        config["num_nextn_predict_layers"] = 0
    config["_dequantized_from"] = str(src)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def copy_aux(src: Path, dst: Path) -> list[str]:
    """Tokenizer and friends. Not `inference/`: it describes the FP4 format."""
    copied = []
    for path in sorted(src.iterdir()):
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        if path.name in {INDEX_NAME, "config.json", "hf_quant_config.json"}:
            continue
        shutil.copy2(path, dst / path.name)
        copied.append(path.name)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="quantized source checkpoint")
    ap.add_argument("--dst", required=True, type=Path, help="BF16 output directory")
    ap.add_argument("--shard-size", type=float, default=4.0,
                    help="target output shard size in GB. Smaller shards mean a "
                         "smaller offload footprint during the REAP run, which "
                         "copies per shard (default: 4)")
    ap.add_argument("--keep-mtp", action="store_true",
                    help="expand the MTP block too. transformers drops those "
                         "tensors on load, so this only costs disk")
    ap.add_argument("--workers", type=int, default=3,
                    help="shards built in parallel; each holds one in memory")
    ap.add_argument("--dry-run", action="store_true", help="report the plan only")
    ap.add_argument("--max-shards", type=int,
                    help="stop after this many shards. The result is a partial "
                         "checkpoint that will not load -- for checking the "
                         "output against the source before committing 568 GB")
    args = ap.parse_args()

    shard_bytes = int(args.shard_size * 1e9)
    shards, weight_map, total = plan(args.src, args.keep_mtp, shard_bytes)
    if args.max_shards:
        shards = shards[: args.max_shards]
        total = sum(
            bf16_bytes(read_header(args.src / weight_map[n])[n])
            for names in shards for n in names
        )
        print(f"PARTIAL: stopping after {len(shards)} shards (--max-shards)")

    print(f"source     : {args.src}")
    print(f"output     : {args.dst}")
    print(f"tensors    : {sum(len(s) for s in shards):,} in {len(shards)} shards "
          f"of ~{args.shard_size:.1f} GB")
    print(f"total size : {total / 1e9:.1f} GB"
          + ("" if args.keep_mtp else "  (mtp.* excluded)"))

    free = shutil.disk_usage(args.dst.parent if args.dst.parent.exists() else ".").free
    print(f"free disk  : {free / 1e9:.1f} GB")
    if free < total * 1.02:
        print(f"\nNOT ENOUGH DISK: need {total / 1e9:.1f} GB, have {free / 1e9:.1f} GB",
              file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    weight_map_json = json.dumps(weight_map)
    new_map: dict[str, str] = {}
    written = params = done = 0

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {}
        for i, names in enumerate(shards, start=1):
            name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            futures[pool.submit(
                build_shard, str(args.src), str(args.dst / name), names, weight_map_json
            )] = name

        for future in as_completed(futures):
            name, sizes, numel = future.result()
            new_map.update(dict.fromkeys(sizes, name))
            written += sum(sizes.values())
            params += numel
            done += 1
            print(f"  [{done}/{len(shards)}] {name}  {sum(sizes.values()) / 1e9:.2f} GB"
                  f"  ({written / 1e9:.1f}/{total / 1e9:.1f} GB)", flush=True)

    (args.dst / INDEX_NAME).write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": params, "total_size": written},
                "weight_map": dict(sorted(new_map.items())),
            },
            indent=2,
        ) + "\n"
    )
    patch_config(args.src, args.dst, args.keep_mtp)
    copy_aux(args.src, args.dst)

    print(f"\nwrote {len(new_map):,} tensors, {written / 1e9:.1f} GB -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
