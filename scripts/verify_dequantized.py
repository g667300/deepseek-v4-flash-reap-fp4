#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Check a dequantized checkpoint against the source it came from.

Dequantization is the one step in this pipeline that still has an independent
reference: the source is untouched, so every tensor in the output must equal
what dequantizing the corresponding source pair produces -- bit for bit, since
the operation is deterministic. Once the REAP run has been through it that is
no longer true of anything, which is exactly what this path gives up.

Worth running on non-ECC memory in particular. 568 GB of reads and writes is
enough exposure to matter, and a flip here lands in the model that gets pruned.
Mismatches are re-read before being reported, so a bad read is not confused
with a bad checkpoint.

Usage::

    verify_dequantized.py --src models/DeepSeek-V4-Flash-0731 \\
        --dst artifacts/dsv4-bf16 [--sample 0.05]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import dequantize_linear  # noqa: E402

INDEX_NAME = "model.safetensors.index.json"


class Shards:
    """Lazily opened safetensors handles, one per shard."""

    def __init__(self, root: Path, weight_map: dict[str, str]):
        self.root = root
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}
        self._headers: dict[str, tuple[int, dict]] = {}

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.root / shard), framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def reread(self, name: str) -> torch.Tensor:
        """Read again through a fresh handle, bypassing any cached mapping."""
        with safe_open(str(self.root / self.weight_map[name]), framework="pt") as f:
            return f.get_tensor(name)

    def file_offset(self, name: str) -> tuple[str, int]:
        """Shard and absolute byte offset of the tensor's first element."""
        shard = self.weight_map[name]
        header = self._headers.get(shard)
        if header is None:
            with open(self.root / shard, "rb") as f:
                size = int.from_bytes(f.read(8), "little")
                header = (8 + size, json.loads(f.read(size)))
            self._headers[shard] = header
        base, meta = header
        return shard, base + meta[name]["data_offsets"][0]


def report_flip(shards: Shards | None, name: str, side: str, bad: torch.Tensor,
                good: torch.Tensor) -> int:
    """Print the byte lane of every bit that differs between two reads.

    docs/memory-faults.md does this arithmetic by hand for each event, because
    the byte lane is the only thing that has ever separated one bad DRAM device
    from a bad cell -- and with the modules swapped it is the measurement that
    decides whether the DIMMs were ever the problem. Throwing the bytes away and
    reporting only the tensor name, as this did before, loses exactly that.

    One JSON object per flipped bit, matching what bitcheck.c emits. Pass
    shards=None for a value that was recomputed rather than read from a file:
    it has no file offset, so no lane can be derived and none is claimed.
    """
    if bad.shape != good.shape or bad.dtype != good.dtype or bad.dtype != torch.bfloat16:
        return 0
    b = bad.flatten().view(torch.int16).to(torch.int32) & 0xFFFF
    g = good.flatten().view(torch.int16).to(torch.int32) & 0xFFFF
    differing = torch.nonzero(b ^ g).flatten()
    shard, start = shards.file_offset(name) if shards else (None, None)
    found = 0
    for i in differing[:8].tolist():
        got, want = int(b[i]), int(g[i])
        for bit in range(16):
            if not (got ^ want) >> bit & 1:
                continue
            # little-endian: bits 8-15 of the u16 live in the odd byte
            off = start + i * 2 + bit // 8 if start is not None else None
            record = {
                "tensor": name, "file": shard, "side": side, "element": i,
                "file_offset": off,
                "got": f"0x{got:04x}", "want": f"0x{want:04x}",
                "xor": f"0x{got ^ want:04x}",
                "popcount": bin(got ^ want).count("1"),
                "bit_in_byte": bit % 8,
                "direction": f"{want >> bit & 1}->{got >> bit & 1}",
            }
            if off is not None:
                record |= {"line_offset": off % 64, "byte_lane": off % 8,
                           "bit_in_word": (off % 8) * 8 + bit % 8}
            print(json.dumps(record), flush=True)
            found += 1
    if differing.numel() > 8:
        print(f"  ... and {differing.numel() - 8:,} more differing elements in {name}",
              flush=True)
    return found


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def expected(src: Shards, name: str) -> torch.Tensor:
    prefix = name[: -len(".weight")] if name.endswith(".weight") else None
    scale = f"{prefix}.scale" if prefix else None
    if scale and scale in src.weight_map:
        raw = {name: src.get(name), scale: src.get(scale)}
        return dequantize_linear(raw, prefix, scheme="deepseek", dtype=torch.bfloat16)
    return src.get(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="quantized source")
    ap.add_argument("--dst", required=True, type=Path, help="dequantized output")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="fraction of tensors to check, 0-1 (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--journal", type=Path,
                    help="record each verdict as it is reached and skip names "
                         "already in the file on a restart. A pass over 568 GB "
                         "takes hours on a machine that has been dying inside it.")
    args = ap.parse_args()

    src_map = json.loads((args.src / INDEX_NAME).read_text())["weight_map"]
    out_map = json.loads((args.dst / INDEX_NAME).read_text())["weight_map"]
    src, dst = Shards(args.src, src_map), Shards(args.dst, out_map)

    problems: list[str] = []
    stray = sorted(n for n in out_map if n.endswith(".scale") or n.startswith("mtp."))
    unknown = sorted(n for n in out_map if n not in src_map)
    if stray:
        problems.append(f"{len(stray)} tensor(s) that should not be here, e.g. {stray[:3]}")
    if unknown:
        problems.append(f"{len(unknown)} name(s) absent from the source, e.g. {unknown[:3]}")

    names = sorted(out_map)
    if args.sample < 1.0:
        rng = random.Random(args.seed)
        names = sorted(rng.sample(names, max(1, int(len(names) * args.sample))))

    # A verdict already on disk is worth as much as one reached now -- each
    # tensor is compared against the untouched source independently -- so a run
    # cut short by a crash resumes instead of starting over.
    header = f"# {args.src} -> {args.dst}"
    done: dict[str, str] = {}
    journal = None
    if args.journal:
        if args.journal.exists():
            lines = args.journal.read_text().splitlines()
            if lines and lines[0] != header:
                print(f"FAIL: journal is from another run: {lines[0]}")
                return 1
            for line in lines[1:]:
                verdict, _, name = line.partition("\t")
                # a torn or garbled line names no real tensor and is redone
                if name and verdict in ("ok", "reread", "bad"):
                    done[name] = verdict
        else:
            args.journal.write_text(header + "\n")
        journal = args.journal.open("a")

    print(f"source : {args.src}")
    print(f"output : {args.dst}")
    print(f"checking {len(names):,} of {len(out_map):,} tensors"
          + (f", {len(done):,} carried over from the journal" if done else ""))

    mismatched: list[str] = []
    bad_reads: list[str] = []
    landed = {"ok": lambda n: None, "reread": bad_reads.append, "bad": mismatched.append}
    for i, name in enumerate(names, start=1):
        verdict = done.get(name)
        if verdict is None:
            first_got, first_want = dst.get(name), expected(src, name)
            if same(first_got, first_want):
                verdict = "ok"
            else:
                # Read both sides again before calling it. A verifier that cannot
                # tell a bad read from a bad checkpoint is worse than none.
                got, want = dst.reread(name), expected(src, name)
                if same(got, want):
                    verdict = "reread"
                    # got == want, so whichever first read disagrees with it is
                    # the one that was corrupted on the way through.
                    if not report_flip(dst, name, "output-read", first_got, want):
                        report_flip(None, name, "source-recompute", first_want, want)
                else:
                    verdict = "bad"
                    report_flip(dst, name, "on-disk", got, want)
            if journal is not None:
                journal.write(f"{verdict}\t{name}\n")
                journal.flush()
                if i % 500 == 0:  # the failure mode here is the box dying, not the process
                    os.fsync(journal.fileno())
        landed[verdict](name)
        if i % 2000 == 0:
            print(f"  {i:,}/{len(names):,}", flush=True)
    if journal is not None:
        journal.close()

    print()
    if bad_reads:
        print(f"{len(bad_reads)} tensor(s) matched only on re-read: {bad_reads[:5]}")
        print("  The bytes on disk are right and something corrupted them on the "
              "way through. Worth knowing about on non-ECC memory.")
    if mismatched:
        problems.append(f"{len(mismatched)} tensor(s) differ from the source, "
                        f"e.g. {mismatched[:3]}")

    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        return 1
    print(f"all {len(names):,} checked tensors match the source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
