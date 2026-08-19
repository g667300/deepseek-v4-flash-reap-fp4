#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Check the re-encoded checkpoint against the two things it was built from.

Step 3 has more of a reference than step 2 left behind, and it is worth using:

* **Every quantized weight must decode back exactly.** A weight in the pruned
  BF16 checkpoint came out of DeepSeek's own FP4 or FP8, so it carries at most
  the significand that format can hold and is representable in BF16. Encoding it
  again is therefore lossless, and dequantizing what step 3 wrote has to return
  the step-2 tensor **bit for bit**. Any difference is an encoder bug or a
  flipped bit, and this cannot tell you which -- but it can tell you it happened.
* **Every plain tensor must equal the reference.** Norms, sinks, ``ape``, the
  hyper-connection scalars: REAP does not touch them and step 3 copies them
  across, so they must be identical to the original checkpoint, dtype included.
* **The three tensors REAP rewrites** are checked against what it wrote: the
  sliced router, the remapped ``tid2eid``, and the router bias, which must be
  the reference's fp32 bias indexed by the retained set.

What this does *not* prove is that the pruning was right -- there is nothing
left to compare that against, which is the cost the README describes. It proves
the re-encoding and the copying were faithful.

Mismatches are re-read before being reported, so a bad read is not confused with
a bad checkpoint, and every differing bit is printed with its byte lane -- see
docs/memory-faults.md for why that is the number that matters here.

Usage::

    verify_quantized.py --src artifacts/dsv4-reap50-bf16 \\
        --dst artifacts/dsv4-reap50 --reference models/DeepSeek-V4-Flash-0731 \\
        [--sample 0.05] [--journal artifacts/verify-q.journal]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import to_hf  # noqa: E402
from quant import dequantize_linear  # noqa: E402
from verify_dequantized import INDEX_NAME, Shards, report_flip, same  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GATE_BIAS = "ffn.gate.bias"
FROM_STEP2 = ("ffn.gate.weight", "ffn.gate.tid2eid", GATE_BIAS)
MTP_EXPERT = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
MTP_ROUTER = re.compile(r"^mtp\.(\d+)\.ffn\.gate\.(weight|bias)$")


def mtp_keep_sets(config: dict, retained: dict) -> dict[str, torch.Tensor] | None:
    """Which source experts each carried MTP block's slots came from.

    carry_mtp.py records one of two things: the main-stack layer whose retained
    set it borrowed, or -- when it scored the blocks itself -- the sets outright.
    Neither is derivable from the checkpoint, so without one of them the carried
    tensors cannot be checked at all.
    """
    explicit = config.get("_mtp_retained")
    if explicit:
        return {b: torch.as_tensor(k, dtype=torch.long) for b, k in explicit.items()}
    layer = config.get("_mtp_pruned_with_layer")
    if layer is None:
        return None
    shared = torch.as_tensor(retained[f"model.layers.{layer}.mlp"], dtype=torch.long)
    return {"*": shared}


def mtp_expected(name: str, ref: Shards, keep: torch.Tensor) -> torch.Tensor:
    """What a carried MTP tensor should be: the source's own bytes, re-indexed.

    carry_mtp.py copies these without dequantizing, so unlike everything else in
    this checkpoint they are still checkable against what DeepSeek published --
    exactly, in whatever dtype they were stored in.
    """
    expert = MTP_EXPERT.match(name)
    if expert is not None:
        block, slot, projection, part = expert.groups()
        original = int(keep[int(slot)])
        return ref.get(f"mtp.{block}.ffn.experts.{original}.{projection}.{part}")
    if MTP_ROUTER.match(name) is not None:
        return ref.get(name)[keep]
    return ref.get(name)


def step2_name(name: str, src: Shards) -> str:
    """The step-2 spelling of a native name.

    ``save_original_format=True`` writes ``head.weight`` natively already, so a
    name step 2 has verbatim is not pushed through the table -- ``to_hf`` would
    turn it into ``lm_head.weight``, which that checkpoint does not contain.
    """
    return name if name in src.weight_map else to_hf(name)


def expected(name: str, dst: Shards, src: Shards, ref: Shards,
             retained: dict[str, list[int]], restored: bool) -> tuple[torch.Tensor, str]:
    """What ``name`` in the output should be, and which side it came from."""
    scale = f"{name[: -len('.weight')]}.scale" if name.endswith(".weight") else None
    if scale and scale in dst.weight_map:
        return src.get(step2_name(name, src)).to(torch.bfloat16), "step2"

    if name.endswith(GATE_BIAS) and restored:
        layer = name.split(".")[1]
        keep = torch.as_tensor(retained[f"model.layers.{layer}.mlp"], dtype=torch.long)
        return ref.get(name)[keep], "reference (sliced)"

    if any(name.endswith(suffix) for suffix in FROM_STEP2) or not restored:
        want = src.get(step2_name(name, src))
        return want.to(ref.get(name).dtype) if name in ref.weight_map else want, "step2"

    return ref.get(name), "reference"


def decoded(dst: Shards, name: str) -> torch.Tensor:
    prefix = name[: -len(".weight")]
    raw = {name: dst.get(name), f"{prefix}.scale": dst.get(f"{prefix}.scale")}
    return dequantize_linear(raw, prefix, scheme="deepseek", dtype=torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO / "artifacts/dsv4-reap50-bf16",
                    help="the pruned BF16 checkpoint step 3 read")
    ap.add_argument("--dst", type=Path, default=REPO / "artifacts/dsv4-reap50",
                    help="the re-encoded checkpoint to check")
    ap.add_argument("--reference", type=Path,
                    default=REPO / "models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--retained", type=Path, default=REPO / "artifacts/retained-sets.json")
    ap.add_argument("--no-restore-untouched", action="store_true",
                    help="the checkpoint was built with the same flag: plain tensors "
                         "come from --src, not from the reference")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="fraction of tensors to check, 0-1 (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--journal", type=Path,
                    help="record each verdict as it is reached and skip names already "
                         "in the file on a restart")
    args = ap.parse_args()

    restored = not args.no_restore_untouched
    maps = {}
    for label, root in (("src", args.src), ("dst", args.dst), ("ref", args.reference)):
        index = root / INDEX_NAME
        if not index.exists():
            print(f"FAIL: no checkpoint index under {root}", file=sys.stderr)
            return 1
        maps[label] = json.loads(index.read_text())["weight_map"]
    src = Shards(args.src, maps["src"])
    dst = Shards(args.dst, maps["dst"])
    ref = Shards(args.reference, maps["ref"])
    retained = json.loads(args.retained.read_text())

    # carry_mtp.py records which layer's retained set it pruned the MTP blocks
    # with; without that there is no way to know which source expert a carried
    # slot came from, so the blocks are reported rather than guessed at.
    config = json.loads((args.dst / "config.json").read_text())
    mtp_keep = mtp_keep_sets(config, retained)

    problems = []
    hf_missing = []
    unexplained_mtp = []
    for name in maps["dst"]:
        if name.endswith(".scale"):
            continue
        if name.startswith("mtp."):
            if mtp_keep is None:
                unexplained_mtp.append(name)
            continue
        try:
            if step2_name(name, src) not in maps["src"]:
                hf_missing.append(name)
        except KeyError:
            hf_missing.append(name)
    if unexplained_mtp:
        problems.append(
            f"{len(unexplained_mtp)} mtp tensor(s) but no _mtp_pruned_with_layer in "
            f"config.json, so nothing says which experts they are, e.g. "
            f"{unexplained_mtp[:3]}")
    if hf_missing:
        problems.append(f"{len(hf_missing)} name(s) with no step-2 counterpart, "
                        f"e.g. {hf_missing[:3]}")
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    if problems:
        return 1

    names = sorted(n for n in maps["dst"] if not n.endswith(".scale"))
    if args.sample < 1.0:
        rng = random.Random(args.seed)
        names = sorted(rng.sample(names, max(1, int(len(names) * args.sample))))

    header = f"# {args.src} + {args.reference} -> {args.dst}"
    done: dict[str, str] = {}
    journal = None
    if args.journal:
        if args.journal.exists():
            lines = args.journal.read_text().splitlines()
            if lines and lines[0] != header:
                print(f"FAIL: journal is from another run: {lines[0]}", file=sys.stderr)
                return 1
            for line in lines[1:]:
                verdict, _, name = line.partition("\t")
                if name and verdict in ("ok", "reread", "bad"):
                    done[name] = verdict
        else:
            args.journal.write_text(header + "\n")
        journal = args.journal.open("a")

    print(f"output    : {args.dst}")
    print(f"step 2    : {args.src}")
    print(f"reference : {args.reference}"
          + ("" if restored else "  (plain tensors expected from step 2)"))
    print(f"checking {len(names):,} of "
          f"{sum(1 for n in maps['dst'] if not n.endswith('.scale')):,} tensors"
          + (f", {len(done):,} carried over from the journal" if done else ""))

    mismatched: list[str] = []
    bad_reads: list[str] = []
    flips = 0

    def want_of(name: str) -> torch.Tensor:
        if name.startswith("mtp."):
            block = name.split(".")[1]
            return mtp_expected(name, ref, mtp_keep.get(block, mtp_keep.get("*")))
        return expected(name, dst, src, ref, retained, restored)[0]

    for i, name in enumerate(names, start=1):
        verdict = done.get(name)
        if verdict is None:
            # An MTP tensor was copied byte for byte, so it is compared as
            # stored -- decoding it would only weaken the check.
            quantized = (not name.startswith("mtp.")) and name.endswith(".weight") \
                and f"{name[: -len('.weight')]}.scale" in maps["dst"]
            got = decoded(dst, name) if quantized else dst.get(name)
            want = want_of(name)
            if same(got, want):
                verdict = "ok"
            else:
                # Read both sides again before calling it: this machine has a
                # history of transient bad reads (docs/memory-faults.md).
                again = decoded(dst, name) if quantized else dst.reread(name)
                want2 = want_of(name)
                if same(again, want2):
                    verdict = "reread"
                    flips += report_flip(None if quantized else dst, name,
                                         "output-read", got, want2)
                else:
                    verdict = "bad"
                    flips += report_flip(None if quantized else dst, name,
                                         "on-disk", again, want2)
                    print(f"  {name}: {tuple(again.shape)} {again.dtype} vs "
                          f"{tuple(want2.shape)} {want2.dtype}", flush=True)
            if journal is not None:
                journal.write(f"{verdict}\t{name}\n")
                journal.flush()
        if verdict == "bad":
            mismatched.append(name)
        elif verdict == "reread":
            bad_reads.append(name)
        if i % 500 == 0 or i == len(names):
            print(f"  [{i:,}/{len(names):,}] {len(mismatched)} mismatched, "
                  f"{len(bad_reads)} bad read(s)", flush=True)

    if journal is not None:
        journal.close()

    print()
    if bad_reads:
        print(f"{len(bad_reads)} tensor(s) differed on the first read and matched on the "
              f"second -- the checkpoint is fine, the machine is not: {bad_reads[:5]}")
    if mismatched:
        print(f"FAIL: {len(mismatched)} tensor(s) do not match, e.g. {mismatched[:5]}")
        return 1
    print(f"all {len(names):,} checked tensors match"
          + (f" ({flips} flipped bit(s) reported)" if flips else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
