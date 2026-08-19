#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Verify the BF16 -> DeepSeek FP4/FP8 encoders against the checkpoint itself.

The reference REAP path goes out through transformers, so the pruned model
comes back as BF16 and has to be re-encoded. That re-encoding is the one step
with no oracle to copy from -- DeepSeek ships ``convert.py``, which only ever
*reads* the format -- so it is checked against the checkpoint instead.

The invariant that makes this possible: a dequantized weight is ``code *
2**e``, at most three significant bits, so every value is exactly
representable in BF16. A correct encoder must therefore land on exactly the
same numbers when the same weight goes back through it. That is checked here
against real tensors, per block, and on adversarial synthetic input:

1. ``dequant(quant(dequant(w))) == dequant(w)`` bit for bit, on real FP4
   expert weights and real FP8 attention weights;
2. the encoder's own fixed point -- a second pass changes nothing, so the
   surgery is idempotent and a re-run cannot drift;
3. scales follow ``2**ceil(log2(amax / type_max))`` from the checkpoint's own
   ``inference/kernel.py``, checked directly against the amax of each block;
4. E2M1 rounding is round-to-nearest-even, checked on every tie and every
   midpoint rather than on random data, where ties never come up;
5. where the bytes do *not* match the source, the difference is confined to
   what the format cannot round-trip: the sign of a zero, and blocks whose
   stored scale was looser than the block's amax required.

Run: ``.venv/bin/python scripts/test_quant_encode.py [--ckpt DIR]``  (CPU only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import (  # noqa: E402
    FP4_BLOCK,
    FP4_MAX,
    FP8_BLOCK,
    FP8_E4M3_MAX,
    dequantize_deepseek_fp4,
    dequantize_deepseek_fp8,
    quantize_deepseek_fp4,
    quantize_deepseek_fp8,
    unpack_fp4,
)

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "models" / "DeepSeek-V4-Flash-0731"
FP4_TENSORS = ("layers.10.ffn.experts.0.w1", "layers.10.ffn.experts.7.w2")
FP8_TENSORS = ("layers.10.attn.wq_b", "layers.10.ffn.shared_experts.w1")
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def read_pair(ckpt: Path, weight_map: dict, prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    out = []
    for suffix in ("weight", "scale"):
        name = f"{prefix}.{suffix}"
        with safe_open(str(ckpt / weight_map[name]), framework="pt") as f:
            out.append(f.get_tensor(name))
    return out[0], out[1]


# --------------------------------------------------------------------------


def test_e2m1_rounding() -> None:
    """Every tie and every midpoint, which random data never produces."""
    print("\n== E2M1 rounding ==")
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # Ties round to an even code: 0.0, 1.0, 2.0 and 4.0 win their midpoints.
    ties = {0.25: 0.0, 0.75: 1.0, 1.25: 1.0, 1.75: 2.0, 2.5: 2.0, 3.5: 4.0, 5.0: 4.0}

    # one 32-wide block, scale forced to 2**0 by planting 6.0 in it
    def encode(values: list[float]) -> torch.Tensor:
        row = torch.full((1, FP4_BLOCK), 6.0, dtype=torch.bfloat16)
        row[0, : len(values)] = torch.tensor(values, dtype=torch.bfloat16)
        w, s = quantize_deepseek_fp4(row)
        return dequantize_deepseek_fp4(w, s, torch.float32)[0, : len(values)]

    exact = encode(magnitudes)
    check("representable magnitudes are preserved",
          torch.equal(exact, torch.tensor(magnitudes)),
          f"got {exact.tolist()}")

    got = encode(list(ties))
    want = torch.tensor(list(ties.values()))
    check("ties round to even", torch.equal(got, want),
          f"got {got.tolist()}, want {want.tolist()}")

    neg = encode([-v for v in ties])
    check("ties round to even, negative", torch.equal(neg, -want),
          f"got {neg.tolist()}")

    # An outlier drags its whole block's scale up with it -- amax is what the
    # scale is derived from -- so the thing to check is that nothing wraps:
    # the outlier stays the largest value in the block and keeps its sign,
    # and no magnitude escapes 6x the scale.
    row = torch.full((1, FP4_BLOCK), 0.01, dtype=torch.bfloat16)
    row[0, 0], row[0, 1] = -100.0, 0.02
    w, s = quantize_deepseek_fp4(row)
    out = dequantize_deepseek_fp4(w, s, torch.float32)[0]
    check("an outlier keeps its sign and stays the block's largest value",
          out[0] < 0 and out.abs().argmax().item() == 0,
          f"outlier came back as {out[0].item()}")
    check("no magnitude escapes 6x the scale",
          bool((out.abs() <= FP4_MAX * s.float()[0, 0]).all()))


def test_scale_rule() -> None:
    """The scale is the kernel's, derived from each block's amax."""
    print("\n== scale derivation ==")
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(64, 4 * FP4_BLOCK, generator=g) * 0.05).bfloat16()
    _, scale = quantize_deepseek_fp4(w)
    amax = w.float().unflatten(-1, (4, FP4_BLOCK)).abs().amax(-1)
    want = torch.exp2(torch.ceil(torch.log2(amax.double() / FP4_MAX)))
    check("FP4 scale == 2**ceil(log2(amax/6))",
          torch.equal(scale.float().double(), want),
          f"max ratio {(scale.float().double() / want).max().item()}")

    w8 = (torch.randn(FP8_BLOCK, 2 * FP8_BLOCK, generator=g) * 3.0).bfloat16()
    _, scale8 = quantize_deepseek_fp8(w8)
    amax8 = w8.float().unflatten(0, (1, FP8_BLOCK)).unflatten(-1, (2, FP8_BLOCK))
    amax8 = amax8.abs().amax(dim=(1, 3))
    want8 = torch.exp2(torch.ceil(torch.log2(amax8.double() / FP8_E4M3_MAX)))
    check("FP8 scale == 2**ceil(log2(amax/448))",
          torch.equal(scale8.float().double(), want8))

    zeros = torch.zeros(1, FP4_BLOCK, dtype=torch.bfloat16)
    zw, zs = quantize_deepseek_fp4(zeros)
    check("an all-zero block encodes as zeros, not NaN",
          bool((dequantize_deepseek_fp4(zw, zs, torch.float32) == 0).all())
          and not bool(zs.float().isnan().any()))


def test_round_trip(ckpt: Path, weight_map: dict) -> None:
    """Real weights: the values have to come back exactly."""
    for prefix in FP4_TENSORS:
        print(f"\n== FP4 round trip: {prefix} ==")
        w, s = read_pair(ckpt, weight_map, prefix)
        ref = dequantize_deepseek_fp4(w, s, torch.bfloat16)
        w2, s2 = quantize_deepseek_fp4(ref)
        again = dequantize_deepseek_fp4(w2, s2, torch.bfloat16)
        print(f"  weight {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)}")

        check("values are bit-exact", torch.equal(ref, again),
              f"max |diff| {(ref.float() - again.float()).abs().max().item():.6g}")
        check("scales are unchanged",
              torch.equal(s2.view(torch.uint8), s.view(torch.uint8)))

        # Bytes may still differ, but only where the format lost the sign of a
        # zero: nibble 0x8 decodes to +0.0, so the sign is gone by the time the
        # encoder sees it and it comes back as 0x0.
        src = unpack_fp4(w.view(torch.uint8), torch.float32)
        differs = (w2.view(torch.uint8) != w.view(torch.uint8))
        lo = differs & ((w.view(torch.uint8) & 0x0F) != (w2 & 0x0F))
        hi = differs & ((w.view(torch.uint8) >> 4) != (w2 >> 4))
        bad = (lo & (src[..., 0::2] != 0)) | (hi & (src[..., 1::2] != 0))
        check("byte differences are only negative zeros",
              not bool(bad.any()),
              f"{int(differs.sum())} bytes differ, "
              f"{int(bad.sum())} of them on a nonzero value")

        w3, s3 = quantize_deepseek_fp4(again)
        check("encoder is idempotent",
              torch.equal(w3.view(torch.uint8), w2.view(torch.uint8))
              and torch.equal(s3.view(torch.uint8), s2.view(torch.uint8)))

    for prefix in FP8_TENSORS:
        print(f"\n== FP8 round trip: {prefix} ==")
        w, s = read_pair(ckpt, weight_map, prefix)
        ref = dequantize_deepseek_fp8(w, s, torch.bfloat16)
        w2, s2 = quantize_deepseek_fp8(ref)
        again = dequantize_deepseek_fp8(w2, s2, torch.bfloat16)
        print(f"  weight {tuple(w.shape)} {w.dtype}, scale {tuple(s.shape)}")

        check("values are bit-exact", torch.equal(ref, again),
              f"max |diff| {(ref.float() - again.float()).abs().max().item():.6g}")

        # A scale is allowed to tighten: the source stored one that left the
        # block's largest value short of 448, and the encoder picks the
        # smallest power of two that fits. The values are unchanged either way.
        same = (s2.view(torch.uint8) == s.view(torch.uint8))
        tighter = (s2.float() <= s.float())
        check("scales are unchanged or tighter", bool(tighter.all()),
              f"{same.float().mean().item():.2%} identical")

        w3, s3 = quantize_deepseek_fp8(again)
        check("encoder is idempotent",
              torch.equal(w3.view(torch.uint8), w2.view(torch.uint8))
              and torch.equal(s3.view(torch.uint8), s2.view(torch.uint8)))


def test_error_bound() -> None:
    """Encoding weights that were never quantized costs what FP4 costs."""
    print("\n== quantization error on unquantized input ==")
    g = torch.Generator().manual_seed(11)
    w = (torch.randn(256, 8 * FP4_BLOCK, generator=g) * 0.02).bfloat16()
    q, s = quantize_deepseek_fp4(w)
    back = dequantize_deepseek_fp4(q, s, torch.float32)
    rel = (back - w.float()).norm() / w.float().norm()
    # E2M1 has 1 mantissa bit; the worst relative step between adjacent
    # magnitudes is 0.5/1.5, so a third is the ceiling and ~7% is typical.
    check("relative error is within FP4's resolution", rel.item() < 0.12,
          f"relative L2 error {rel.item():.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    test_e2m1_rounding()
    test_scale_rule()
    test_error_bound()

    index = args.ckpt / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        test_round_trip(args.ckpt, weight_map)
    else:
        print(f"\n== real weights == \n  skipped, no checkpoint at {args.ckpt}")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
