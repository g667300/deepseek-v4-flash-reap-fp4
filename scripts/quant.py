#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Dequantization schemes, one per checkpoint family.

Two families are supported today:

``nvfp4``
    nvidia/modelopt NVFP4: E2M1 nibbles,
    a per-16-column F8_E4M3 scale and an F32 global scale.  Implemented in
    :mod:`nvfp4`; re-exported here so callers only need one import.

``deepseek``
    The format DeepSeek ships in ``DeepSeek-V4-Flash-0731``.  Derived from the
    checkpoint's own ``inference/`` sources rather than guessed:

    * routed experts are FP4 -- ``convert.py`` stores ``[out, in/2]`` nibbles
      (advertised as ``I8`` in the safetensors header, the bytes are
      ``float4_e2m1fn_x2``) with ``FP4_TABLE`` = ``[0, .5, 1, 1.5, 2, 3, 4, 6]``
      then the same negated, i.e. bit 3 is the sign, and the **low nibble is the
      earlier element along K** (``convert.py:31-33``).  The scale is
      ``[out, in/32]`` F8_E8M0 -- one power-of-two per 32 elements along the
      reduction dim (``model.py`` ``Linear``: "1x32 quant on K").
    * everything else is FP8 E4M3 with a **128x128 block** scale, also E8M0
      (``config.json`` ``scale_fmt: ue8m0``).  ``convert.py:126`` spells the
      dequantization out: ``w.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128))
      .float() * scale[:, None, :, None]``.

    Note there is no second/global scale here -- unlike NVFP4 a single E8M0
    factor per block is the whole story.
"""

from __future__ import annotations

import torch

from nvfp4 import dequantize_nvfp4, unpack_fp4  # noqa: F401  (re-exported)

FP4_BLOCK = 32          # DeepSeek FP4: 32 elements along K share one scale
FP8_BLOCK = 128         # DeepSeek FP8: 128x128 weight blocks share one scale


def e8m0_to_float(scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode an E8M0 scale to a real number.

    E8M0 is a bare biased exponent: the stored byte ``b`` means ``2**(b - 127)``
    (``b == 255`` is NaN).  ``torch.float8_e8m0fnu`` knows how to cast itself, so
    prefer that when safetensors handed us the typed tensor; fall back to the
    bit trick when it arrives as raw ``uint8``.
    """
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.to(dtype) - 127.0)
    return scale.to(dtype)


def dequantize_deepseek_fp4(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a DeepSeek FP4 weight.

    :param weight: ``[out, in/2]`` uint8, two packed E2M1 values per byte,
        packed along the reduction (last) dim
    :param scale: ``[out, in/32]`` E8M0, one power-of-two per 32 columns
    """
    if weight.dtype != torch.uint8:
        weight = weight.view(torch.uint8)
    out_dim, packed = weight.shape
    in_dim = packed * 2
    n_groups = scale.shape[-1]
    if n_groups * FP4_BLOCK != in_dim:
        raise ValueError(
            f"scale groups {n_groups} * {FP4_BLOCK} != unpacked columns {in_dim} "
            f"(weight {tuple(weight.shape)}, scale {tuple(scale.shape)})"
        )
    if scale.shape[0] != out_dim:
        raise ValueError(f"scale rows {scale.shape[0]} != weight rows {out_dim}")

    unpacked = unpack_fp4(weight, torch.float32)  # [out, in]
    factor = e8m0_to_float(scale, torch.float32)  # [out, in/32]
    out = unpacked.unflatten(-1, (n_groups, FP4_BLOCK)) * factor.unsqueeze(-1)
    return out.flatten(-2).to(dtype)


def dequantize_deepseek_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a DeepSeek block-scaled FP8 weight.

    :param weight: ``[out, in]`` float8_e4m3fn
    :param scale: ``[ceil(out/128), ceil(in/128)]`` E8M0, one per weight block
    """
    out_dim, in_dim = weight.shape
    b_out, b_in = scale.shape
    if b_out * FP8_BLOCK < out_dim or b_in * FP8_BLOCK < in_dim:
        raise ValueError(
            f"scale {tuple(scale.shape)} too small for weight {tuple(weight.shape)} "
            f"at block size {FP8_BLOCK}"
        )
    factor = e8m0_to_float(scale, torch.float32)
    w = weight.to(torch.float32)
    if out_dim % FP8_BLOCK or in_dim % FP8_BLOCK:
        # Ragged tail: expand the scale per element and crop. DeepSeek's own
        # shapes are all multiples of 128, so this is defensive only.
        factor = factor.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
        return (w * factor[:out_dim, :in_dim]).to(dtype)
    w = w.unflatten(0, (b_out, FP8_BLOCK)).unflatten(-1, (b_in, FP8_BLOCK))
    w = w * factor[:, None, :, None]
    return w.reshape(out_dim, in_dim).to(dtype)


# --------------------------------------------------------------------------
# the other direction: BF16 -> DeepSeek's own encoding
# --------------------------------------------------------------------------
#
# Taken from the checkpoint's own ``inference/kernel.py`` rather than invented.
# ``fp4_quant_kernel`` and ``act_quant_kernel`` both quantize a block as::
#
#     amax = max(|x| over the block, floor)
#     s    = 2 ** ceil(log2(amax / type_max))     # fast_round_scale
#     y    = clamp(x / s, -type_max, type_max)    # then cast, round-to-nearest
#
# with ``type_max`` 6.0 for FP4 and 448.0 for FP8, and ``s`` stored as E8M0.
# The kernels group along the last dim; weights use the same rule, over 32
# columns for FP4 and over a 128x128 tile for FP8.

FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0

# Midpoints between adjacent E2M1 magnitudes. A value landing exactly on one is
# a tie, and the cast rounds those to even -- even meaning an even *code*, i.e.
# magnitudes 0.0, 1.0, 2.0 and 4.0.
_FP4_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_scale_exponent(amax: torch.Tensor, type_max: float) -> torch.Tensor:
    """``ceil(log2(amax / type_max))`` as an integer exponent.

    ``fast_round_scale`` in DeepSeek's kernel, done in float64 so the log of a
    value that is already a power of two lands exactly on the integer rather
    than a hair above it -- which would cost a factor of two on every block
    whose amax is exactly representable.
    """
    ratio = (amax.double() / type_max).clamp_min(torch.finfo(torch.float64).tiny)
    return torch.ceil(torch.log2(ratio))


def _e8m0_bytes(exponent: torch.Tensor) -> torch.Tensor:
    """Encode integer exponents as E8M0: the stored byte is ``e + 127``."""
    return (exponent + 127.0).clamp(0, 254).to(torch.uint8)


def _fp4_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Round non-negative magnitudes to an E2M1 code in 0..7, ties to even."""
    bounds = torch.tensor(_FP4_BOUNDS, dtype=magnitude.dtype, device=magnitude.device)
    # searchsorted counts the bounds strictly below the value, so a value
    # sitting exactly on bound i lands on code i -- the round-down side. That
    # is what ties to even wants at the even bounds; at the odd ones (0.75,
    # 1.75, 3.5) even is the code above, so nudge those up.
    codes = torch.searchsorted(bounds, magnitude.contiguous())
    for i in (1, 3, 5):
        codes = codes + (magnitude == _FP4_BOUNDS[i])
    return codes.clamp_(0, 7).to(torch.uint8)


def quantize_deepseek_fp4(
    weight: torch.Tensor, block: int = FP4_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a BF16 weight as DeepSeek FP4.

    :param weight: ``[out, in]``, ``in`` a multiple of ``block``
    :returns: ``([out, in/2]`` uint8 nibble pairs, ``[out, in/block]`` E8M0
        scales). The low nibble holds the earlier element along the reduction
        dim, matching ``convert.py``'s unpack order, and bit 3 is the sign.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    if in_dim % block:
        raise ValueError(f"columns {in_dim} not a multiple of the FP4 block {block}")

    w = weight.float().unflatten(-1, (in_dim // block, block))
    # 6 * 2**-126 is the kernel's floor: the smallest amax whose scale is still
    # a normal float, so an all-zero block encodes as zeros rather than NaN.
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(FP4_MAX * 2.0**-126)
    exponent = _round_scale_exponent(amax, FP4_MAX)
    scaled = (w.double() / torch.exp2(exponent)).clamp(-FP4_MAX, FP4_MAX)

    codes = _fp4_codes(scaled.abs().float())
    codes |= ((scaled < 0) & (codes > 0)).to(torch.uint8) << 3  # keep zero unsigned
    codes = codes.flatten(-2)

    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    scale = _e8m0_bytes(exponent.squeeze(-1))
    return packed.contiguous(), scale.view(torch.float8_e8m0fnu).contiguous()


def quantize_deepseek_fp8(
    weight: torch.Tensor, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a BF16 weight as DeepSeek block-scaled FP8.

    :param weight: ``[out, in]``; both dims are padded up to ``block`` for the
        amax reduction, as DeepSeek's own shapes are all multiples of 128
    :returns: ``([out, in]`` float8_e4m3fn, ``[ceil(out/block),
        ceil(in/block)]`` E8M0 scales)
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    b_out = -(-out_dim // block)
    b_in = -(-in_dim // block)

    w = weight.float()
    padded = w
    if out_dim % block or in_dim % block:
        padded = torch.zeros(b_out * block, b_in * block, dtype=w.dtype, device=w.device)
        padded[:out_dim, :in_dim] = w

    tiles = padded.unflatten(0, (b_out, block)).unflatten(-1, (b_in, block))
    amax = tiles.abs().amax(dim=(1, 3)).clamp_min(FP8_E4M3_MAX * 2.0**-126)
    exponent = _round_scale_exponent(amax, FP8_E4M3_MAX)

    factor = torch.exp2(exponent).repeat_interleave(block, 0).repeat_interleave(block, 1)
    scaled = (padded.double() / factor).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q = scaled[:out_dim, :in_dim].to(torch.float8_e4m3fn)
    return q.contiguous(), _e8m0_bytes(exponent).view(torch.float8_e8m0fnu).contiguous()


def quantize_linear(
    weight: torch.Tensor, kind: str, scheme: str = "deepseek"
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode one linear's weight, returning ``(weight, scale)``.

    :param kind: ``fp4`` for a routed expert, ``fp8`` for everything else, or
        ``keep`` to pass the tensor through unquantized (which is how both
        families mark "this tensor stays in BF16")
    """
    if scheme != "deepseek":
        raise ValueError(f"no encoder for quantization scheme {scheme!r}")
    if kind == "keep":
        return weight, None
    if kind == "fp4":
        return quantize_deepseek_fp4(weight)
    if kind == "fp8":
        return quantize_deepseek_fp8(weight)
    raise ValueError(f"unknown quantization kind {kind!r}")


def dequantize_linear(
    tensors: dict[str, torch.Tensor],
    prefix: str,
    scheme: str = "nvfp4",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one linear's weight from a dict of its raw tensors.

    Returns the plain ``weight`` untouched when no scale accompanies it, which
    is how both families mark "this tensor was left in BF16".

    :param scheme: ``nvfp4`` or ``deepseek``
    """
    w = tensors[f"{prefix}.weight"]
    if scheme == "nvfp4":
        scale = tensors.get(f"{prefix}.weight_scale")
        if scale is None:
            return w.to(dtype)
        return dequantize_nvfp4(w, scale, tensors[f"{prefix}.weight_scale_2"], dtype)
    if scheme == "deepseek":
        scale = tensors.get(f"{prefix}.scale")
        if scale is None:
            return w.to(dtype)
        # FP4 weights arrive packed two-per-byte, FP8 weights one-per-byte; the
        # element count is what tells them apart.
        if w.dtype in (torch.uint8, torch.int8):
            return dequantize_deepseek_fp4(w, scale, dtype)
        return dequantize_deepseek_fp8(w, scale, dtype)
    raise ValueError(f"unknown quantization scheme {scheme!r}")
