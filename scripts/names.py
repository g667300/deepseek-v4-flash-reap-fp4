#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""DeepSeek-V4 tensor names: native <-> transformers.

Only one direction needs writing. transformers 5.14 ships a 42-rule checkpoint
conversion mapping for ``deepseek_v4`` (``get_checkpoint_conversion_mapping``),
so it reads DeepSeek's native names -- ``layers.{L}.attn.wq_a.weight``,
``embed.weight``, ``ffn.gate.bias`` -- with no help from us. The dequantization
pass therefore leaves every name alone and only replaces FP4/FP8 pairs with
BF16 tensors.

Coming back is the direction that needs a table. ``save_pretrained`` writes
transformers' *modern* names (``model.layers.{L}.mlp.experts.{E}.gate_proj.
weight``): the native spellings are reached through what transformers treats as
a legacy mapping and it does not write them back. vLLM reads the checkpoint
DeepSeek published, so the output has to carry native names, and
:func:`to_native` is what puts them there.

The table below is the inverse of transformers' mapping, cross-checked against
the real checkpoint's index by ``test_names.py`` -- every name the model
expects must map onto a name the checkpoint has, and nothing may be left over.
"""

from __future__ import annotations

import re

__all__ = ["to_native", "to_hf", "EXPERT_PROJECTIONS"]

# Which DeepSeek ``w{1,2,3}`` each HF projection is. w1/w3 are the SwiGLU gate
# and up branches, w2 the down projection -- the same on the shared expert and
# on every routed one.
EXPERT_PROJECTIONS = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

# Names outside the decoder stack.
_TOP_LEVEL = (
    ("model.embed_tokens.weight", "embed.weight"),
    ("lm_head.weight", "head.weight"),
    ("model.norm.weight", "norm.weight"),
    ("model.hc_head.hc_fn", "hc_head_fn"),
    ("model.hc_head.hc_base", "hc_head_base"),
    ("model.hc_head.hc_scale", "hc_head_scale"),
)

# Everything under ``model.layers.{L}.``, applied to the part after the index.
# Order matters: the first match wins, so the deeper attention names -- which
# are prefixes of the shallower ones -- come first.
_IN_LAYER = (
    ("input_layernorm.", "attn_norm."),
    ("post_attention_layernorm.", "ffn_norm."),
    ("attn_hc.fn", "hc_attn_fn"),
    ("attn_hc.base", "hc_attn_base"),
    ("attn_hc.scale", "hc_attn_scale"),
    ("ffn_hc.fn", "hc_ffn_fn"),
    ("ffn_hc.base", "hc_ffn_base"),
    ("ffn_hc.scale", "hc_ffn_scale"),
    # attention -- the indexer sits under the compressor in transformers but
    # beside it in the checkpoint, so these have to precede the plain
    # compressor names
    ("self_attn.compressor.indexer.scorer.weights_proj.", "attn.indexer.weights_proj."),
    ("self_attn.compressor.indexer.kv_norm.", "attn.indexer.compressor.norm."),
    ("self_attn.compressor.indexer.position_bias", "attn.indexer.compressor.ape"),
    ("self_attn.compressor.indexer.kv_proj.", "attn.indexer.compressor.wkv."),
    ("self_attn.compressor.indexer.gate_proj.", "attn.indexer.compressor.wgate."),
    ("self_attn.compressor.indexer.q_b_proj.", "attn.indexer.wq_b."),
    ("self_attn.compressor.kv_norm.", "attn.compressor.norm."),
    ("self_attn.compressor.position_bias", "attn.compressor.ape"),
    ("self_attn.compressor.kv_proj.", "attn.compressor.wkv."),
    ("self_attn.compressor.gate_proj.", "attn.compressor.wgate."),
    ("self_attn.sinks", "attn.attn_sink"),
    ("self_attn.q_a_norm.", "attn.q_norm."),
    ("self_attn.q_a_proj.", "attn.wq_a."),
    ("self_attn.q_b_proj.", "attn.wq_b."),
    ("self_attn.kv_norm.", "attn.kv_norm."),
    ("self_attn.kv_proj.", "attn.wkv."),
    ("self_attn.o_a_proj.", "attn.wo_a."),
    ("self_attn.o_b_proj.", "attn.wo_b."),
    # MoE
    ("mlp.gate.e_score_correction_bias", "ffn.gate.bias"),
    ("mlp.gate.tid2eid", "ffn.gate.tid2eid"),
    ("mlp.gate.weight", "ffn.gate.weight"),
    ("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1."),
    ("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2."),
    ("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3."),
)

_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.+)$")
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def to_native(name: str) -> str:
    """Map one transformers tensor name to the name DeepSeek ships it under.

    :raises KeyError: if the name is not one this model has. Silence here would
        mean writing a checkpoint with a tensor vLLM will not look for, so it
        is deliberately loud.
    """
    for hf, native in _TOP_LEVEL:
        if name == hf:
            return native

    match = _LAYER_RE.match(name)
    if match is None:
        raise KeyError(f"no native name known for {name!r}")
    layer, rest = match.group(1), match.group(2)

    expert = _EXPERT_RE.match(rest)
    if expert is not None:
        index, projection, suffix = expert.groups()
        return f"layers.{layer}.ffn.experts.{index}.{EXPERT_PROJECTIONS[projection]}.{suffix}"

    for hf, native in _IN_LAYER:
        if rest.startswith(hf):
            return f"layers.{layer}.{native}{rest[len(hf):]}"
    raise KeyError(f"no native name known for {name!r}")


_TO_HF_TOP = {native: hf for hf, native in _TOP_LEVEL}
_TO_HF_LAYER = tuple((native, hf) for hf, native in _IN_LAYER)
_NATIVE_EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(.+)$")
_NATIVE_LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
_HF_PROJECTIONS = {v: k for k, v in EXPERT_PROJECTIONS.items()}


def to_hf(name: str) -> str:
    """The inverse of :func:`to_native`, for tests and for reading a source shard."""
    if name in _TO_HF_TOP:
        return _TO_HF_TOP[name]

    expert = _NATIVE_EXPERT_RE.match(name)
    if expert is not None:
        layer, index, projection, suffix = expert.groups()
        return (f"model.layers.{layer}.mlp.experts.{index}."
                f"{_HF_PROJECTIONS[projection]}.{suffix}")

    match = _NATIVE_LAYER_RE.match(name)
    if match is None:
        raise KeyError(f"no transformers name known for {name!r}")
    layer, rest = match.group(1), match.group(2)

    # Longest native prefix first: `attn.compressor.wkv.` must beat `attn.wkv.`
    # would-be matches, and `ffn.gate.bias` must beat nothing at all.
    for native, hf in sorted(_TO_HF_LAYER, key=lambda pair: -len(pair[0])):
        if rest.startswith(native):
            return f"model.layers.{layer}.{hf}{rest[len(native):]}"
    raise KeyError(f"no transformers name known for {name!r}")
