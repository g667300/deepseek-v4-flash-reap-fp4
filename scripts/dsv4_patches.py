#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""What DeepSeek-V4 needs before llm-compressor's REAP modifier will run on it.

Two independent problems, both found by running the reference path on a
synthetic DeepSeek-V4 (``test_reference_reap.py`` reproduces each).

**1. The model cannot forward in bfloat16.**
``DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict`` pins every norm --
``input_layernorm``, ``post_attention_layernorm``, ``q_a_norm``, ``kv_norm``,
``norm`` -- to fp32 whatever dtype the checkpoint is loaded in. But
``DeepseekV4RMSNorm.forward`` ends on::

    return self.weight * hidden_states.to(input_dtype)

An fp32 weight times a bf16 activation promotes back to fp32, so the norm
hands fp32 to the next Linear, whose weight is bf16, and torch raises
``expected m1 and m2 to have the same dtype``. Loading in fp32 avoids it and
doubles the checkpoint to 1.1 TB, which is why this patch exists instead:
compute in fp32 as the model intends, hand back the input dtype.

**2. REAP breaks the hash-routed layers.**
DeepSeek-V4's first ``num_hash_layers`` (3) blocks route through
``DeepseekV4HashRouter``, which picks experts from a frozen ``tid2eid`` table
-- a buffer whose *values* are expert ids. The reference modifier's
``_prune_router`` shrinks ``weight``, ``bias`` and ``e_score_correction_bias``
and knows nothing about ``tid2eid``, so after a layer is pruned the table
still names experts that no longer exist and the next forward raises
``index N is out of bounds``. It fails during calibration, not at the end.

Leaving the hash layers out via REAP's ``ignore`` is not an escape:
``n_routed_experts`` is one value for the whole model, so those three layers
have to end up at the same expert count as the rest or the checkpoint will not
load. They have to be pruned, which means the table has to be rewritten.
"""

from __future__ import annotations

import torch

__all__ = ["patch_rmsnorm_dtype", "REAPPruningModifierDSV4", "build_value_map"]


# --------------------------------------------------------------------------
# 1. bf16 forward
# --------------------------------------------------------------------------


def patch_rmsnorm_dtype() -> None:
    """Make ``DeepseekV4RMSNorm`` return its input dtype. Idempotent."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling

    if getattr(modeling.DeepseekV4RMSNorm.forward, "_dsv4_dtype_patched", False):
        return

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight.to(torch.float32) * h).to(input_dtype)

    forward._dsv4_dtype_patched = True
    modeling.DeepseekV4RMSNorm.forward = forward


# --------------------------------------------------------------------------
# 2. hash-routed layers
# --------------------------------------------------------------------------


def build_value_map(
    gate_weight: torch.Tensor, keep: list[int], mode: str = "balanced"
) -> list[int]:
    """Map every original expert id to a surviving expert's new index.

    Survivors map to their own new position. A dropped expert's tokens have to
    go somewhere -- ``tid2eid`` must name exactly ``top_k`` experts for every
    token id -- so they are merged into a survivor. That is a merge, not a
    prune, and REAP's own argument is against merging; it is only defensible
    because it touches the three hash-routed layers, which have no score-based
    routing that pruning could simply renormalise.

    ``nearest`` sends each dropped expert to its most similar survivor,
    independently. That maximises similarity but concentrates load, because
    popular survivors win many times over.

    ``balanced`` (the default) caps how many dropped experts any one survivor
    may absorb at ``ceil(n_dropped / n_kept)`` -- exactly one each at 50%
    sparsity -- and fills the assignment greedily from the most similar pair
    down. Hash routing is content-independent, so an even spread of token ids
    is what keeps every expert's share of the vocabulary comparable to what it
    had before pruning.
    """
    n_orig = gate_weight.shape[0]
    new_of = {old: i for i, old in enumerate(keep)}
    dropped = [e for e in range(n_orig) if e not in new_of]
    if not dropped:
        return [new_of[e] for e in range(n_orig)]

    w = torch.nn.functional.normalize(gate_weight.float(), dim=-1)
    keep_idx = torch.tensor(keep, dtype=torch.long)
    sim = w @ w.index_select(0, keep_idx).T          # [n_orig, n_keep]

    if mode == "nearest":
        nearest = sim.argmax(dim=1)                  # position within `keep`
        return [new_of.get(e, int(nearest[e])) for e in range(n_orig)]
    if mode != "balanced":
        raise ValueError(f"unknown remap mode {mode!r}")

    capacity = -(-len(dropped) // len(keep))         # ceil
    load = [0] * len(keep)
    assigned: dict[int, int] = {}
    pairs = sorted(
        ((float(sim[e, j]), e, j) for e in dropped for j in range(len(keep))),
        reverse=True,
    )
    for _score, e, j in pairs:
        if e in assigned or load[j] >= capacity:
            continue
        assigned[e] = j
        load[j] += 1
        if len(assigned) == len(dropped):
            break
    for e in dropped:  # only reachable if capacity*len(keep) < len(dropped)
        if e not in assigned:
            j = min(range(len(keep)), key=lambda k: load[k])
            assigned[e] = j
            load[j] += 1
    # not dict.get(e, assigned[e]) -- that evaluates assigned[e] for survivors too
    return [new_of[e] if e in new_of else assigned[e] for e in range(n_orig)]


def _import_reap():
    from llmcompressor.modifiers.pruning import REAPPruningModifier

    return REAPPruningModifier


class REAPPruningModifierDSV4(_import_reap()):  # type: ignore[misc]
    """REAP, plus the ``tid2eid`` rewrite the hash-routed layers need.

    The remap has to be computed from the router rows *before* they are
    sliced -- similarity between a dropped expert and its survivors is what
    decides where its token ids go, and after the slice the dropped rows are
    gone. So: compute the mapping for every hash layer about to be pruned,
    let the reference implementation do the pruning untouched, then apply the
    mapping to the table.

    ``hash_remap`` selects the assignment strategy; see :func:`build_value_map`.
    """

    hash_remap: str = "balanced"

    def on_sequential_epoch_end(self, state, event, **kwargs):
        model = state.model

        pending: dict[str, torch.Tensor] = {}
        for layer_name, tracker in self._saliency_trackers.items():
            if tracker.total_count <= 0:
                continue
            gate = model.get_submodule(layer_name).gate
            if not hasattr(gate, "tid2eid"):
                continue
            retained = tracker.compute_retained_experts(
                self._n_experts_to_drop,
                self._n_experts_to_drop_per_group,
                self._moe_attrs,
            )
            pending[layer_name] = torch.tensor(
                build_value_map(
                    gate.weight.detach().float().cpu(), retained, self.hash_remap
                ),
                dtype=torch.long,
            )

        super().on_sequential_epoch_end(state, event, **kwargs)

        for layer_name, value_map in pending.items():
            gate = model.get_submodule(layer_name).gate
            table = gate.tid2eid
            remapped = value_map[table.detach().cpu().long()]
            gate.tid2eid = remapped.to(device=table.device, dtype=table.dtype)
