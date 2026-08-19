#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Hold the name table against the real checkpoint and the real model.

A hand-written rename table drifts silently: a name that maps to something the
checkpoint does not use produces a tensor vLLM never looks for, and the model
loads with that weight freshly initialised instead. So neither side of the
mapping is trusted here. Both ends are taken from something authoritative --
the parameter names ``DeepseekV4ForCausalLM`` declares on the meta device, and
the tensor names in the published checkpoint's index -- and the table has to
carry each onto the other with nothing left over.

Run: ``.venv/bin/python scripts/test_names.py [--ckpt DIR]``  (CPU only)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from names import to_hf, to_native  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "models" / "DeepSeek-V4-Flash-0731"
FAILED: list[str] = []

# Never in a checkpoint: rebuilt from the config at load time.
_DERIVED = re.compile(r"(inv_freq|original_inv_freq)$")
# The fused 3D expert parameters. The checkpoint stores experts one at a time,
# and llm-compressor's linearize path splits them back out, so these two names
# only ever exist in memory.
_FUSED = re.compile(r"mlp\.experts\.(gate_up_proj|down_proj)$")


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def model_parameter_names(ckpt: Path) -> set[str]:
    """Every name DeepseekV4ForCausalLM declares, without loading any weights."""
    from accelerate import init_empty_weights
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    config = DeepseekV4Config(**json.loads((ckpt / "config.json").read_text()))
    with init_empty_weights():
        model = DeepseekV4ForCausalLM(config)
    names = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    return {n for n in names if not _DERIVED.search(n)}


def expand_fused(names: set[str], n_experts: int) -> set[str]:
    """Replace the fused expert parameters with their per-expert 2D names."""
    out = set()
    for name in names:
        if not _FUSED.search(name):
            out.add(name)
            continue
        prefix = name.rsplit(".", 1)[0]
        projections = (
            ("gate_proj", "up_proj") if name.endswith("gate_up_proj") else ("down_proj",)
        )
        for expert in range(n_experts):
            for projection in projections:
                out.add(f"{prefix}.{expert}.{projection}.weight")
    return out


def test_model_to_checkpoint(ckpt: Path) -> None:
    print("\n== every name the model wants exists in the checkpoint ==")
    config = json.loads((ckpt / "config.json").read_text())
    wanted = expand_fused(model_parameter_names(ckpt), config["n_routed_experts"])
    have = set(json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"])

    unmapped, missing = [], []
    for name in sorted(wanted):
        try:
            native = to_native(name)
        except KeyError:
            unmapped.append(name)
            continue
        if native not in have:
            missing.append(f"{name} -> {native}")

    check("every parameter maps to a native name", not unmapped,
          f"{len(unmapped)} unmapped, e.g. {unmapped[:3]}")
    check("every mapped name is in the checkpoint", not missing,
          f"{len(missing)} missing, e.g. {missing[:3]}")
    print(f"  ({len(wanted):,} parameter names checked)")


def test_checkpoint_to_model(ckpt: Path) -> None:
    print("\n== every tensor in the checkpoint maps back ==")
    have = set(json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"])
    # `.scale` companions disappear with dequantization; mtp.* is not modelled
    # by transformers at all (_keys_to_ignore_on_load_unexpected drops it), so
    # it is carried separately and is not this table's business.
    weights = {
        n for n in have
        if not n.endswith(".scale") and not n.startswith("mtp.")
    }

    unmapped, mismatched = [], []
    for name in sorted(weights):
        try:
            hf = to_hf(name)
        except KeyError:
            unmapped.append(name)
            continue
        if to_native(hf) != name:
            mismatched.append(f"{name} -> {hf} -> {to_native(hf)}")

    check("every checkpoint tensor maps to a transformers name", not unmapped,
          f"{len(unmapped)} unmapped, e.g. {unmapped[:3]}")
    check("the mapping round-trips", not mismatched,
          f"{len(mismatched)} broken, e.g. {mismatched[:3]}")
    print(f"  ({len(weights):,} checkpoint tensors checked)")


def test_rejects_unknown() -> None:
    print("\n== unknown names are refused, not guessed ==")
    for name in ("model.layers.0.mlp.nonsense.weight", "who.knows", "lm_head.bias"):
        try:
            to_native(name)
            check(f"refuses {name!r}", False, "it returned a name")
        except KeyError:
            check(f"refuses {name!r}", True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    args = ap.parse_args()

    test_rejects_unknown()
    if (args.ckpt / "model.safetensors.index.json").exists():
        test_model_to_checkpoint(args.ckpt)
        test_checkpoint_to_model(args.ckpt)
    else:
        print(f"\n== real checkpoint ==\n  skipped, nothing at {args.ckpt}")

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
