#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Run llm-compressor's REAP modifier end to end on a synthetic DeepSeek-V4.

The point is not accuracy -- the weights are random -- but that the reference
path completes at all, and that each patch in :mod:`dsv4_patches` is load
bearing. The model is tiny (5 layers, 8 experts, hidden 128) and written out
under DeepSeek's *native* tensor names, because that is the form the real
checkpoint ships in and transformers 5.14 already knows how to read it.

Checks:

1. transformers loads native names without any renaming on our side -- the
   experts arrive with real weights, not freshly initialised ones;
2. without the RMSNorm patch a plain forward raises on dtype, and with it the
   model runs in bf16;
3. plain ``REAPPruningModifier`` fails during calibration on the hash-routed
   layers, and ``REAPPruningModifierDSV4`` completes;
4. after pruning: expert count halved in the config, routers sliced, every
   ``tid2eid`` entry inside the new expert range, and the model still runs;
5. ``reap_reference.py``'s retained-set journal ends up describing every MoE
   layer, not only the subgraph that happened to be in flight last;
6. the offloaded, sharded save path -- the one the real run uses, and the one
   transformers 5.14.1 cannot do unpatched -- completes and writes every tensor;
7. saved and reloaded, the pruned model produces the same logits.

Run: ``.venv/bin/python scripts/test_reference_reap.py``  (CPU only, ~1 min)
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Check 3 relies on an out-of-bounds expert index raising a catchable IndexError.
# With a GPU visible it surfaces as a device-side assert instead, which fails the
# check and leaves the CUDA context unusable, taking every check after it down
# too. The model here is 5 layers of hidden 128; CPU is where it belongs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from datasets import Dataset
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv4_patches import REAPPruningModifierDSV4, patch_rmsnorm_dtype  # noqa: E402
from names import to_native  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TOKENIZER_SRC = REPO / "models" / "DeepSeek-V4-Flash-0731"
VOCAB, N_EXPERTS, SEQ, N_HASH, N_LAYERS = 512, 8, 32, 2, 5
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def quiet():
    """llm-compressor is chatty and most of it is progress bars."""
    return contextlib.redirect_stderr(io.StringIO())


# --------------------------------------------------------------------------


def build_config():
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        vocab_size=VOCAB, hidden_size=128, moe_intermediate_size=64,
        num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=1,
        head_dim=32, q_lora_rank=64, o_lora_rank=64, num_experts_per_tok=2,
        n_routed_experts=N_EXPERTS, n_shared_experts=1, num_hash_layers=N_HASH,
        hc_mult=4, index_n_heads=4, index_head_dim=16, index_topk=16,
        sliding_window=16, o_groups=2, max_position_embeddings=128,
        dtype="bfloat16",
    )


def write_native_checkpoint(model, dst: Path) -> None:
    """Save the model the way DeepSeek ships it: native names, one shard."""
    dst.mkdir(parents=True, exist_ok=True)
    out: dict[str, torch.Tensor] = {}
    for name, tensor in {**dict(model.named_parameters()),
                         **dict(model.named_buffers())}.items():
        if name.endswith(("inv_freq", "original_inv_freq")):
            continue  # rotary caches, rebuilt from the config
        # the fused experts parameters are [n_experts, ...]; split them per
        # expert, which is how the real checkpoint stores them
        if name.endswith("mlp.experts.gate_up_proj"):
            block = name[len("model.layers."):].split(".", 1)[0]
            gate, up = tensor.chunk(2, dim=1)
            for e in range(tensor.shape[0]):
                out[f"layers.{block}.ffn.experts.{e}.w1.weight"] = gate[e].clone()
                out[f"layers.{block}.ffn.experts.{e}.w3.weight"] = up[e].clone()
            continue
        if name.endswith("mlp.experts.down_proj"):
            block = name[len("model.layers."):].split(".", 1)[0]
            for e in range(tensor.shape[0]):
                out[f"layers.{block}.ffn.experts.{e}.w2.weight"] = tensor[e].clone()
            continue
        out[to_native(name)] = tensor.detach().clone()

    save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg["architectures"] = ["DeepseekV4ForCausalLM"]
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(TOKENIZER_SRC / f, dst / f)


def make_checkpoint(dst: Path) -> None:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    torch.manual_seed(0)
    model = DeepseekV4ForCausalLM(build_config()).to(torch.bfloat16)
    for layer in model.model.layers:
        gate = layer.mlp.gate
        if hasattr(gate, "tid2eid"):
            gate.tid2eid.copy_(
                torch.randint(0, N_EXPERTS, gate.tid2eid.shape, dtype=torch.long)
            )
    write_native_checkpoint(model, dst)


def load_linearized(src: Path):
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from transformers import AutoModelForCausalLM

    with quiet(), load_quantizable_moe(AutoModelForCausalLM):
        return AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)


def calibration(n: int = 8) -> tuple[Dataset, torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (n, SEQ), generator=g)
    ds = Dataset.from_dict(
        {"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()}
    )
    return ds, ids


def run_reap(model, tokenizer, ds, recipe):
    from llmcompressor import oneshot

    with quiet():
        oneshot(
            model=model, dataset=ds, processor=tokenizer, recipe=recipe,
            max_seq_length=SEQ, num_calibration_samples=len(ds),
            moe_calibrate_all_experts=False, pipeline="sequential",
        )


# --------------------------------------------------------------------------


def test_native_names(src: Path) -> None:
    print("\n== transformers reads DeepSeek's native names ==")
    from llmcompressor.modeling.moe.linear_experts import LinearExperts2D

    model = load_linearized(src)
    experts = model.model.layers[4].mlp.experts
    check("experts linearized to LinearExperts2D", isinstance(experts, LinearExperts2D))
    norms = [float(experts[i].gate_proj.weight.detach().float().norm()) for i in range(3)]
    check("expert weights actually loaded", all(n > 0 for n in norms),
          f"first three gate_proj norms {[round(n, 3) for n in norms]}")
    check("hash router on the first layers, top-k router after",
          type(model.model.layers[0].mlp.gate).__name__ == "DeepseekV4HashRouter"
          and type(model.model.layers[4].mlp.gate).__name__ == "DeepseekV4TopKRouter")
    return model


def test_bf16_forward(src: Path, ids: torch.Tensor) -> None:
    print("\n== bf16 forward needs the RMSNorm patch ==")
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as modeling

    original = modeling.DeepseekV4RMSNorm.forward
    patched = getattr(original, "_dsv4_dtype_patched", False)
    if patched:  # restore the stock implementation to show it failing
        def stock(self, hidden_states):
            input_dtype = hidden_states.dtype
            h = hidden_states.to(torch.float32)
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
            return self.weight * h.to(input_dtype)

        modeling.DeepseekV4RMSNorm.forward = stock

    model = load_linearized(src)
    try:
        with torch.no_grad():
            model(input_ids=ids[:1])
        check("stock transformers fails to forward in bf16", False, "it succeeded")
    except RuntimeError as e:
        check("stock transformers fails to forward in bf16", "dtype" in str(e),
              str(e)[:60])

    modeling.DeepseekV4RMSNorm.forward = original
    patch_rmsnorm_dtype()
    model = load_linearized(src)
    with torch.no_grad():
        out = model(input_ids=ids[:1])
    check("patched, the model forwards in bf16", out.logits.dtype == torch.bfloat16)


def test_stock_reap_fails(src: Path, tokenizer, ds) -> None:
    print("\n== stock REAP breaks the hash-routed layers ==")
    from llmcompressor.modifiers.pruning import REAPPruningModifier

    model = load_linearized(src)
    try:
        run_reap(model, tokenizer, ds, REAPPruningModifier(sparsity=0.5))
        check("stock REAP fails on tid2eid", False, "the run completed")
    except (RuntimeError, IndexError) as e:
        check("stock REAP fails on tid2eid", "out of bounds" in str(e), str(e)[:70])


def test_patched_reap(src: Path, tokenizer, ds, ids: torch.Tensor):
    print("\n== patched REAP completes ==")
    model = load_linearized(src)
    run_reap(model, tokenizer, ds, REAPPruningModifierDSV4(sparsity=0.5))

    kept = N_EXPERTS // 2
    check("config expert count halved", model.config.n_routed_experts == kept,
          f"n_routed_experts={model.config.n_routed_experts}")

    shapes, tid_max = [], []
    for layer in model.model.layers:
        gate = layer.mlp.gate
        shapes.append(gate.weight.shape[0])
        if hasattr(gate, "tid2eid"):
            tid_max.append(int(gate.tid2eid.max()))
    check("every router sliced to the kept experts", set(shapes) == {kept},
          f"router rows {sorted(set(shapes))}")
    check("tid2eid points inside the new expert range",
          bool(tid_max) and max(tid_max) < kept,
          f"max id {max(tid_max) if tid_max else '-'} < {kept}")

    with torch.no_grad():
        out = model(input_ids=ids[:1])
    check("pruned model still forwards", torch.isfinite(out.logits).all().item())
    return model, out.logits


def test_journal_accumulates(src: Path, tokenizer, ds, journal: Path) -> None:
    """The journal is the crash insurance; it has to describe the whole run.

    ``_saliency_trackers`` only ever holds the subgraph being calibrated, so a
    journal that writes each epoch's snapshot on its own ends up describing the
    last layer and nothing else. That failure is invisible until the run dies --
    which is the one moment the file matters -- so it is checked here.
    """
    print("\n== the retained-set journal accumulates every layer ==")
    from reap_reference import make_modifier

    model = load_linearized(src)
    run_reap(model, tokenizer, ds, make_modifier(0.5, "balanced", journal))
    recorded = json.loads(journal.read_text())

    kept = N_EXPERTS // 2
    check("every MoE layer is in the journal, not just the last",
          len(recorded) == N_LAYERS, f"{len(recorded)} of {N_LAYERS} layer(s)")
    check("each entry keeps the right number of distinct experts",
          all(len(set(v)) == kept and len(v) == kept for v in recorded.values()),
          f"sizes {sorted({len(v) for v in recorded.values()})}, want {kept}")
    check("no journalled expert id is out of range",
          all(0 <= e < N_EXPERTS for v in recorded.values() for e in v),
          f"max id {max((e for v in recorded.values() for e in v), default='-')}")


def test_per_source_saliency(src: Path, tokenizer, ds, out: Path) -> None:
    """Splitting the saliency by source must not change what it says in total.

    The whole point of the split is that a re-weighted mixture can be computed
    from it exactly, and that rests on one property: ``sum_saliency`` and
    ``count`` are sums over disjoint sets of tokens, so the per-source parts add
    back up to the unsplit accumulators. If they ever do not -- a forward that
    carried more than one sample, a delta taken against the wrong snapshot --
    every re-weighting built on the file is quietly wrong, and nothing
    downstream would say so.

    Checked here rather than trusted, on the same synthetic model the rest of
    these checks use.
    """
    print("\n== saliency splits by source without changing the totals ==")
    from reap_reference import make_modifier

    labels = ["ja", "en", "code"][: len(ds)] * len(ds)
    labels = labels[: len(ds)]
    model = load_linearized(src)
    recipe = make_modifier(0.5, "balanced", None, out, {}, labels, SEQ)
    run_reap(model, tokenizer, ds, recipe)

    doc = json.loads(out.read_text())["layers"]
    check("every layer carries a per-source split",
          all("by_source" in v for v in doc.values()),
          f"{sum('by_source' in v for v in doc.values())} of {len(doc)}")

    worst_sum, worst_count, seen = 0.0, 0.0, set()
    for layer in doc.values():
        split = layer.get("by_source", {})
        seen |= set(split)
        for field, ref in (("sum_saliency", layer["sum_saliency"]), ("count", layer["count"])):
            total = [sum(part[field][j] for part in split.values()) for j in range(len(ref))]
            err = max((abs(a - b) for a, b in zip(total, ref)), default=0.0)
            if field == "sum_saliency":
                worst_sum = max(worst_sum, err)
            else:
                worst_count = max(worst_count, err)

    check("per-source sum_saliency adds back to the unsplit total",
          worst_sum < 1e-9, f"worst absolute error {worst_sum:.3e}")
    check("per-source token counts add back exactly",
          worst_count == 0.0, f"worst absolute error {worst_count:.3e}")
    check("the sources present are the ones that were labelled",
          seen == set(labels), f"{sorted(seen)} vs {sorted(set(labels))}")

    # And the identity the re-weighting uses: equal weights reproduce the mean.
    layer = next(iter(doc.values()))
    split = layer["by_source"]
    n = len(layer["mean"])
    num = [sum(p["sum_saliency"][j] for p in split.values()) for j in range(n)]
    den = [sum(p["count"][j] for p in split.values()) for j in range(n)]
    remixed = [num[j] / den[j] if den[j] else 0.0 for j in range(n)]
    err = max((abs(a - b) for a, b in zip(remixed, layer["mean"])), default=0.0)
    check("an equal-weight remix reproduces the recorded mean",
          err < 1e-9, f"worst absolute error {err:.3e}")


def test_offloaded_save(model, dst: Path) -> None:
    """The real run saves an offloaded, sharded model -- check that path here.

    transformers 5.14.1 cannot do it unpatched: ``save_pretrained`` reverts weight
    conversions shard by shard and then runs

        weight_map.update({k: os.path.basename(shard_file)} for k in ...)

    which hands dict.update() a generator of dicts and raises ValueError for every
    non-empty shard, masked by a bare ``except Exception`` into a RuntimeError
    blaming ``max_shard_size``. No shard size avoids it. Discovering that costs an
    hour of calibration when it is discovered at the save step of the real run, so
    it is discovered here instead, in seconds.

    The offloaded branch is chosen by ``hf_device_map`` alone, so setting the
    attribute is enough -- no need to actually offload this tiny model.
    """
    print("\n== an offloaded, sharded save completes ==")
    from safetensors import safe_open

    def tensor_names(d: Path) -> set[str]:
        out: set[str] = set()
        for f in sorted(d.glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as handle:
                out |= set(handle.keys())
        return out

    plain_dst = dst.with_name(dst.name + "-plain")
    with quiet():
        model.save_pretrained(plain_dst, max_shard_size="200KB")
    expected = tensor_names(plain_dst)

    model.hf_device_map = {"model.embed_tokens": "cpu", "model.layers.0": "disk"}
    try:
        with quiet():
            model.save_pretrained(dst, max_shard_size="200KB")
        failure = None
    except Exception as e:  # noqa: BLE001 -- the point is to report it, not handle it
        failure = f"{type(e).__name__}: {str(e)[:80]}"
    finally:
        # save_pretrained clears the attribute itself on the way out, so pop it
        # rather than deleting it -- the model is reused by the next check.
        model.__dict__.pop("hf_device_map", None)

    check("offloaded sharded save completes", failure is None, failure or "")
    if failure is None:
        got = tensor_names(dst)
        check("offloaded save writes the same tensors as a plain one",
              got == expected,
              f"{len(got)} tensors, {len(expected ^ got)} differing name(s)")
        check("the index covers every written tensor",
              json.loads((dst / "model.safetensors.index.json").read_text())
              ["weight_map"].keys() == got,
              f"index lists {len(json.loads((dst / 'model.safetensors.index.json').read_text())['weight_map'])}")


def test_save_reload(model, logits, ids: torch.Tensor, dst: Path) -> None:
    print("\n== the pruned model survives a save/reload ==")
    with quiet():
        model.save_pretrained(dst)
    # Reload the way it was written: save_pretrained keeps the linearized 2D
    # expert names, which the plain loader does not expect.
    reloaded = load_linearized(dst)
    with torch.no_grad():
        again = reloaded(input_ids=ids[:1]).logits
    check("logits match after a round trip through disk",
          torch.equal(logits, again),
          f"max |diff| {(logits.float() - again.float()).abs().max().item():.6g}")


def main() -> int:
    from transformers import AutoTokenizer

    if not (TOKENIZER_SRC / "tokenizer.json").exists():
        print(f"needs a tokenizer at {TOKENIZER_SRC}")
        return 2

    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "tiny", Path(td) / "pruned"
        make_checkpoint(src)
        tokenizer = AutoTokenizer.from_pretrained(src)
        ds, ids = calibration()

        test_native_names(src)
        test_bf16_forward(src, ids)
        test_stock_reap_fails(src, tokenizer, ds)
        model, logits = test_patched_reap(src, tokenizer, ds, ids)
        test_journal_accumulates(src, tokenizer, ds, Path(td) / "retained-sets.json")
        test_per_source_saliency(src, tokenizer, ds, Path(td) / "saliency.json")
        test_offloaded_save(model, Path(td) / "offloaded-save")
        test_save_reload(model, logits, ids, dst)

    print()
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
