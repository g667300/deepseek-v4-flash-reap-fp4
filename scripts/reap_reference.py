#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Run llm-compressor's REAP modifier over the BF16 DeepSeek-V4 checkpoint.

This is step 2 of the pipeline in the README: 568.7 GB of BF16 in, 291.6 GB of
pruned BF16 out. Steps 1 and 3 are format conversions; this is the only step
that decides anything.

Three things this has to get right, none of which the reference path does on
its own -- see :mod:`dsv4_patches` for why each exists:

* ``patch_rmsnorm_dtype()`` before any forward, or the model cannot run in
  bf16 at all;
* ``REAPPruningModifierDSV4`` rather than the stock modifier, or the run dies
  mid-calibration on the hash-routed layers' ``tid2eid`` table;
* the model has to be loaded **offloaded**. 568.7 GB does not fit in host RAM,
  so ``device_map="auto_offload"`` keeps what fits on the CPU and puts the rest
  on disk, and the sequential pipeline onloads one subgraph at a time.

**Retained sets are written as they are computed**, to ``--retained-out``, and
rewritten after every subgraph. That is deliberate: this run is long, the
machine it runs on has been crashing under memory pressure (see
``docs/memory-faults.md``), and the retained sets are the only part of the run
that is expensive to recompute and cheap to store. They are also what
``carry_mtp.py`` needs later. If the box hangs at layer 50, the file still says
what the first 49 decided.

**The saliency behind them goes to ``--saliency-out``.** The retained set is a
bottom-k removal over per-expert saliency, so storing only the set answers one
sparsity and discards the measurement that answers all of them. A few hundred KB
buys the ability to ask what 160 or 176 experts would have kept without spending
another hour of calibration to find out.

Usage::

    reap_reference.py --model artifacts/dsv4-bf16 --tokens artifacts/calib.pt \\
        --sparsity 0.5 --out artifacts/dsv4-reap50-bf16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsv4_patches import REAPPruningModifierDSV4, patch_rmsnorm_dtype  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    """Identify the calibration tokens by content, not by the name they sit under."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# the modifier, plus a retained-set journal
# --------------------------------------------------------------------------


def make_modifier(sparsity: float, hash_remap: str, journal: Path | None,
                  saliency_out: Path | None = None, provenance: dict | None = None,
                  sample_sources: list[str] | None = None, seq_len: int | None = None):
    """REAP for DeepSeek-V4, writing out what it decided *and what it decided from*.

    ``compute_retained_experts`` is a pure selection over accumulated saliency
    (llm-compressor ``modifiers/pruning/reap/utils.py``), so calling it here as
    well as inside the modifier costs a topk and changes nothing.

    The saliency itself is worth keeping and costs nothing to keep. It is a public
    property on the tracker -- ``S_j = mean(g_j * ||f_j||_2)`` per expert -- and
    the retained set is only its bottom-k removed, so 43 x 256 float64 is the
    difference between "we can see what any other sparsity would keep" and "run
    the hour of calibration again to find out". Writing only the selection throws
    the measurement away and keeps the conclusion.

    The *raw* accumulators go out with it. A mean says nothing about how much it
    rests on: routing is sparse, so an expert the calibration set barely reached
    has a saliency computed from a handful of tokens, and one that has never been
    routed to at all has none. ``count`` is what distinguishes "scored low" from
    "never measured", and it is the difference between a ranking you can push to
    a higher sparsity and one you cannot.

    ``_saliency_trackers`` holds only the subgraph currently being calibrated --
    the trackers for layers already pruned are gone by the next epoch end. So both
    files have to *accumulate*: writing each epoch's snapshot on its own would
    leave a file describing one layer, which is exactly the file that is useless
    after a crash.

    ``sample_sources`` splits the accumulators by which calibration source each
    sample came from, and that turns a one-mixture measurement into every
    mixture. ``sum_saliency`` is a plain sum of ``g_j * ||f_j||_2`` over routed
    tokens and ``count`` is how many there were, so both are additive across
    sources and any re-weighting is exact arithmetic on what is already
    measured::

        mean_w(j) = sum_s w_s * sum_s(j) / sum_s w_s * count_s(j)

    Without it, asking "what if the calibration were weighted towards code"
    costs another hour per question; with it, it costs a topk. The split is
    taken as the *delta* across ``_experts_block_hook`` rather than by
    recomputing REAP's contribution here, so it stays correct if the upstream
    formula changes, and it needs one sample per forward -- checked, not
    assumed, on the first call.
    """

    recorded: dict[str, list[int]] = {}
    saliency: dict[str, dict] = {}
    # layer -> source -> {"sum_saliency": Tensor, "count": Tensor}
    by_source: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    hook_calls: dict[str, int] = {}
    attribution = {"on": sample_sources is not None, "warned": False}

    def write(path: Path, payload) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
        os.replace(tmp, path)  # atomic; a crash mid-write cannot truncate it

    def accumulators(tracker, layer_name: str) -> dict:
        out = {"mean": tracker.mean_saliency.tolist()}
        for field in ("sum_saliency", "count"):
            value = getattr(tracker, field, None)
            if value is not None:
                out[field] = value.to("cpu", torch.float64).tolist()
        split = by_source.get(layer_name)
        if split:
            out["by_source"] = {
                src: {k: v.to("cpu", torch.float64).tolist() for k, v in acc.items()}
                for src, acc in sorted(split.items())
            }
        return out

    class REAPPruningModifierDSV4Journaled(REAPPruningModifierDSV4):
        def _experts_block_hook(self, layer_name, module, args, output):
            if not attribution["on"]:
                return super()._experts_block_hook(layer_name, module, args, output)

            tracker = self._saliency_trackers.get(layer_name)
            if tracker is None:
                return super()._experts_block_hook(layer_name, module, args, output)

            # One forward must be one calibration sample, or the call index is
            # not a sample index and every label would be wrong. Check it on the
            # first call and fall back to the unsplit measurement rather than
            # writing a confidently mislabelled one.
            rows = int(args[1].shape[0]) if hasattr(args[1], "shape") else -1
            if seq_len is not None and rows not in (seq_len, -1) and not attribution["warned"]:
                attribution["on"] = False
                attribution["warned"] = True
                log(f"  WARNING: {rows} routed rows per forward against seq_len "
                    f"{seq_len} -- calibration is not one sample per forward, so "
                    "per-source saliency is disabled for this run")
                return super()._experts_block_hook(layer_name, module, args, output)

            before = None
            if tracker.sum_saliency is not None:
                before = (tracker.sum_saliency.clone(), tracker.count.clone())

            out = super()._experts_block_hook(layer_name, module, args, output)

            index = hook_calls.get(layer_name, 0)
            hook_calls[layer_name] = index + 1
            if tracker.sum_saliency is None or index >= len(sample_sources):
                return out
            if before is None:
                delta_s, delta_c = tracker.sum_saliency, tracker.count
            else:
                delta_s = tracker.sum_saliency - before[0]
                delta_c = tracker.count - before[1]
            acc = by_source.setdefault(layer_name, {}).setdefault(
                sample_sources[index],
                {"sum_saliency": torch.zeros_like(delta_s, device="cpu"),
                 "count": torch.zeros_like(delta_c, device="cpu")},
            )
            acc["sum_saliency"] += delta_s.to("cpu", torch.float64)
            acc["count"] += delta_c.to("cpu", torch.float64)
            return out

        def on_sequential_epoch_end(self, state, event, **kwargs):
            if journal is not None or saliency_out is not None:
                fresh = 0
                for layer_name, tracker in self._saliency_trackers.items():
                    if tracker.total_count <= 0:
                        continue
                    retained = tracker.compute_retained_experts(
                        self._n_experts_to_drop,
                        self._n_experts_to_drop_per_group,
                        self._moe_attrs,
                    )
                    previous = recorded.get(layer_name)
                    if previous is None:
                        fresh += 1
                    elif previous != retained:
                        # A layer is decided once and then pruned; a second, different
                        # answer means the saliency it was decided from has moved.
                        log(f"  WARNING: {layer_name} retained set changed after it "
                            "was journalled -- keeping the newer one")
                    recorded[layer_name] = retained
                    saliency[layer_name] = accumulators(tracker, layer_name)
                if journal is not None:
                    write(journal, recorded)
                if saliency_out is not None:
                    write(saliency_out, {"run": provenance or {}, "layers": saliency})
                log(f"  retained sets recorded for {len(recorded)} layer(s) "
                    f"(+{fresh} this subgraph)")

            super().on_sequential_epoch_end(state, event, **kwargs)

    # The accumulators are attached to the class, not hidden in the closure, so a
    # test can check that the per-source split sums back to the unsplit totals --
    # which is the one property the whole re-weighting rests on.
    REAPPruningModifierDSV4Journaled.recorded = recorded
    REAPPruningModifierDSV4Journaled.saliency = saliency
    REAPPruningModifierDSV4Journaled.by_source = by_source

    return REAPPruningModifierDSV4Journaled(sparsity=sparsity, hash_remap=hash_remap)


# --------------------------------------------------------------------------


def count_hash_layers(src: Path) -> int:
    """How many layers route by hash table, counted from the checkpoint itself.

    Not from the config: ``DeepseekV4Config`` does not declare ``num_hash_layers``
    and drops it, so ``getattr(config, "num_hash_layers", 0)`` reads 0 on a
    checkpoint whose config.json says 3. The modifier finds these layers the same
    way this does -- by the presence of ``tid2eid`` -- so counting the tables is
    also the number that matters.
    """
    index = json.loads((src / "model.safetensors.index.json").read_text())
    return sum(1 for name in index["weight_map"] if name.endswith(".tid2eid"))


def load_calibration(path: Path, vocab_size: int, max_samples: int | None):
    """``calib.pt`` is a flat list of token id sequences, as build_calibration.py writes it."""
    from datasets import Dataset

    blob = torch.load(path, weights_only=True)
    samples = [torch.as_tensor(t, dtype=torch.long) for t in blob]
    if max_samples:
        samples = samples[:max_samples]

    hi = max(int(s.max()) for s in samples)
    if hi >= vocab_size:
        raise ValueError(f"token id {hi} exceeds vocab_size {vocab_size}")

    seq_len = min(len(s) for s in samples)
    if any(len(s) != seq_len for s in samples):
        log(f"truncating all samples to the shortest length, {seq_len}")
        samples = [s[:seq_len] for s in samples]

    ids = torch.stack(samples)
    ds = Dataset.from_dict(
        {"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()}
    )
    return ds, len(samples), seq_len


class HiddenStateCapture:
    """Save the hidden states the draft head reads, while the run is making them.

    The DSpark draft head consumes ``main_norm(main_proj(concat of layers
    dspark_target_layer_ids))`` -- 40, 41 and 42 on this checkpoint. Those
    activations exist for free during REAP's calibration, and by the time the
    pipeline reaches those layers the ones below are already pruned, so they are
    the activations the head will actually face. Recomputing them afterwards
    costs another full forward over the calibration set, which is the hour this
    class exists to avoid: **if the drafter is ever to be scored, capture it
    here, on the pass that is already running.**

    Tokens are subsampled by ``stride``: saliency is a mean over tokens, so it
    converges long before every token is used, and 512 x 2048 x 4096 x 3 layers
    is 25.8 GB at stride 1.

    The sequential pipeline can run each subgraph twice -- once to trigger
    modifier hooks, once to propagate error -- so only the first ``batches``
    calls per layer are kept; the rest are the same activations a second time.
    """

    def __init__(self, out_dir: Path, layer_ids: tuple[int, ...], batches: int,
                 stride: int = 4):
        self.out_dir = out_dir
        self.layer_ids = layer_ids
        self.batches = batches
        self.stride = stride
        self.calls: dict[int, int] = {i: 0 for i in layer_ids}
        self.written = 0
        self.handles: list = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def attach(self, model) -> None:
        import torch

        for layer_id in self.layer_ids:
            layer = model.model.layers[layer_id]

            def hook(_module, _inputs, output, layer_id=layer_id):
                index = self.calls[layer_id]
                self.calls[layer_id] = index + 1
                if index >= self.batches:
                    return          # second pass over the same batches
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None or not hasattr(hidden, "dim"):
                    return
                # Subsample along the *token* axis. mHC returns [B, S, hc, D], so
                # tokens are axis 1, not -2: slicing -2 thins the stream axis and
                # keeps all 2048 tokens, which is the opposite of the intent.
                sliced = hidden[:, :: self.stride] if hidden.dim() == 4 else \
                    hidden[..., :: self.stride, :]
                torch.save(sliced.detach().to("cpu", torch.bfloat16),
                           self.out_dir / f"h{layer_id}-b{index:05d}.pt")
                self.written += 1

            self.handles.append(layer.register_forward_hook(hook))

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_model(src: Path, offload_dir: Path):
    """Load linearized and offloaded.

    Nesting matters. The innermost context's patch runs *first* in the call
    chain, so ``load_offloaded_model`` inside means: set the device map, hand
    off to the MoE loader, which loads the experts in their linearized 2D form
    (DeepSeek ships per-expert 2D weights, so this is a direct load, not a
    post-hoc restructure of 568 GB), then convert accelerate's offload into
    compressed-tensors' offload on the way back out.
    """
    from compressed_tensors.offload import load_offloaded_model
    from llmcompressor.modeling.moe.linearize import load_quantizable_moe
    from transformers import AutoModelForCausalLM

    offload_dir.mkdir(parents=True, exist_ok=True)
    with load_quantizable_moe(AutoModelForCausalLM):
        with load_offloaded_model(AutoModelForCausalLM):
            return AutoModelForCausalLM.from_pretrained(
                src,
                dtype=torch.bfloat16,
                device_map="auto_offload",
                offload_folder=str(offload_dir),
            )


def preflight(args) -> int:
    problems = []
    if not (args.model / "model.safetensors.index.json").exists():
        problems.append(f"no checkpoint index under {args.model}")
    if not args.tokens.exists():
        problems.append(
            f"no calibration tokens at {args.tokens} -- build them with "
            "scripts/build_calibration.py --mix calibration/mix-dsv4.json"
        )
    if not (args.tokenizer / "tokenizer.json").exists():
        problems.append(f"no tokenizer under {args.tokenizer}")

    # A calibrate-only run writes two small journals and nothing else, so neither
    # an existing output directory nor room for a second one is its business.
    if not args.calibrate_only:
        if args.out.exists() and any(args.out.iterdir()):
            problems.append(f"{args.out} exists and is not empty")

        free = shutil.disk_usage(args.out.parent).free
        if free < args.reserve_gb * 1024**3:
            problems.append(
                f"only {human(free)} free under {args.out.parent}; "
                f"the output plus the offload cache need about {args.reserve_gb} GB"
            )

    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", type=Path, default=REPO / "artifacts/dsv4-bf16",
                    help="BF16 checkpoint from dequantize_checkpoint.py")
    ap.add_argument("--tokens", type=Path, default=REPO / "artifacts/calib.pt")
    ap.add_argument("--tokenizer", type=Path,
                    default=REPO / "models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/dsv4-reap50-bf16")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--hash-remap", default="balanced", choices=["balanced", "nearest"],
                    help="where a dropped hash-routed expert's token ids go")
    ap.add_argument("--offload-dir", type=Path, default=REPO / "artifacts/offload",
                    help="disk offload cache; needs to be on the big volume")
    ap.add_argument("--retained-out", type=Path,
                    default=REPO / "artifacts/retained-sets.json")
    ap.add_argument("--saliency-out", type=Path,
                    default=REPO / "artifacts/saliency.json",
                    help="per-expert saliency, journalled alongside the retained sets. "
                         "It is what the selection was made from, so keeping it means "
                         "another sparsity can be costed without calibrating again")
    ap.add_argument("--sample-sources", type=Path,
                    help="JSON list naming the calibration source of each sample, as "
                         "label_calibration.py writes it. Splits the saliency by source, "
                         "which makes any other mixture weighting a topk instead of "
                         "another hour: sum_saliency and count are both additive over "
                         "tokens, so re-weighting them is exact, not an approximation")
    ap.add_argument("--max-samples", type=int,
                    help="use only the first N calibration samples")
    ap.add_argument("--max-shard-size", default="4GB",
                    help="output shard size; smaller shards cost less offload "
                         "footprint in any later run that reads this checkpoint")
    ap.add_argument("--reserve-gb", type=int, default=400,
                    help="free space the preflight insists on")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and load only, then stop before calibration")
    ap.add_argument("--capture-hidden", type=Path,
                    help="save the hidden states the DSpark draft head reads "
                         "(config's dspark_target_layer_ids) into this directory "
                         "while calibrating. Free here, an extra hour later: "
                         "score_mtp.py turns them into saliency for the MTP "
                         "blocks, which REAP itself never sees")
    ap.add_argument("--capture-stride", type=int, default=4,
                    help="keep every Nth token of the captured states (default: 4)")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="stop after calibration, before writing the pruned "
                         "checkpoint. For recovering the saliency of a run whose "
                         "output already exists: the decisions are deterministic, "
                         "so re-saving 272 GB of identical tensors buys nothing")
    args = ap.parse_args()

    if preflight(args):
        return 1

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    log(f"model    : {args.model}")
    log(f"experts  : {config.n_routed_experts} routed, "
        f"{count_hash_layers(args.model)} hash-routed layer(s)")
    log(f"sparsity : {args.sparsity} -> "
        f"{int(config.n_routed_experts * (1 - args.sparsity))} experts kept")

    ds, n_samples, seq_len = load_calibration(args.tokens, config.vocab_size,
                                              args.max_samples)
    log(f"calib    : {n_samples} samples x {seq_len} tokens = "
        f"{n_samples * seq_len:,} tokens")

    # Before any forward: fp32-pinned norms otherwise hand fp32 into bf16 Linears.
    patch_rmsnorm_dtype()

    log(f"loading offloaded (cache under {args.offload_dir})")
    t0 = time.perf_counter()
    model = load_model(args.model, args.offload_dir)
    log(f"loaded in {time.perf_counter() - t0:.0f}s")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dry_run:
        log("dry run: stopping before calibration")
        return 0

    capture = None
    if args.capture_hidden is not None:
        target_ids = tuple(json.loads((args.model / "config.json").read_text())
                           .get("dspark_target_layer_ids", ()))
        if not target_ids:
            log("no dspark_target_layer_ids in the config; nothing to capture")
        else:
            capture = HiddenStateCapture(args.capture_hidden, target_ids, n_samples,
                                         args.capture_stride)
            capture.attach(model)
            log(f"capturing layers {target_ids} to {args.capture_hidden} "
                f"(every {args.capture_stride}th token)")

    from llmcompressor import oneshot

    # Enough to tell later which run a saliency file came out of, without having
    # to trust that the surrounding files were not moved since.
    import transformers
    provenance = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(args.model),
        "sparsity": args.sparsity,
        "experts_kept": int(config.n_routed_experts * (1 - args.sparsity)),
        "n_routed_experts": config.n_routed_experts,
        "hash_remap": args.hash_remap,
        "samples": n_samples,
        "seq_len": seq_len,
        "tokens": str(args.tokens),
        "tokens_sha256": sha256(args.tokens),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    sources = None
    if args.sample_sources:
        sources = json.loads(args.sample_sources.read_text())
        if isinstance(sources, dict):        # {"sources": [...], ...} or {name: [idx]}
            sources = sources.get("sample_sources", sources)
        if len(sources) < n_samples:
            log(f"FAIL: {args.sample_sources} labels {len(sources)} sample(s) but "
                f"{n_samples} are being calibrated")
            return 1
        sources = list(sources[:n_samples])
        provenance["sample_sources"] = str(args.sample_sources)
        counts = Counter(sources)
        log(f"per-source saliency over {len(counts)} source(s): "
            + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    recipe = make_modifier(args.sparsity, args.hash_remap, args.retained_out,
                           args.saliency_out, provenance, sources, seq_len)
    log(f"calibrating (journalling to {args.retained_out} and {args.saliency_out})")
    t0 = time.perf_counter()
    oneshot(
        model=model,
        dataset=ds,
        processor=tokenizer,
        recipe=recipe,
        max_seq_length=seq_len,
        num_calibration_samples=n_samples,
        moe_calibrate_all_experts=False,
        pipeline="sequential",
    )
    log(f"calibration + pruning done in {(time.perf_counter() - t0) / 60:.1f} min")
    if capture is not None:
        capture.detach()
        log(f"captured {capture.written} hidden-state tensors to {args.capture_hidden}")

    if args.calibrate_only:
        log(f"calibrate-only: stopping before saving. Retained sets in "
            f"{args.retained_out}, saliency in {args.saliency_out}")
        return 0

    log(f"saving to {args.out}")
    t0 = time.perf_counter()
    model.save_pretrained(args.out, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(args.out)
    log(f"saved in {(time.perf_counter() - t0) / 60:.1f} min")

    kept = model.config.n_routed_experts
    log(f"done: n_routed_experts={kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
