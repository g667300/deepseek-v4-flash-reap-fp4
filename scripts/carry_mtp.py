#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Put the MTP blocks back into the pruned checkpoint, still quantized.

Step 4, and the only step that touches tensors transformers never showed us.
``DeepseekV4PreTrainedModel`` sets ``_keys_to_ignore_on_load_unexpected =
[r"(^|\\.)mtp\\..*"]``, so all 4,705 ``mtp.*`` tensors are dropped on load and no
checkpoint the reference path saves can contain them. They are not lost, though
-- they are still sitting in the original shards, quantized, untouched. This
copies them across.

**The bytes are copied, not re-encoded.** Every other weight in the output went
through BF16 and came back; these do not have to, so they do not. What is
carried is exactly what DeepSeek published, which also means this is the one
part of the pruned checkpoint that *can* still be checked against the source.

**The experts have to be chosen without REAP.** An MTP block has its own 256
experts and its own router, and the REAP run never saw either -- transformers
drops ``mtp.*`` on load, so no saliency for them exists. ``--score`` picks how
to decide instead, and the three answers are not equally good; each was checked
by how often it reproduces REAP's *own* retained set on the 40 main-stack layers
where both are known:

* ``gate-bias`` (default) -- keep the lowest-bias experts. ``ffn.gate.bias`` is
  the aux-loss-free balancing correction noaux_tc learns, pushed down on experts
  the router already favours, so it is a trained record of usage that ships in
  the un-pruned checkpoint. **68%** agreement, Spearman -0.476 against saliency.
* ``router-norm`` -- rank by router row norm. **58%**.
* ``layer`` -- borrow a main-stack layer's set (``--from-layer``, default the
  last scored). On the MTP blocks this overlaps their own ranking **45-48%** of
  the time, i.e. chance: the borrowed set says nothing about these experts. It
  is what the first build used, kept for reproducing it.

Survivors are renumbered 0..127 so the block matches the ``n_routed_experts``
the rest of the model now declares, and ``--blocks`` limits which blocks are
carried at all.

The block's router is sliced with the same set, its bias with it, and everything
else in the block -- attention, norms, the hyper-connection scalars, the
confidence and markov heads -- is copied unchanged.

Usage::

    carry_mtp.py --src models/DeepSeek-V4-Flash-0731 --dst artifacts/dsv4-reap50
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dequantize_checkpoint import DTYPE_BYTES, INDEX_NAME, read_header  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EXPERT = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)$")
ROUTER = re.compile(r"^mtp\.(\d+)\.ffn\.gate\.(weight|bias)$")
BLOCK = re.compile(r"^mtp\.(\d+)\.")
LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.mlp$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def pick_retained(retained: dict[str, list[int]], from_layer: int | None) -> tuple[list[int], int]:
    """The retained set to prune every MTP block with, and the layer it came from."""
    layers = {}
    for key, keep in retained.items():
        match = LAYER_KEY.match(key)
        if match:
            layers[int(match.group(1))] = keep
    if not layers:
        raise KeyError("no `model.layers.N.mlp` entries in the retained sets")
    layer = max(layers) if from_layer is None else from_layer
    if layer not in layers:
        raise KeyError(f"no retained set for layer {layer}; have {min(layers)}..{max(layers)}")
    return layers[layer], layer


def _mtp_tensor(src: Path, name: str) -> torch.Tensor:
    weight_map = json.loads((src / INDEX_NAME).read_text())["weight_map"]
    with safe_open(str(src / weight_map[name]), framework="pt") as handle:
        return handle.get_tensor(name).float()


def router_norm_retained(src: Path, block: str, keep_n: int) -> list[int]:
    """The block's own top-``keep_n`` experts by router row norm.

    A weak proxy: REAP scores ``gate weight x expert output norm`` over real
    tokens, and this sees only the router matrix. Measured on this checkpoint it
    reproduces REAP's own choice on layer 42 about 58% of the time.
    """
    norms = _mtp_tensor(src, f"mtp.{block}.ffn.gate.weight").norm(dim=1)
    return sorted(torch.topk(norms, keep_n).indices.tolist())


def gate_bias_retained(src: Path, block: str, keep_n: int) -> list[int]:
    """The block's ``keep_n`` experts with the *lowest* routing bias.

    ``ffn.gate.bias`` is the aux-loss-free load-balancing correction that
    noaux_tc routing learns: it is pushed **down** on experts the router already
    favours and **up** on the ones it neglects, so a low bias marks a popular
    expert. That makes it a trained record of expert usage that ships inside the
    un-pruned checkpoint, needing no calibration to read.

    Checked against REAP's own decisions on the 40 main-stack layers that have a
    bias: saliency and bias correlate at **-0.476** mean Spearman (negative in
    all 40), and "keep the 128 lowest-bias experts" reproduces REAP's retained
    set **68%** of the time, against 58% for router norms and 50% for chance.
    Still a proxy -- but the best one available without scoring these experts.
    """
    bias = _mtp_tensor(src, f"mtp.{block}.ffn.gate.bias")
    return sorted(torch.topk(-bias, keep_n).indices.tolist())


def usage_retained(usage_file: Path, block: str, keep_n: int,
                   slot_to_expert: list[int]) -> list[int]:
    """The block's ``keep_n`` most-routed-to experts, measured while serving.

    ``moe_usage_probe.py`` counts expert selections per MoE layer while
    ``replay_calibration.py`` pushes the calibration tokens through a served
    checkpoint. The drafter's blocks are the last three MoE layers it records,
    and its counts are indexed by *slot* in the already-pruned block, so they
    are mapped back through the retained set that built that checkpoint.

    This is the only signal about these experts that was ever measured rather
    than guessed. It is a frequency, not REAP's contribution -- the two do not
    correlate on the main stack (-0.142 Spearman) -- but for *dropping* experts
    it is decisive in one direction: an expert selected zero times in 7,822
    generated tokens is not carrying the draft head.
    """
    counts = json.loads(usage_file.read_text())["selections_by_layer"]
    drafter = [counts[k] for k in sorted(counts)][-3:]
    block_counts = drafter[int(block)]
    ranked = sorted(range(len(block_counts)), key=lambda s: -block_counts[s])
    return sorted(slot_to_expert[s] for s in ranked[:keep_n] if s < len(slot_to_expert))


def saliency_retained(saliency_file: Path, block: str, keep_n: int,
                      key: str = "mean") -> list[int]:
    """The block's ``keep_n`` experts by REAP saliency, measured on this head.

    ``score_mtp_captured.py`` writes this from activations captured inside the
    serving model, so it is the only selection here computed the way REAP
    computes the main stack's -- ``gate weight x expert output norm`` over real
    tokens, rather than a property of the weights read off the checkpoint.

    ``key`` picks which accumulator ranks the experts, and the two disagree here
    in a way they do not on the main stack:

    ``mean``
        what REAP itself uses (``REAPSaliencyTracker.compute_retained_experts``
        reads ``mean_saliency``), so this is the faithful choice. It is a
        per-token contribution and deliberately ignores how often an expert
        fires -- fine where every expert sees thousands of tokens, as in the
        main stack.
    ``sum_saliency``
        the same quantity before dividing by the token count, so it is
        frequency-weighted. The draft head's routing is heavily skewed (its
        busiest expert takes ~8% of all selections while a fifth of them are
        touched fewer than ten times in 4k tokens), and measured on the capture
        the ``mean`` top-128 keeps only 43-50% of the head's routing traffic
        where ``sum_saliency`` keeps 88-95%.
    """
    blocks = json.loads(saliency_file.read_text())["blocks"]
    if block not in blocks:
        raise KeyError(f"no saliency for mtp.{block} in {saliency_file}")
    if key not in blocks[block]:
        raise KeyError(f"no '{key}' in mtp.{block}'s saliency; "
                       f"have {sorted(blocks[block])}")
    score = blocks[block][key]
    return sorted(sorted(range(len(score)), key=lambda i: -score[i])[:keep_n])


def tensor_bytes(meta: dict) -> int:
    numel = 1
    for dim in meta["shape"]:
        numel *= dim
    return numel * DTYPE_BYTES[meta["dtype"]]


def plan(src: Path, keeps: dict[str, list[int]], blocks: list[str] | None):
    """Which source tensor becomes which output tensor, and how big the result is.

    ``keeps`` maps block index to that block's retained experts -- one shared
    list under ``--score layer``, a per-block one under ``--score router-norm``.

    Returns ``(records, total_bytes, blocks_seen)``. A record is
    ``(source name, output name, is_router)`` -- the flag marks the router
    tensors, which get indexed rather than renamed.
    """
    weight_map = json.loads((src / INDEX_NAME).read_text())["weight_map"]
    mtp = {n: s for n, s in weight_map.items() if n.startswith("mtp.")}
    headers: dict[str, dict] = {}
    for shard in sorted(set(mtp.values())):
        headers[shard] = read_header(src / shard)

    slot_of = {b: {old: new for new, old in enumerate(keep)} for b, keep in keeps.items()}
    records: list[tuple[str, str, bool]] = []
    total = 0
    seen: set[str] = set()

    for name in sorted(mtp):
        block = BLOCK.match(name).group(1)
        if blocks is not None and block not in blocks:
            continue
        if block not in keeps:
            continue
        seen.add(block)
        meta = headers[mtp[name]][name]

        expert = EXPERT.match(name)
        if expert is not None:
            index = int(expert.group(2))
            if index not in slot_of[block]:
                continue        # this expert did not survive
            out = (f"mtp.{expert.group(1)}.ffn.experts.{slot_of[block][index]}."
                   f"{expert.group(3)}.{expert.group(4)}")
            records.append((name, out, False))
            total += tensor_bytes(meta)
            continue

        router = ROUTER.match(name)
        if router is not None:
            records.append((name, name, True))
            # sliced to len(keep) rows
            rows = meta["shape"][0]
            total += tensor_bytes(meta) * len(keeps[block]) // rows
            continue

        records.append((name, name, False))
        total += tensor_bytes(meta)

    return records, total, sorted(seen, key=int)


def publish(dst: Path, extra: list[tuple[Path, list[str]]], weight_map: dict[str, str]) -> None:
    """Renumber every shard to the new total and write the index in one go.

    The existing shards are named ``-of-00021``; adding files makes that a lie,
    and a checkpoint whose filenames disagree with its own count is the kind of
    thing that loads everywhere except the one place it matters. Renames are
    metadata-only, and the index is written last and atomically, so an
    interrupted run leaves either the old checkpoint or the new one.
    """
    existing = sorted(p for p in dst.glob("model-*.safetensors"))
    total = len(existing) + len(extra)
    renamed: dict[str, str] = {}

    for i, path in enumerate(existing, start=1):
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        if path.name != new:
            path.rename(dst / new)
        renamed[path.name] = new
    for j, (path, _names) in enumerate(extra, start=len(existing) + 1):
        new = f"model-{j:05d}-of-{total:05d}.safetensors"
        if path.name != new:
            path.rename(dst / new)
        renamed[path.name] = new

    updated = {name: renamed.get(shard, shard) for name, shard in weight_map.items()}
    tmp = dst / (INDEX_NAME + ".tmp")
    size = sum((dst / s).stat().st_size for s in set(updated.values()))
    tmp.write_text(json.dumps(
        {"metadata": {"total_size": size}, "weight_map": dict(sorted(updated.items()))},
        indent=2,
    ) + "\n")
    os.replace(tmp, dst / INDEX_NAME)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO / "models/DeepSeek-V4-Flash-0731",
                    help="the original checkpoint, which still has the MTP blocks")
    ap.add_argument("--dst", type=Path, default=REPO / "artifacts/dsv4-reap50",
                    help="the pruned checkpoint to add them to (modified in place)")
    ap.add_argument("--retained", type=Path, default=REPO / "artifacts/retained-sets.json")
    ap.add_argument("--score",
                    choices=["layer", "router-norm", "gate-bias", "usage", "saliency"],
                    default="gate-bias",
                    help="how to choose which MTP experts survive. 'gate-bias' keeps "
                         "the lowest-bias experts, i.e. the ones noaux_tc routing had "
                         "to suppress because they were popular -- it agrees with "
                         "REAP's own choice 68%% of the time on the main stack. "
                         "'router-norm' ranks by router row norm (58%%). 'layer' "
                         "borrows a main-stack layer's retained set (chance, 45-48%%), "
                         "which is what the first build used. 'saliency' uses "
                         "REAP's own formula, measured on the head itself -- the "
                         "only one of these that is not a proxy")
    ap.add_argument("--from-layer", type=int,
                    help="with --score layer: prune the blocks with this layer's "
                         "retained set (default: the last layer the REAP run scored)")
    ap.add_argument("--saliency", type=Path, default=REPO / "artifacts/mtp-saliency.json",
                    help="with --score saliency: what score_mtp_captured.py wrote")
    ap.add_argument("--saliency-key", default="mean", choices=("mean", "sum_saliency"),
                    help="with --score saliency: which accumulator ranks the experts. "
                         "'mean' is what REAP itself selects on; 'sum_saliency' is the "
                         "same contribution before dividing by token count, so it keeps "
                         "the experts the head leans on rather than the ones with the "
                         "highest per-token contribution (default: mean)")
    ap.add_argument("--usage", type=Path, default=REPO / "artifacts/moe-usage.json",
                    help="with --score usage: the histogram moe_usage_probe.py wrote")
    ap.add_argument("--experts", type=int,
                    help="how many experts to keep per block (default: as many as the "
                         "main stack keeps). Do not go below 128 on the strength of the "
                         "old usage histogram: it was taken with CUDA graphs on and saw "
                         "195 rows, and eager re-measurement puts 232-247 of 256 experts "
                         "in use. A 64-expert head was built on that reading and drafted "
                         "0 accepted tokens. vLLM sizes the drafter from n_routed_experts, "
                         "so any smaller block needs the dspark patch that reads "
                         "dspark_n_routed_experts")
    ap.add_argument("--blocks", nargs="*",
                    help="carry only these block indices, e.g. --blocks 0")
    ap.add_argument("--shard-size", type=float, default=4.0,
                    help="target shard size in GB for the added files (default: 4)")
    ap.add_argument("--dry-run", action="store_true", help="report the plan only")
    args = ap.parse_args()

    for path, what in ((args.src / INDEX_NAME, "source checkpoint"),
                       (args.dst / INDEX_NAME, "pruned checkpoint"),
                       (args.retained, "retained sets")):
        if not path.exists():
            print(f"FAIL: no {what} at {path}", file=sys.stderr)
            return 1

    dst_map = json.loads((args.dst / INDEX_NAME).read_text())["weight_map"]
    if any(n.startswith("mtp.") for n in dst_map):
        print(f"FAIL: {args.dst} already has mtp tensors", file=sys.stderr)
        return 1

    retained = json.loads(args.retained.read_text())
    keep, layer = pick_retained(retained, args.from_layer)

    present = sorted({BLOCK.match(n).group(1) for n in
                      json.loads((args.src / INDEX_NAME).read_text())["weight_map"]
                      if n.startswith("mtp.")}, key=int)
    wanted = [b for b in present if args.blocks is None or b in args.blocks]
    keep_n = args.experts or len(keep)
    if args.score == "saliency":
        keeps = {b: saliency_retained(args.saliency, b, keep_n, args.saliency_key)
                 for b in wanted}
        how = (f"each block's top-{keep_n} by REAP saliency ({args.saliency_key}), "
               f"measured on the head")
    elif args.score == "usage":
        keeps = {b: usage_retained(args.usage, b, keep_n, keep) for b in wanted}
        how = f"each block's {keep_n} most-used experts, measured while serving"
    elif args.score == "gate-bias":
        keeps = {b: gate_bias_retained(args.src, b, keep_n) for b in wanted}
        how = f"each block's own {len(keep)} lowest-bias experts"
    elif args.score == "router-norm":
        keeps = {b: router_norm_retained(args.src, b, keep_n) for b in wanted}
        how = f"each block's own router-norm top-{len(keep)}"
    else:
        keeps = {b: keep for b in wanted}
        how = f"layer {layer}'s retained set"

    records, total, blocks = plan(args.src, keeps, args.blocks)
    if not records:
        print("FAIL: no mtp tensors matched", file=sys.stderr)
        return 1

    log(f"source   : {args.src}")
    log(f"target   : {args.dst}")
    log(f"blocks   : {', '.join(blocks)}")
    log(f"experts  : keeping {keep_n} of 256 per block, chosen by {how}")
    if args.score != "layer":
        # How far this lands from the borrowed set is the whole point of the
        # option, so say it rather than making the reader diff two checkpoints.
        for b in blocks:
            shared = len(set(keeps[b]) & set(keep))
            log(f"           mtp.{b}: {shared}/{keep_n} shared with layer {layer}'s set")
    log(f"tensors  : {len(records):,} carried, {human(total)}")

    free = shutil.disk_usage(args.dst).free
    if free < total * 1.02:
        print(f"\nNOT ENOUGH DISK: need {human(total)}, have {human(free)}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    src_map = json.loads((args.src / INDEX_NAME).read_text())["weight_map"]
    handles: dict[str, object] = {}

    def get(name: str) -> torch.Tensor:
        shard = src_map[name]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(args.src / shard), framework="pt")
            handles[shard] = handle
        return handle.get_tensor(name)

    index_of = {b: torch.as_tensor(k, dtype=torch.long) for b, k in keeps.items()}
    shard_bytes = int(args.shard_size * 1e9)
    added: list[tuple[Path, list[str]]] = []
    tensors: dict[str, torch.Tensor] = {}
    pending = 0
    written = 0

    def flush() -> None:
        nonlocal tensors, pending, written
        if not tensors:
            return
        path = args.dst / f"mtp-{len(added) + 1:05d}.part"
        save_file(tensors, str(path), metadata={"format": "pt"})
        added.append((path, sorted(tensors)))
        written += pending
        log(f"  {path.name}: {len(tensors):,} tensors, {human(pending)}"
            f"  ({human(written)}/{human(total)})")
        tensors, pending = {}, 0

    for source, out, is_router in records:
        tensor = get(source)
        if is_router:
            tensor = tensor[index_of[BLOCK.match(source).group(1)]]
        tensors[out] = tensor
        pending += tensor.numel() * tensor.element_size()
        if pending >= shard_bytes:
            flush()
    flush()

    weight_map = dict(dst_map)
    for path, names in added:
        for name in names:
            weight_map[name] = path.name
    publish(args.dst, added, weight_map)

    config = json.loads((args.dst / "config.json").read_text())
    reference = json.loads((args.src / "config.json").read_text())
    # The count DeepSeek published, not the number of blocks on disk: the
    # checkpoint ships three and the config asks for one, and that is their call.
    config["num_nextn_predict_layers"] = reference.get("num_nextn_predict_layers", 0)
    # verify_quantized.py reads this to know which source expert each carried
    # slot came from; it refuses to check the MTP tensors without it.
    if args.score != "layer":
        config.pop("_mtp_pruned_with_layer", None)
        config["_mtp_retained"] = {b: keeps[b] for b in blocks}
        config["_mtp_scored_by"] = args.score
        config.pop("_mtp_saliency_key", None)
        if args.score == "saliency":
            # 'mean' and 'sum_saliency' select differently enough to matter
            # (their top-128 sets share 84-87 of 128), so which one produced this
            # checkpoint is part of what identifies it.
            config["_mtp_saliency_key"] = args.saliency_key
        if keep_n != config["n_routed_experts"]:
            # vLLM builds the draft head from this; the stock loader reads
            # n_routed_experts and would look for experts that are not there.
            config["dspark_n_routed_experts"] = keep_n
        else:
            config.pop("dspark_n_routed_experts", None)
    else:
        config.pop("_mtp_retained", None)
        config.pop("_mtp_scored_by", None)
        config["_mtp_pruned_with_layer"] = layer
    (args.dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    log(f"carried {len(records):,} tensors, {human(written)} -> {args.dst}")
    log(f"num_nextn_predict_layers = {config['num_nextn_predict_layers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
