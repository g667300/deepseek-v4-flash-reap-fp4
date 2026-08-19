#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Re-encode the pruned BF16 checkpoint into DeepSeek's format, under native names.

Step 3 of the README pipeline: 272 GB of pruned BF16 in, about 82 GB of FP4/FP8
out. Two things happen here, and the second one is not optional.

**Every name changes.** ``save_pretrained`` writes transformers' modern spellings
-- measured, not assumed: of 257 tensors in the synthetic checkpoint exactly one
name survives the round trip through transformers, and on the real thing
``layers.{L}.ffn.experts.{E}.w1.weight`` comes back as
``model.layers.{L}.mlp.experts.{E}.gate_proj.weight``. vLLM reads what DeepSeek
published, so :func:`names.to_native` puts every name back.

**Every weight is re-encoded.** That is the cost this repo exists to measure: a
round trip through BF16 rewrites even the weights REAP never touched, so unlike
the companion repo's file-level surgery, nothing here can be checked against the
source afterwards. The encoding itself is exact where it can be -- see
``test_quant_encode.py`` -- but the comparison is gone either way.

Which encoding each tensor gets is read off the **reference** checkpoint rather
than decided here: whatever ``models/DeepSeek-V4-Flash-0731`` stores a tensor as,
this writes the same thing. Routed experts are FP4 (I8 nibble pairs plus an E8M0
scale per 32 elements), attention and shared-expert projections are FP8 E4M3 with
a 128x128 E8M0 scale, and everything else -- norms, biases, ``tid2eid``, the
hyper-connection scalars -- is stored plain.

**The plain tensors come from the reference, not from step 2.** Step 2 loads with
``dtype=torch.bfloat16``, which flattens the ~400 tensors DeepSeek keeps in fp32:
``attn_sink``, ``hc_*``, ``compressor.ape`` and, worst, ``ffn.gate.bias`` -- the
bias added to router scores before top-k, which bf16 moves by up to 0.031. REAP
does not touch any of them (verified bit-identical for the tensors that survive
unsliced), so taking them from the reference restores the values instead of
casting a rounded copy back up. ``--no-restore-untouched`` disables that and
takes everything from step 2, if what you want is exactly what the reference path
produced.

The three tensors REAP *does* change always come from step 2: the sliced
``ffn.gate.weight``, the remapped ``ffn.gate.tid2eid``, and ``ffn.gate.bias``,
which is rebuilt by slicing the reference's fp32 bias with the retained set.

Usage::

    quantize_to_deepseek.py --src artifacts/dsv4-reap50-bf16 \\
        --dst artifacts/dsv4-reap50 --reference models/DeepSeek-V4-Flash-0731
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dequantize_checkpoint import DTYPE_BYTES, INDEX_NAME, read_header  # noqa: E402
from names import to_native  # noqa: E402
from quant import FP4_BLOCK, FP8_BLOCK, quantize_linear  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

TORCH_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16,
                "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}

# The three tensors REAP rewrites. Everything else it either drops whole (the
# experts that lost) or leaves exactly as it found it.
GATE_WEIGHT = re.compile(r"^layers\.(\d+)\.ffn\.gate\.weight$")
GATE_BIAS = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$")
GATE_TABLE = re.compile(r"^layers\.(\d+)\.ffn\.gate\.tid2eid$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def human(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def native_of(name: str, reference: set[str]) -> str:
    """The DeepSeek name for a step-2 tensor.

    ``save_original_format=True`` already writes one name natively
    (``head.weight``), so a name the reference itself has is passed straight
    through rather than pushed through the table.
    """
    if name in reference:
        return name
    return to_native(name)


def encoded_bytes(shape: list[int], kind: str, dtype: str) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    if kind == "fp4":
        return numel // 2 + numel // FP4_BLOCK
    if kind == "fp8":
        rows, cols = shape
        return numel + -(-rows // FP8_BLOCK) * -(-cols // FP8_BLOCK)
    return numel * DTYPE_BYTES[dtype]


def plan(src: Path, ref: Path, restore: bool, shard_bytes: int):
    """Decide, for every step-2 tensor, what it becomes and where it comes from.

    Returns ``(shards, total_bytes)`` where each shard is a list of records the
    workers can act on without consulting any global map.
    """
    src_map = json.loads((src / INDEX_NAME).read_text())["weight_map"]
    ref_map = json.loads((ref / INDEX_NAME).read_text())["weight_map"]
    ref_names = set(ref_map)

    src_headers = {s: read_header(src / s) for s in sorted(set(src_map.values()))}
    ref_headers = {s: read_header(ref / s) for s in sorted(set(ref_map.values()))}

    def ref_meta(name: str) -> dict:
        return ref_headers[ref_map[name]][name]

    shards: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = total = 0
    unknown: list[str] = []

    for shard in sorted(set(src_map.values())):
        for name, meta in src_headers[shard].items():
            native = native_of(name, ref_names)
            if native not in ref_names:
                unknown.append(f"{name} -> {native}")
                continue

            # The reference is the authority on how a tensor is stored.
            scale = f"{native[: -len('.weight')]}.scale" if native.endswith(".weight") else None
            if scale in ref_names:
                kind = "fp4" if ref_meta(native)["dtype"] in ("I8", "U8") else "fp8"
            else:
                kind = "keep"

            record = {
                "native": native,
                "kind": kind,
                "from": "src",
                "file": shard,
                "name": name,
                "slice": None,
                "dtype": ref_meta(native)["dtype"],
            }
            if kind == "keep" and restore:
                layer = GATE_BIAS.match(native)
                if layer is not None:
                    # Sliced, but out of the reference's fp32 copy rather than the
                    # bf16 one step 2 wrote.
                    record["from"] = "ref"
                    record["file"] = ref_map[native]
                    record["name"] = native
                    record["slice"] = int(layer.group(1))
                elif not (GATE_WEIGHT.match(native) or GATE_TABLE.match(native)):
                    record["from"] = "ref"
                    record["file"] = ref_map[native]
                    record["name"] = native

            size = encoded_bytes(meta["shape"], kind, record["dtype"])
            record["bytes"] = size
            if current and current_bytes + size > shard_bytes:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(record)
            current_bytes += size
            total += size

    if unknown:
        raise KeyError(
            f"{len(unknown)} tensor(s) have no counterpart in the reference, e.g. "
            + ", ".join(unknown[:3])
        )
    if current:
        shards.append(current)
    return shards, total


# --------------------------------------------------------------------------
# the work
# --------------------------------------------------------------------------


def build_shard(src_dir: str, ref_dir: str, dst_path: str, records_json: str,
                retained_json: str) -> tuple[str, dict[str, int]]:
    """Encode one output shard. Returns (filename, {native name: nbytes})."""
    records = json.loads(records_json)
    retained = json.loads(retained_json)
    roots = {"src": Path(src_dir), "ref": Path(ref_dir)}
    handles: dict[tuple[str, str], object] = {}

    def get(where: str, file: str, name: str) -> torch.Tensor:
        key = (where, file)
        handle = handles.get(key)
        if handle is None:
            handle = safe_open(str(roots[where] / file), framework="pt")
            handles[key] = handle
        return handle.get_tensor(name)

    tensors: dict[str, torch.Tensor] = {}
    for r in records:
        tensor = get(r["from"], r["file"], r["name"])

        if r["slice"] is not None:
            keep = torch.as_tensor(retained[f"model.layers.{r['slice']}.mlp"],
                                   dtype=torch.long)
            tensor = tensor[keep]

        if r["kind"] == "keep":
            want = TORCH_DTYPES[r["dtype"]]
            tensors[r["native"]] = tensor.to(want) if tensor.dtype != want else tensor
            continue

        weight, scale = quantize_linear(tensor.to(torch.bfloat16), r["kind"])
        tensors[r["native"]] = weight
        tensors[f"{r['native'][: -len('.weight')]}.scale"] = scale

    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, dst_path, metadata={"format": "pt"})
    return Path(dst_path).name, {n: t.numel() * t.element_size() for n, t in tensors.items()}


def shard_is_done(path: Path, records: list[dict]) -> bool:
    """True if this shard was already written, with exactly the tensors planned.

    Long runs on this machine are expected to die (see docs/memory-faults.md), and
    a shard is either wholly present or absent -- ``save_file`` writes it in one
    call -- so an existing file with the right header can be skipped.
    """
    if not path.exists():
        return False
    expected = set()
    for r in records:
        expected.add(r["native"])
        if r["kind"] != "keep":
            expected.add(f"{r['native'][: -len('.weight')]}.scale")
    try:
        return set(read_header(path)) == expected
    except Exception:
        return False


def write_config(src: Path, ref: Path, dst: Path) -> None:
    """DeepSeek's own config, with the pruning's one structural change.

    The reference config is the base rather than step 2's: transformers 5.14
    rewrote that one into its modern schema (``rope_parameters``, ``layer_types``)
    and dropped the quantization declaration this step restores. Only
    ``n_routed_experts`` actually differs between the two for this model.
    """
    config = json.loads((ref / "config.json").read_text())
    pruned = json.loads((src / "config.json").read_text())
    config["n_routed_experts"] = pruned["n_routed_experts"]
    # carry_mtp.py puts the block back and restores the count.
    config["num_nextn_predict_layers"] = 0
    config["_pruned_from"] = str(ref)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def copy_aux(ref: Path, dst: Path) -> list[str]:
    """Tokenizer and friends, from the reference. Directories are left behind."""
    copied = []
    for path in sorted(ref.iterdir()):
        if path.is_dir() or path.suffix == ".safetensors":
            continue
        if path.name in {INDEX_NAME, "config.json"}:
            continue
        shutil.copy2(path, dst / path.name)
        copied.append(path.name)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO / "artifacts/dsv4-reap50-bf16",
                    help="pruned BF16 checkpoint from reap_reference.py")
    ap.add_argument("--dst", type=Path, default=REPO / "artifacts/dsv4-reap50")
    ap.add_argument("--reference", type=Path, default=REPO / "models/DeepSeek-V4-Flash-0731",
                    help="the original checkpoint: says how each tensor is stored, "
                         "and supplies the tensors REAP never touched")
    ap.add_argument("--retained", type=Path, default=REPO / "artifacts/retained-sets.json",
                    help="retained sets from the REAP run, for slicing the router bias")
    ap.add_argument("--shard-size", type=float, default=4.0,
                    help="target output shard size in GB (default: 4)")
    ap.add_argument("--workers", type=int, default=3,
                    help="shards encoded in parallel; each holds one in memory")
    ap.add_argument("--no-restore-untouched", action="store_true",
                    help="take every tensor from --src, including the fp32 ones step 2 "
                         "rounded to bf16. Produces exactly what the reference path made")
    ap.add_argument("--max-shards", type=int,
                    help="stop after this many shards -- a partial checkpoint that will "
                         "not load, for checking the encoding before committing 82 GB")
    ap.add_argument("--dry-run", action="store_true", help="report the plan only")
    args = ap.parse_args()

    for path, what in ((args.src / INDEX_NAME, "pruned checkpoint"),
                       (args.reference / INDEX_NAME, "reference checkpoint"),
                       (args.retained, "retained sets")):
        if not path.exists():
            print(f"FAIL: no {what} at {path}", file=sys.stderr)
            return 1

    restore = not args.no_restore_untouched
    shards, total = plan(args.src, args.reference, restore, int(args.shard_size * 1e9))
    if args.max_shards:
        shards = shards[: args.max_shards]
        total = sum(r["bytes"] for records in shards for r in records)
        print(f"PARTIAL: stopping after {len(shards)} shards (--max-shards)")

    kinds = {"fp4": 0, "fp8": 0, "keep": 0}
    from_ref = 0
    for names in shards:
        for r in names:
            kinds[r["kind"]] += 1
            from_ref += r["from"] == "ref"

    log(f"source    : {args.src}")
    log(f"reference : {args.reference}")
    log(f"output    : {args.dst}")
    log(f"tensors   : {sum(len(s) for s in shards):,} in {len(shards)} shards of "
        f"~{args.shard_size:.1f} GB")
    log(f"encoding  : {kinds['fp4']:,} FP4, {kinds['fp8']:,} FP8, {kinds['keep']:,} plain")
    log(f"untouched : {from_ref:,} tensor(s) taken from the reference"
        if restore else "untouched : none -- everything comes from --src")
    log(f"total     : {human(total)}")

    free = shutil.disk_usage(args.dst.parent if args.dst.parent.exists() else ".").free
    log(f"free disk : {human(free)}")
    if free < total * 1.02:
        print(f"\nNOT ENOUGH DISK: need {human(total)}, have {human(free)}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    args.dst.mkdir(parents=True, exist_ok=True)
    retained_json = args.retained.read_text()
    weight_map: dict[str, str] = {}
    written = done = skipped = 0

    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {}
        for i, records in enumerate(shards, start=1):
            name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            if shard_is_done(args.dst / name, records):
                for r in records:
                    weight_map[r["native"]] = name
                    if r["kind"] != "keep":
                        weight_map[f"{r['native'][: -len('.weight')]}.scale"] = name
                skipped += 1
                continue
            futures[pool.submit(
                build_shard, str(args.src), str(args.reference), str(args.dst / name),
                json.dumps(records), retained_json,
            )] = name

        if skipped:
            log(f"resuming: {skipped} shard(s) already written")
        for future in as_completed(futures):
            name, sizes = future.result()
            weight_map.update(dict.fromkeys(sizes, name))
            written += sum(sizes.values())
            done += 1
            log(f"  [{done}/{len(futures)}] {name}  {human(sum(sizes.values()))}"
                f"  ({human(written)} written)")

    (args.dst / INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {"total_size": written},
             "weight_map": dict(sorted(weight_map.items()))},
            indent=2,
        ) + "\n"
    )
    write_config(args.src, args.reference, args.dst)
    copy_aux(args.reference, args.dst)

    log(f"wrote {len(weight_map):,} tensors, {human(written)} -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
