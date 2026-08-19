#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Turn captured draft-head activations into REAP saliency.

``moe_capture_probe.py`` records, from inside the serving model, exactly what
each draft-head router saw and decided: the hidden states reaching the FFN, the
expert ids chosen, and the gate weights. This finishes the arithmetic REAP would
have done -- ``S_j = mean(g_j * ||f_j||_2)`` -- by running the experts over their
own routed tokens, using the checkpoint's weights.

Splitting it this way is the point. The first attempt computed the router input
offline too, from the main model's layer 40-42 hidden states, and it scored
**+0.225** rank correlation against the real routing while touching 186 of 256
experts where vLLM touches 74: the draft head's router reads the state *after*
its block's attention, which no amount of care outside the model reproduces. The
activations have to be measured. The expert arithmetic does not -- it is a
matmul over weights that are sitting in the checkpoint.

Usage::

    score_mtp_captured.py --capture artifacts/mtp-capture \\
        --src models/DeepSeek-V4-Flash-0731 --out artifacts/mtp-saliency.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import dequantize_linear  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
INDEX = "model.safetensors.index.json"
FILE = re.compile(r"^L(\d+)-(\d+)\.pt$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Experts:
    """Dequantized expert weights, fetched once and cached per expert."""

    def __init__(self, src: Path, block: int, device: str):
        self.src = src
        self.block = block
        self.device = device
        self.map = json.loads((src / INDEX).read_text())["weight_map"]
        self._handles: dict[str, object] = {}

    def _raw(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.src / shard), framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def weights(self, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = []
        for which in ("w1", "w3", "w2"):
            prefix = f"mtp.{self.block}.ffn.experts.{expert}.{which}"
            tensors = {f"{prefix}.weight": self._raw(f"{prefix}.weight")}
            scale = f"{prefix}.scale"
            if scale in self.map:
                tensors[scale] = self._raw(scale)
            out.append(dequantize_linear(tensors, prefix, scheme="deepseek",
                                         dtype=torch.bfloat16).to(self.device))
        return tuple(out)


def score_block(capture_dir: Path, layer: int, src: Path, block: int,
                n_experts: int, device: str) -> dict:
    """Accumulate saliency for one block from its captured router calls.

    Tokens are grouped by expert across the whole capture before any expert
    weight is loaded: each expert's 3 x 8.4M parameters are then dequantized
    once instead of once per file.
    """
    files = sorted(p for p in capture_dir.glob(f"L{layer}-*.pt"))
    if not files:
        return {}
    log(f"block {block}: {len(files)} captured router call(s)")

    rows_by_expert: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    hidden: list[torch.Tensor] = []
    for index, path in enumerate(files):
        blob = torch.load(path, weights_only=True)
        hidden.append(blob["hidden"])
        ids, gates = blob["ids"], blob["gates"]
        for token in range(ids.shape[0]):
            for slot in range(ids.shape[1]):
                rows_by_expert[int(ids[token, slot])].append(
                    (index, token, float(gates[token, slot])))

    tokens = sum(h.shape[0] for h in hidden)
    log(f"block {block}: {tokens:,} token(s), "
        f"{len(rows_by_expert)} expert(s) touched of {n_experts}")

    experts = Experts(src, block, device)
    total = torch.zeros(n_experts, dtype=torch.float64)
    count = torch.zeros(n_experts, dtype=torch.float64)

    for order, (expert, rows) in enumerate(sorted(rows_by_expert.items()), start=1):
        x = torch.stack([hidden[i][t] for i, t, _ in rows]).to(device, torch.bfloat16)
        gate = torch.tensor([g for _, _, g in rows], dtype=torch.float64)
        w1, w3, w2 = experts.weights(expert)
        out = (torch.nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
        norms = out.float().norm(dim=-1).double().cpu()
        total[expert] = (gate * norms).sum()
        count[expert] = len(rows)
        del w1, w3, w2, out
        if order % 32 == 0:
            log(f"  block {block}: {order}/{len(rows_by_expert)} experts scored")

    mean = torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))
    return {"mean": mean.tolist(), "sum_saliency": total.tolist(),
            "count": count.tolist(), "tokens": tokens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, default=REPO / "artifacts/mtp-capture")
    ap.add_argument("--src", type=Path, default=REPO / "models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/mtp-saliency.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    n_experts = config["n_routed_experts"]

    layers = sorted({int(FILE.match(p.name).group(1)) for p in args.capture.glob("L*.pt")
                     if FILE.match(p.name)})
    if not layers:
        print(f"FAIL: no captures under {args.capture}", file=sys.stderr)
        return 1
    log(f"captured layers {layers} -> MTP blocks {list(range(len(layers)))}")

    # Record what the capture was, not just what it said. Whether the server ran
    # eager decides whether these numbers mean anything at all -- with CUDA
    # graphs on, the probe sees only the handful of rows that run before capture
    # -- and the file outlives the shell history that would otherwise say so.
    files = sorted(args.capture.glob("L*.pt"))
    out = {"run": {"capture": str(args.capture), "n_routed_experts": n_experts,
                   "capture_files": len(files),
                   "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
                       max((p.stat().st_mtime for p in files), default=0))),
                   "requires": "server run with --enforce-eager; a Python probe "
                               "cannot observe a replayed CUDA graph"},
           "blocks": {}}
    for block, layer in enumerate(layers):
        started = time.perf_counter()
        scored = score_block(args.capture, layer, args.src, block, n_experts,
                             args.device)
        if scored:
            out["blocks"][str(block)] = scored
            log(f"block {block} scored in {(time.perf_counter() - started) / 60:.1f} min")
            args.out.write_text(json.dumps(out, indent=1))

    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
