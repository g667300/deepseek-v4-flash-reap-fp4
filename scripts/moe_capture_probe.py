#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Capture what the draft head's FFNs actually see, from inside the served model.

Mounted into the serving container as ``sitecustomize.py``.

Scoring the draft head's experts needs REAP's ``S_j = mean(g_j * ||f_j||_2)``,
and the first attempt computed it offline from the main model's hidden states.
That failed a check it was built to fail: against the routing the real drafter
does, the offline version scored **+0.225** rank correlation and touched 186 of
256 experts where vLLM touches 74. The draft head's router reads the hidden
state *after the block's attention*, and skipping that attention routes to a
different distribution entirely.

So capture the router's own input instead. ``BaseRouter._select_experts``
receives the hidden states and returns ``(topk_weights, topk_ids)`` -- the three
quantities REAP's formula needs, at the exact point the real model computes
them. The expert arithmetic is then done offline from the checkpoint's weights,
where it is cheap and exact; only the activations have to come from here.

Captures **only the draft head's layers**, identified by their expert count
(``global_num_experts`` = ``MOE_CAPTURE_EXPERTS``, 256 when the head is carried
unpruned): the main stack is already scored by REAP itself, and capturing all 46
layers would write two orders of magnitude more data for no use.

**Everything here is written to say why it produced nothing.** The first version
captured zero files and left no trace, because it swallowed the first exception
into a file the driver then deleted along with the warmup captures, and set a
one-shot flag that suppressed every later report. So: diagnostics live *outside*
``MOE_CAPTURE_DIR`` where nothing sweeps them, errors are recorded once per
distinct type rather than once per process, and ``probe-status.json`` records
how many times each router width was seen whether or not anything was captured.
A run that captures nothing must still be able to tell you which of "the hook
never fired", "the width never matched" and "every call raised" happened.

**Serve with ``--enforce-eager``.** A Python hook cannot see a replayed CUDA
graph: once vLLM captures the decode path, the graph runs on the device and this
function is never called again. The symptom is exact -- requests answer normally
while the call counter sits still -- and it is also what limited the earlier
usage measurement, where the draft head recorded 1,170 selections against the
main stack's 28,212 for the same tokens. Eager costs decode speed, which does
not matter for a calibration pass over 32 samples.

Environment:
    MOE_CAPTURE_DIR      where to write (default /probe/capture)
    MOE_CAPTURE_EXPERTS  only capture routers with this many experts (default 256)
    MOE_CAPTURE_STRIDE   keep every Nth token (default 1 -- the draft head sees
                         few tokens per step, so thinning is usually unnecessary)
    MOE_CAPTURE_MAX      stop after this many files per layer (default 4000)
"""

from __future__ import annotations

import json
import os
import threading

_DIR = os.environ.get("MOE_CAPTURE_DIR", "/probe/capture")
_EXPERTS = int(os.environ.get("MOE_CAPTURE_EXPERTS", "256"))
_STRIDE = int(os.environ.get("MOE_CAPTURE_STRIDE", "1"))
_MAX = int(os.environ.get("MOE_CAPTURE_MAX", "4000"))
# Diagnostics sit beside the capture directory, not inside it: the driver clears
# the captures between warmup and the real run, and taking the error log with
# them is what hid the last failure.
_DIAG = os.path.dirname(_DIR.rstrip("/")) or "/tmp"


def _install() -> None:
    try:
        import torch
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    except Exception as exc:
        try:
            with open(os.path.join(_DIAG, "probe-error.txt"), "a") as f:
                f.write(f"install failed: {type(exc).__name__}: {exc}\n")
        except Exception:
            pass
        return

    os.makedirs(_DIR, exist_ok=True)
    lock = threading.Lock()
    ordinals: dict[int, int] = {}
    written: dict[int, int] = {}
    seen: dict[str, int] = {}
    errors: dict[str, str] = {}
    skips: dict[str, int] = {}
    calls = [0]
    saved = [0]
    original = BaseRouter._select_experts

    def status() -> None:
        try:
            with open(os.path.join(_DIAG, "probe-status.json"), "w") as f:
                json.dump({"calls": calls[0], "routers_by_width": seen,
                           "dir": _DIR, "saved": saved[0], "skipped": skips,
                           "files_by_layer": {str(k): v for k, v in written.items()},
                           "want_experts": _EXPERTS,
                           "errors": errors}, f, indent=1)
        except Exception:
            pass

    def skip(reason: str) -> None:
        with lock:
            skips[reason] = skips.get(reason, 0) + 1

    def note(where: str, exc: BaseException) -> None:
        key = f"{where}:{type(exc).__name__}"
        with lock:
            if key in errors:
                return
            errors[key] = str(exc)[:400]
        status()

    def pick_hidden(args, kwargs):
        """The router's input, not its logits.

        Both are float tensors on the call, and which one arrives first has
        changed between vLLM versions, so discriminate on shape instead of
        position: router logits are ``[tokens, global_num_experts]``, hidden
        states are ``[tokens, hidden_size]``.
        """
        cand = kwargs.get("hidden_states")
        if cand is not None:
            return cand
        for item in args:
            if hasattr(item, "dtype") and getattr(item.dtype, "is_floating_point", False) \
                    and item.dim() >= 2 and item.shape[-1] != _EXPERTS:
                return item
        return None

    def select_experts(self, *args, **kwargs):      # noqa: ANN001, ANN002, ANN003
        out = original(self, *args, **kwargs)
        try:
            width = getattr(self, "global_num_experts", None)
            with lock:
                calls[0] += 1
                key = str(width)
                seen[key] = seen.get(key, 0) + 1
                due = calls[0] % 200 == 0
            if due:
                status()
            if width != _EXPERTS:
                return out
            # torch.save synchronises, which is illegal mid-graph-capture and
            # would abort the capture rather than merely fail here.
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                skip("graph-capture")
                return out

            weights = ids = None
            for item in out if isinstance(out, (tuple, list)) else ():
                if not hasattr(item, "dtype"):
                    continue
                if item.dtype.is_floating_point:
                    weights = item
                else:
                    ids = item
            if ids is None or weights is None or ids.numel() == 0:
                note("extract", ValueError(f"out={type(out).__name__} "
                                           f"ids={ids is not None} w={weights is not None}"))
                return out

            x = pick_hidden(args, kwargs)
            if x is None:
                note("hidden", ValueError(f"args={[tuple(a.shape) for a in args if hasattr(a,'shape')]} "
                                          f"kwargs={sorted(kwargs)}"))
                return out
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])

            with lock:
                layer = ordinals.setdefault(id(self), len(ordinals))
                count = written.get(layer, 0)
                if count >= _MAX:
                    skip("per-layer-max")
                    return out
                written[layer] = count + 1

            # ids/gates are [tokens, top_k]; during speculation the router runs
            # over a different token count than a plain decode step, so trust
            # their own leading dimension rather than the hidden states'.
            rows = min(int(ids.shape[0]), int(x.shape[0]))
            keep = slice(None, None, _STRIDE)
            torch.save(
                {"hidden": x[:rows][keep].detach().to("cpu", torch.bfloat16),
                 "ids": ids[:rows][keep].detach().cpu(),
                 "gates": weights[:rows][keep].detach().to("cpu", torch.float32)},
                os.path.join(_DIR, f"L{layer}-{count:05d}.pt"),
            )
            with lock:
                saved[0] += 1
        except Exception as exc:    # never break serving, but always say so
            note("capture", exc)
        return out

    BaseRouter._select_experts = select_experts

    import atexit
    atexit.register(status)
    status()


_install()
