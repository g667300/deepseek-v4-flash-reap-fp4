# deepseek-v4-flash-reap-fp4

The pipeline that produced
[**DeepSeek-V4-Flash-REAP128-FP4**](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP128-FP4)
and its 152-expert sibling: REAP expert pruning of
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
that stays in DeepSeek's native FP4/FP8 format **and keeps the DSpark draft
head**, so the result still speculates.

Every published number came from the code in here, and the point of publishing
it is that you can check them rather than believe them.

| | |
|---|---|
| base | `deepseek-ai/DeepSeek-V4-Flash-0731`, 256 routed experts, 166.9 GB |
| published builds | 128 experts / 88.1 GB, and 152 experts / 101.9 GB |
| format | unchanged: FP4 E2M1 experts, FP8 E4M3 elsewhere, native tensor names |
| draft head | all three MTP blocks carried across, pruned by their own calibration |
| target hardware | one DGX Spark (GB10, 128 GB unified memory) |

## Start here

[`scripts/README.md`](scripts/README.md) is the map: what each script does, the
two ways to build a checkpoint, the traps that cost real runs, and the commands
that reproduce the published evaluations.

The measurements live with the models, not here — quality
([QUALITY.md](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP128-FP4/blob/main/docs/QUALITY.md)),
serving
([SERVING.md](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP128-FP4/blob/main/docs/SERVING.md))
and provenance
([PROVENANCE.md](https://huggingface.co/noooop/DeepSeek-V4-Flash-REAP128-FP4/blob/main/docs/PROVENANCE.md)) —
along with the expert selections themselves, which are what you need to rebuild
a published checkpoint without recalibrating.

## Two ways to build it, and they agree

**The reference path** dequantizes to BF16, runs llm-compressor's own
`REAPPruningModifier` over all 43 MoE layers, re-encodes to FP4/FP8 and carries
the MTP blocks across. It needs a GPU, a 568.7 GB intermediate and about 100
minutes.

**The byte-copy path** does the same thing in 84 seconds on a CPU. REAP never
modifies a surviving expert, so expert slot *i* of a pruned model is
byte-for-byte the reference checkpoint's expert `retained[i]`; only 86 router
tensors have to be computed. Rebuilding the 128-expert model both ways gave
**17,711 tensors, 0 differing in value** — the only byte differences are FP4
negative zero against positive zero, which E2M1 decodes identically.

So a different sparsity is a byte copy plus a top-k over the published saliency,
not another calibration run:

```bash
python scripts/build_pruned.py --reference /path/to/DeepSeek-V4-Flash-0731 \
    --saliency target-saliency.json --experts 160 --out dsv4-reap160
python scripts/carry_mtp.py --src /path/to/DeepSeek-V4-Flash-0731 \
    --dst dsv4-reap160 --score saliency --saliency mtp-saliency.json
```

Changing the calibration *mixture* is the expensive case, and that is what the
reference path is for.

## The draft head needs its own calibration

DeepSeek-V4-Flash's DSpark head is three MTP blocks with 256 experts each, and
`transformers` drops them on load, so REAP never scores them. Every static proxy
tried here — a target layer's retained set, router row norms, the gate bias —
landed within 1.3 points of the others and in a different order per workload.
What worked was measuring the head itself: capture its router's own inputs from
inside the serving model (`moe_capture_probe.py`, **eager** — a Python hook
cannot see a replayed CUDA graph), then score with REAP's own formula offline.
That is worth about +1.4 points of acceptance overall and +5 on code, for free.

Cutting the head to its 64 most-used experts, by contrast, produced a model that
generated **nothing**: 5,110 tokens drafted, 0 accepted. Usage is not saliency.

## What is not here

* **`calib.pt`** — 3.1 MB of token IDs from third-party corpora, two of them
  share-alike licensed. [`calib/`](calib) has the exact mixture that rebuilds
  it, and the published models carry its SHA-256 and the resulting selections.
* **The evaluation corpora**, for the same reason.
* **Weights.** They are on the Hub, with per-file checksums.

## Layout

| | |
|---|---|
| [`scripts/`](scripts) | the build, calibration, verification and evaluation code |
| [`patches/`](patches) | three vLLM 0.25.1 overlays that SM120/SM121 needs, with their upstream Apache-2.0 headers and a notice of what was changed |
| [`calib/`](calib) | the calibration mixture, the perplexity holdout mixture, and a two-language example to copy |

## Licence

MIT, except `patches/`, which are modified copies of vLLM sources under
Apache-2.0 — each keeps its SPDX header and states what was changed. The
sparse-MLA page re-view inside `patches/flashinfer_sparse.py` is derived from
[`anemll/dspark-vllm-gx10`](https://github.com/anemll/dspark-vllm-gx10) (MIT)
and is attributed at its use site. The model weights inherit DeepSeek's MIT
licence.
