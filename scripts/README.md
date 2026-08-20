# Build and evaluation scripts

Everything used to produce and check the published builds, so a third party
can rebuild them, build a different one, or re-run the numbers in
[`../docs/QUALITY.md`](../docs/QUALITY.md) instead of taking them on trust.

**This is a snapshot.** The canonical copy of these scripts, with its history,
is [`g667300/deepseek-v4-flash-reap-fp4`](https://github.com/g667300/deepseek-v4-flash-reap-fp4) on GitHub; they are shipped with
the weights so a download is self-contained.

These are the working scripts, published as they were run. They are MIT licensed
(see [`../LICENSE-CODE`](../LICENSE-CODE)), carry no support promise, and their
defaults name the author's own paths (`models/…`, `artifacts/…`) — pass the paths
explicitly rather than relying on a default.

Nothing here needs a checkout of the private development repository, with one
documented exception: the two `eval_code_challenge.py` suites drive the official
LiveCodeBench and BigCodeBench evaluators, which you supply with `--lcb-root`
and `--bcb-root`.

## There are two ways to build a checkpoint, and they agree

| | reference path | byte-copy path |
|---|---|---|
| scripts | `dequantize_checkpoint.py` → `reap_reference.py` → `quantize_to_deepseek.py` → `carry_mtp.py` | `build_pruned.py` → `carry_mtp.py` |
| needs | GPU, 568.7 GB BF16 copy, 272 GB REAP output, ~100 min | the official checkpoint and a saliency file; CPU only, **84 s** |
| gives | any sparsity, calibrating from scratch | any sparsity, from the recorded saliency |
| used for | the REAP128 build | the REAP152 build |

The second is not an approximation. REAP rebuilds the expert list from the
survivors and slices the router; it never modifies a surviving expert, so expert
slot *i* of a pruned model is byte-for-byte reference expert `retained[i]` and
its quantized bytes already exist in the official checkpoint. Rebuilding the
128-expert model both ways gave **17,711 tensors, 0 different in value** — the
only byte differences are FP4 nibble `8` against `0`, negative zero against
positive zero, which E2M1 decodes identically.

Only 86 tensors are computed rather than copied: 43 `ffn.gate.weight` and 40
`ffn.gate.bias` (row selection), plus `tid2eid` on the three hash-routed layers,
whose remap needs the *unpruned* router rows.

**Use the byte-copy path unless you are changing the calibration mixture.** The
reference path exists because it is the reference — it is what the published
saliency was measured with, not what you have to re-run to get a checkpoint.

## Environment

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Version-sensitive, and not incidentally:

* **`transformers==5.14.1` cannot save an offloaded sharded model** without a
  one-character source fix. `modeling_utils.py`'s `save_pretrained` calls
  `weight_map.update({k: basename} for k in …)` — a generator of dicts handed to
  `dict.update()` — which raises `ValueError` for every non-empty shard. A bare
  `except Exception` around it re-raises the misleading advice "Try reducing
  `max_shard_size`"; no shard size works. Change the generator to a dict
  comprehension `{k: basename for k in …}`. Without it, `reap_reference.py` dies
  *after* the hour of calibration.
* `torchvision` is in `requirements.txt` although nothing imports it directly:
  `transformers`' patch machinery walks `dir(transformers)`, which imports
  aria's image processor, which imports `torchvision` at module scope.
* `test_reference_reap.py` is the guard for both. Run the tests before starting
  a long run.

## 0. Calibration input

`calib.pt` is not distributed — see [`../calibration/README.md`](../calibration/README.md)
for why, and for the licences of the sources. Rebuild it from the shipped mixture:

```bash
python scripts/build_calibration.py --mix calibration/mix-dsv4.json \
    --tokenizer /path/to/DeepSeek-V4-Flash-0731 --out calib.pt
python scripts/build_calibration.py --mix calibration/mix-dsv4.json \
    --tokenizer /path/to/DeepSeek-V4-Flash-0731 --probe   # reachability only
```

`--tokenizer` must be the *official checkpoint directory*, not a tokenizer-only
copy: DeepSeek-V4-Flash ships no HF `chat_template`, and `model_profiles.py`
renders the instruction and chat sources through the checkpoint's own
`encoding/encoding_dsv4.py`.

The mixture pins 512 samples × 2048 tokens and seed 0, but not immutable dataset
revisions, so a later upstream dataset edit can legitimately change the digest in
`calibration/calib.pt.sha256`. Treat a mismatch as a different calibration run.

`label_calibration.py` reconstructs which source each 2048-token sample came
from, which is what makes a *per-source* saliency run possible:

```bash
python scripts/label_calibration.py --mix calibration/mix-dsv4.json \
    --tokens calib.pt --out calib-sources.json
```

## 1–4. The reference path

```bash
# 1. dequantize to BF16, tensor names untouched        (568.7 GB, CPU, disk-bound)
python scripts/dequantize_checkpoint.py \
    --src /path/to/DeepSeek-V4-Flash-0731 --dst dsv4-bf16

# 2. the reference REAP run                            (272 GB out, GPU)
python scripts/reap_reference.py \
    --model dsv4-bf16 --tokens calib.pt --sparsity 0.5 --out dsv4-reap50-bf16

# 3. re-encode into DeepSeek's native format and names (82.4 GB, CPU)
python scripts/quantize_to_deepseek.py \
    --src dsv4-reap50-bf16 --dst dsv4-reap50 \
    --reference /path/to/DeepSeek-V4-Flash-0731

# 4. carry the three MTP blocks across, pruned to match
python scripts/carry_mtp.py \
    --src /path/to/DeepSeek-V4-Flash-0731 --dst dsv4-reap50 \
    --score saliency --saliency calibration/mtp-saliency.json
```

Measured on this checkpoint: step 2 61 min, step 3 28 min with 3 workers, step 4
three seconds — it copies bytes rather than re-encoding them.

Four things about this path that are easy to get wrong:

* **Step 2 renames everything.** Of 257 tensors in a synthetic checkpoint,
  exactly one name survives a `llmcompressor` load-and-save round trip, and
  `save_original_format=True` does not restore the rest. Step 3 therefore does a
  full rename back to DeepSeek's names, not a spot fix.
* **Step 2 flattens ~400 fp32 tensors to bf16**, because it loads with
  `dtype=torch.bfloat16`. The one that can change behaviour is
  `ffn.gate.bias`, added to the router scores before top-k, which bf16 moves by
  up to 0.031 — and non-uniformly, since layers 0-2 stay resident in fp32 while
  the rest are restored from the bf16 offload cache. Step 3 takes those families
  from the reference instead and re-slices the bias from the reference's fp32.
  `--no-restore-untouched` turns that off and reproduces the reference path's
  rounding exactly.
* **Step 2 has nothing to verify against**, since REAP re-encodes every weight.
  Steps 1, 3 and 4 do; see below.
* `reap_reference.py` journals its retained set after every subgraph
  (`--retained-out`), because a multi-hour run that dies should not be repeated
  from the beginning. `--calibrate-only --saliency-out` produces the saliency
  file without writing a checkpoint; `--sample-sources` splits it per source.

## The byte-copy path

```bash
# any expert count, from the published saliency
python scripts/build_pruned.py --reference /path/to/DeepSeek-V4-Flash-0731 \
    --saliency calibration/target-saliency.json --experts 152 --out dsv4-reap152
# or from an exact recorded selection
python scripts/build_pruned.py --reference /path/to/DeepSeek-V4-Flash-0731 \
    --retained calibration/reap128-retained-sets.json --out dsv4-reap128
python scripts/carry_mtp.py --src /path/to/DeepSeek-V4-Flash-0731 \
    --dst dsv4-reap152 --score saliency --saliency calibration/mtp-saliency.json
```

`remix_saliency.py` reweights a **source-split** saliency file into a retained
set — the shipped `target-saliency.json` holds combined totals only, so it can
change the expert count for this mixture but cannot be reweighted into another
language or task profile:

```bash
python scripts/remix_saliency.py --saliency saliency-de-en.json \
    --sources calib-de-en-sources.json --weights de-wiki=0.7,en-wiki=0.3 \
    --experts 128 --out retained-de-en-128.json
```

`new_mtp_variant.py` makes a variant that differs only in its draft head by
hardlinking the shared shards and replacing the MTP ones — 5.7 GB instead of
88 GB. **It unlinks metadata files before writing them**, and that is not
cosmetic: `cp -al` links `model.safetensors.index.json` too, and Python's
`write_text()` truncates the *inode*, so editing the copy's index rewrites the
original's. That cost the REAP128 checkpoint its MTP entries once.

## Verifying a build

```bash
# step 1: every tensor must equal what dequantizing the source pair produces
python scripts/verify_dequantized.py --src /path/to/DeepSeek-V4-Flash-0731 \
    --dst dsv4-bf16 --journal verify-bf16.journal

# steps 3-4: quantized weights must decode back to step 2 bit for bit, plain
# tensors must equal the reference, carried MTP tensors must equal it exactly
python scripts/verify_quantized.py --src dsv4-reap50-bf16 --dst dsv4-reap50 \
    --reference /path/to/DeepSeek-V4-Flash-0731 \
    --retained retained-sets.json --journal verify-final.journal
```

Both re-read a mismatch before reporting it, so a bad read is not mistaken for a
bad checkpoint. That is not defensive programming for its own sake: the machine
these were built on had a memory fault that put **three wrong tensors on disk**
in one pass over the BF16 copy, and the verify pass is what found them. Both
scripts record the byte lane of every differing bit for the same reason.

`verify_quantized.py` refuses to check carried MTP tensors unless the output
config records `_mtp_pruned_with_layer` or the saliency selection, because
nothing else says which source expert a slot came from.

## Calibrating the draft head

The DSpark head's experts **cannot** be chosen by proxy. Measured end to end,
three static proxies (a layer's retained set, router row norm, lowest gate bias)
land within 1.3 points of each other and in a different order per workload,
while REAP's own formula computed against the head's *live* router inputs gains
+1.4 points overall and +5.0 on code. Two facts behind that: the MTP experts are
not the main stack's experts renumbered (cosine 0.012 at best against all 256 of
layer 42), and scoring them offline from the main model's hidden states
correlates only +0.225 with what the real head routes, because its router reads
the state *after its own block's attention*.

So capture from inside the serving model:

```bash
# 1. serve with the probe mounted as sitecustomize.py -- EAGER, see below
docker run -d --gpus all --ipc=host -p 8000:8000 \
  -v /path/to/checkpoint:/model:ro \
  -v $PWD/scripts/moe_capture_probe.py:/probe/sitecustomize.py:ro \
  -v /tmp/probe:/probe-out \
  -e PYTHONPATH=/probe -e MOE_CAPTURE_EXPERTS=128 -e MOE_CAPTURE_DIR=/probe-out/capture \
  <vllm-image> serve /model --served-model-name dsv4-cap \
  --gpu-memory-utilization 0.85 --max-model-len 4096 --max-num-seqs 6 \
  --kv-cache-dtype fp8_ds_mla --enforce-eager \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'

# 2. drive it with the calibration tokens, then check the probe actually fired
python scripts/replay_calibration.py --url http://HOST:8000 --model dsv4-cap \
    --tokens calib.pt --samples 160 --prefix 1024 --generate 128
cat /tmp/probe/probe-status.json     # calls per router width, files, skips, errors

# 3. score with REAP's own S_j = mean(g_j * ||f_j||_2), then build
python scripts/score_mtp_captured.py --capture mtp-capture \
    --src /path/to/DeepSeek-V4-Flash-0731 --out mtp-saliency.json
python scripts/new_mtp_variant.py --src dsv4-reap50 --dst dsv4-reap50-salmean
python scripts/carry_mtp.py --src /path/to/DeepSeek-V4-Flash-0731 \
    --dst dsv4-reap50-salmean --score saliency --saliency mtp-saliency.json
```

Four traps, each of which cost a run:

* **`--enforce-eager` is mandatory.** Once vLLM captures the decode path into a
  CUDA graph, `BaseRouter._select_experts` never runs in Python again. The
  symptom is silent: requests answer 200 while the probe's counter does not
  move. Same prompts, graphs on: 195 routed rows. Eager: 3,930 per block from 32
  samples, and 19,970 from 96. Always check the call counter against the request
  count before believing a distribution.
* **`probe-status.json` is written outside `MOE_CAPTURE_DIR`** on purpose: the
  driver clears the capture directory between warmup and the real run, and
  taking the error log with it hid the first failure for two runs.
* **Replay the calibration tokens**, not arbitrary prompts. Scoring the head on
  a different distribution than the stack around it selects different experts.
* **Check convergence with an even/odd split of the capture** before building.
  At 3,930 routed tokens per block the top-128 set reproduced itself only 82-84%
  across the split; at 19,970 it reached 87-94% (`sum_saliency` 94-97%).
* **Capture with the head unpruned** (`MOE_CAPTURE_EXPERTS=256`, built by
  `carry_mtp.py --experts 256`). Capturing a 128-expert head only measures the
  128 already chosen; the selection cannot be redone from it.
* **A larger target may not fit at 0.85.** A 152-expert target with the 256-expert
  head loads 100.37 GiB and leaves *negative* KV room there. What worked was 0.87
  with `--max-num-seqs 2 --max-num-batched-tokens 2048 --max-model-len 2048`,
  which yielded 1.71 GiB of cache — the batch settings do more than the
  utilization does, and 0.9 is the line that hangs this host.
* **Never delete the capture directory while the server runs.** The probe does not
  recreate it: every save then fails with `Parent directory … does not exist`,
  the run records 12,000 router calls and writes nothing, and the engine stops
  answering. Clear the *files* and keep the directory.

`mean` is the default because REAP itself selects on `mean_saliency`;
`--saliency-key sum_saliency` is kept because the two disagree here (sharing
only 84-87 of 128 experts) while scoring the same, 41.3% against 41.1%.

Two things not to do. Do not select the head's experts by **usage**: on the main
stack, where REAP's scores are known, usage rank and saliency rank correlate at
−0.142 mean Spearman. And do not cut the head to 64 experts — that build
generated *nothing*, 5,110 drafted and 0 accepted, because `mtp.0.ffn.gate.bias`
has mean 3.015 with standard deviation 0.088, so `noaux_tc` selection is nearly
bias-ordered and the same six experts win every time.

## Serving

`serve_patched.sh` is the recipe that produced the speculative numbers, written
for the author's two-machine layout (it rsyncs to a Spark and mounts the
overlays from `artifacts/patches/`). Read it as a recipe; the canonical launch
commands are in [`../docs/SERVING.md`](../docs/SERVING.md), and the overlays
themselves in [`../patches/`](../patches).

```bash
python scripts/check_serving.py --url http://HOST:8000 --model dsv4 --bench
```

`check_serving.py --bench` is the acceptance measurement: drafted, accepted,
acceptance rate and accepted-by-position. Its prompts are short; the published
acceptance figures use the 512-token DSpark benchmark, and a 1.3-point spread on
short prompts is noise.

## Reproducing the published evaluations

Every number in [`../docs/QUALITY.md`](../docs/QUALITY.md) comes from these,
against a running OpenAI-compatible server:

```bash
# generative MMLU / global_MMLU -- works against vLLM and llama-server alike
python scripts/eval_generative.py --url http://HOST:8000/v1 --model dsv4 \
    --tasks mmlu --limit 205 --out gen-mmlu.json
python scripts/eval_generative.py --url http://HOST:8000/v1 --model dsv4 \
    --tasks global_mmlu:es,global_mmlu:fr,global_mmlu:zh,global_mmlu:ja,global_mmlu:hi \
    --limit 200

# held-out perplexity: build the holdout first, with THIS tokenizer
python scripts/build_calibration.py --mix calibration/ppl-holdout-dsv4.json \
    --tokenizer /path/to/DeepSeek-V4-Flash-0731 --out ppl-holdout.pt
python scripts/eval_perplexity.py --data ppl-holdout.pt \
    --base-url http://HOST:8000/v1 --model dsv4 --out ppl.json

# code: 20 older LiveCodeBench/BigCodeBench tasks, and 6 post-release AtCoder ones
python scripts/eval_code_challenge.py generate --url http://HOST:8000/v1 --model dsv4 --tag mine
python scripts/eval_code_challenge.py score --tag mine \
    --lcb-root /path/to/LiveCodeBench --bcb-root /path/to/bigcodebench
python scripts/eval_code_fresh.py generate --url http://HOST:8000/v1 --model dsv4 --tag mine
python scripts/eval_code_fresh.py score --tag mine

# the lm-eval battery (JCommonsenseQA, RULER, log-likelihood MMLU)
MODEL=dsv4 TOKENIZER=/path/to/DeepSeek-V4-Flash-0731 TAG=mine \
  PPL_DATA=ppl-holdout.pt scripts/run_eval_suite.sh
```

Notes that decide whether your numbers are comparable:

* **discard the first run after a server load.** On vLLM it is **1.66-1.80x
  slower** than steady state — measured on three checkpoints — while the
  speculative counters come out bit-identical, because greedy decoding fixes the
  tokens and only wall-clock time moves. llama.cpp shows a 1.04x version of the
  same effect. A cold vLLM number next to a warm llama.cpp one distorts the
  comparison by more than the difference being measured, and this repository
  published exactly that mistake once.

* **the perplexity holdout must be built with this model's tokenizer.** It is
  token IDs, and the mixture (`calibration/ppl-holdout-dsv4.json`, seed 999, 128
  samples) is disjoint from the calibration set by seed only. A holdout from
  another model showed up as a bare HTTP 400 several stages into a run.
* `eval_generative.py` scores by **generation**, not log-likelihood, so that a
  GGUF served by llama-server can be scored the same way — `llama-server`
  ignores `echo` and cannot return prompt logprobs. The two disagree
  substantially on this model (log-likelihood MMLU 64.04% against generative
  51.22%), so never compare one against the other.
* `run_eval_suite.sh` expects a virtualenv at `../.venv` relative to itself and
  a `local-completions` lm-eval backend; JCommonsenseQA needs its dataset placed
  where `eval_tasks/jcommonsenseqa_local/jcommonsenseqa_local.yaml` points. Skip
  stages with `SKIP="ruler ppl"`.
* **never edit a shell script while it is running.** bash reads it by byte
  offset and will resume mid-line at whatever now sits there. Editing
  `run_eval_suite.sh` during a four-hour run ended it with a syntax error on a
  file that is perfectly valid on disk. Copy it, edit the copy.

## Tests

```bash
python scripts/test_reference_reap.py   # REAP end to end on a synthetic DSv4
python scripts/test_names.py            # the step-2 -> native rename
python scripts/test_quant_encode.py     # FP4/FP8 encode round trip
```

`test_reference_reap.py` pins `CUDA_VISIBLE_DEVICES=""` and must stay CPU-only:
one check relies on an out-of-bounds index raising a catchable `IndexError`,
which on a GPU becomes a device-side assert that poisons the CUDA context for
every later check. It also covers the offloaded save, which is the guard for the
`transformers` fix above.

## File map

| file | role |
|---|---|
| `build_calibration.py`, `model_profiles.py` | build `calib.pt` from a mixture; per-model prompt rendering and tensor-name knowledge |
| `label_calibration.py` | recover each sample's source, for a per-source saliency run |
| `dequantize_checkpoint.py` | step 1: FP4/FP8 → BF16, names untouched |
| `verify_dequantized.py` | step 1 check against the source pair |
| `reap_reference.py` | step 2: `llmcompressor`'s REAP modifier over the BF16 copy |
| `dsv4_patches.py` | what DeepSeek-V4 needs before that modifier will run on it |
| `quantize_to_deepseek.py` | step 3: back to native FP4/FP8 and native names |
| `names.py` | the step-2 → DeepSeek name mapping |
| `quant.py`, `nvfp4.py` | FP4/FP8 encode and decode |
| `verify_quantized.py` | steps 3-4 check against step 2 and the reference |
| `carry_mtp.py` | step 4: carry the three MTP blocks, pruned |
| `build_pruned.py` | the whole prune as a byte copy from the official checkpoint |
| `remix_saliency.py` | reweight a source-split saliency file into a retained set |
| `new_mtp_variant.py` | a draft-head-only variant, hardlinked |
| `moe_capture_probe.py` | capture the draft head's router inputs inside vLLM |
| `replay_calibration.py` | drive the served model with the calibration tokens |
| `score_mtp_captured.py` | REAP's formula over that capture |
| `serve_patched.sh` | the speculative serving recipe |
| `check_serving.py` | generation smoke test and draft-acceptance benchmark |
| `eval_generative.py` | generative MMLU / global_MMLU |
| `eval_perplexity.py` | held-out perplexity through `/v1/completions` |
| `eval_code_challenge.py` | 10 LiveCodeBench + 10 BigCodeBench tasks |
| `eval_code_fresh.py` | 6 post-release AtCoder tasks, generated test inputs |
| `run_eval_suite.sh`, `eval_tasks/` | the lm-eval battery and the local JCommonsenseQA task |
| `test_reference_reap.py`, `test_names.py`, `test_quant_encode.py` | the guards |
| `requirements.txt` | pinned to what these were actually run with |

## What is deliberately not here

* **`calib.pt` and the evaluation data.** Both are third-party corpora in token
  form; two calibration sources are share-alike licensed. The recipes and the
  digests are published instead.
* **The BF16 intermediate and the REAP output.** 568.7 GB and 272 GB, and the
  byte-copy path does not need them.
* **Machine-specific tooling.** The memory-fault investigation, lane histograms
  and host-level notes stay in the development repository; what survives here is
  the reason the verify scripts re-read before reporting.
