#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
# The evaluation battery against one served model, cheapest stage first so a run
# cut short still leaves a comparison table. Adapted from the companion repo's
# suite so the numbers land beside its DSv4 REAP-50 baseline, which was measured
# with the same tasks, the same held-out ids and the same harness.
#
#   MODEL=dsv4-eval TOKENIZER=models/DeepSeek-V4-Flash-0731 TAG=salmean \
#     PPL_DATA=artifacts/ppl-holdout-dsv4.pt scripts/run_eval_suite.sh
#
# Skip stages with SKIP="ruler ppl" (names: jcqa mmlu multilingual ppl ruler).
set -u

VENV=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin
BASE="${BASE:-http://192.168.100.2:8000/v1/completions}"
MODEL="${MODEL:?set MODEL to the served model name}"
TOKENIZER="${TOKENIZER:?set TOKENIZER to a local tokenizer path}"
TAG="${TAG:?set TAG, used in output filenames}"
# local-completions is the OpenAI-compatible backend, so it drives vLLM and
# llama-server alike -- both expose /v1/completions with echo+logprobs, which
# is what the loglikelihood tasks need. Overridable for a stack that does not.
BACKEND="${BACKEND:-local-completions}"
CONCURRENT="${CONCURRENT:-16}"
TIMEOUT="${TIMEOUT:-300}"
SKIP="${SKIP:-}"
MAXLEN="${MAXLEN:-65536}"

# PPL_DATA has no default on purpose: it must be held-out ids built with *this*
# model's tokenizer. The companion repo learned that the hard way -- a fallback
# to another model's file showed up only as a bare 400 several stages in.
# Keep the message free of apostrophes: inside ${var:?word} bash starts a quote
# on one even within double quotes, and the script then fails to parse at EOF
# with a line number pointing at the end of the file rather than at this line.
PPL_DATA="${PPL_DATA:?set PPL_DATA to held-out ids built with the tokenizer of this model}"

# Long context is a different regime. lm_eval's timeout is a *total* per request,
# queue wait included, so sixteen in flight leaves ten queued past the default
# and the client kills a run the server is still working through.
RULER_CONCURRENT="${RULER_CONCURRENT:-8}"
RULER_TIMEOUT="${RULER_TIMEOUT:-1800}"

ARGS="model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${CONCURRENT},max_retries=3,max_length=${MAXLEN},timeout=${TIMEOUT}"
RULER_ARGS="model=${MODEL},base_url=${BASE},tokenizer=${TOKENIZER},num_concurrent=${RULER_CONCURRENT},max_retries=3,max_length=${MAXLEN},timeout=${RULER_TIMEOUT}"

run_stage() {
  local name="$1"; shift
  case " $SKIP " in *" $name "*) echo "#### skip ${name}"; return;; esac
  echo "#### ${name} started $(date +%H:%M:%S)"
  "$@"
  echo "#### ${name} finished rc=$? at $(date +%H:%M:%S)"
}

run_stage jcqa "$VENV/lm_eval" run --model "$BACKEND" \
  --model_args "$ARGS" --tasks jcommonsenseqa_local \
  --include_path eval_tasks --output_path "artifacts/eval-${TAG}"

run_stage mmlu "$VENV/lm_eval" run --model "$BACKEND" \
  --model_args "$ARGS" --tasks mmlu --limit 10 \
  --output_path "artifacts/eval-${TAG}"

LANGS="ar bn de en es fr hi id it ja ko pt sw yo zh"
ML=$(for l in $LANGS; do printf "global_mmlu_%s," "$l"; done | sed 's/,$//')
run_stage multilingual "$VENV/lm_eval" run --model "$BACKEND" \
  --model_args "$ARGS" --tasks "$ML" --output_path "artifacts/eval-${TAG}"

run_stage ppl "$VENV/python" -u scripts/eval_perplexity.py \
  --data "${PPL_DATA}" \
  --base-url "${BASE%/completions}" --model "$MODEL" \
  --out "artifacts/ppl-${TAG}-result.json"

# One length per invocation, always: passing several to max_seq_lengths together
# with --limit takes the first N of the concatenated set, so only the shortest
# length would actually be evaluated.
for LEN in ${RULER_LENGTHS:-4096 16384 32768 65536}; do
  run_stage ruler "$VENV/lm_eval" run --model "$BACKEND" \
    --model_args "$RULER_ARGS" \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multiquery,niah_multivalue,ruler_vt,ruler_cwe,ruler_fwe,ruler_qa_squad \
    --limit 10 --metadata "{\"max_seq_lengths\":[${LEN}]}" \
    --output_path "artifacts/ruler-${TAG}"
done

echo "#### suite complete at $(date +%H:%M:%S)"
