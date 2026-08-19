#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
# Serve the pruned checkpoint on a DGX Spark *with* DSpark speculative decoding.
#
#   serve_patched.sh [user@]host:/remote/ckpt/dir
#
# e.g. serve_patched.sh user@spark:/srv/models/dsv4-reap128
#
# It rsyncs the checkpoint to the host, mounts the overlays over the image, and
# starts the server; docs/SERVING.md has the same launch as a plain docker run.
# Non-speculative serving needs none of this. This script adds the three things
# speculation needs on GB10, none of which is optional:
#
# * **the patch.** Stock vLLM 0.25.1 cannot speculate on sm120/sm121 at all.
#   vLLM gives the SWA segment a 256-token page (V4's indexer needs it) while
#   FlashInfer instantiates the DSv4 decode kernel only for 64-token pages, so
#   the call falls through to the prefill orchestrator and hits
#   `Check failed: num_tokens > 64` -- which speculative verification, submitting
#   1-6 tokens, can never satisfy. artifacts/patches/flashinfer_sparse.py re-views
#   the page at 64 tokens (zero copy) and is mounted over the image's copy, so the
#   image stays untouched and removing the container undoes everything.
# * **--kv-cache-dtype fp8_ds_mla.** Plain fp8 also holds far less: measured
#   45,169 vs 7,084 KV tokens under otherwise identical settings.
# * **num_speculative_tokens 5.** The checkpoint's config says
#   `dspark_block_size: 5`; the upstream recipe reports lower values *produce
#   incorrect output*, so this is correctness, not tuning.
#
# Measured with all three (2026-08-15, one Spark, 4 prompts x 64 tokens, greedy):
# 26.6 tok/s against 16.5 without speculation, 48.5% acceptance, 2.4 tokens
# accepted per step.
set -euo pipefail

DST="${1:?usage: $0 [user@]host:/remote/ckpt/dir}"
[[ "$DST" == *:* ]] || { echo "destination must be [user@]host:/path" >&2; exit 2; }
host="${DST%%:*}"
remote_dir="${DST#*:}"
remote_dir="${remote_dir%/}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PATCH_SRC="${PATCH_SRC:-$HERE/../artifacts/patches/flashinfer_sparse.py}"
DSPARK_PATCH_SRC="${DSPARK_PATCH_SRC:-$HERE/../artifacts/patches/dspark.py}"
ROUTER_PATCH_SRC="${ROUTER_PATCH_SRC:-$HERE/../artifacts/patches/fused_topk_bias_router.py}"
NAME="${NAME:-dsv4-patched}"
IMAGE="${IMAGE:-vllm-dsv4:0.25.1-fi0614}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-65536}"   # measured: fits at GPUUTIL 0.75 with the draft head
MAXSEQS="${MAXSEQS:-6}"
GPUUTIL="${GPUUTIL:-0.75}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

[[ -f "$PATCH_SRC" ]] || { echo "no patched file at $PATCH_SRC" >&2; exit 2; }
[[ -f "$DSPARK_PATCH_SRC" ]] || { echo "no patched file at $DSPARK_PATCH_SRC" >&2; exit 2; }
[[ -f "$ROUTER_PATCH_SRC" ]] || { echo "no patched file at $ROUTER_PATCH_SRC" >&2; exit 2; }

remote_patch=/tmp/flashinfer_sparse_patched.py
remote_dspark_patch=/tmp/dspark_patched.py
remote_router_patch=/tmp/fused_topk_bias_router_patched.py
echo "== copying the patched kernel, DSpark, and router wrappers to ${host}"
scp -q "$PATCH_SRC" "${host}:${remote_patch}"
scp -q "$DSPARK_PATCH_SRC" "${host}:${remote_dspark_patch}"
scp -q "$ROUTER_PATCH_SRC" "${host}:${remote_router_patch}"

echo "== (re)starting ${NAME} on ${host}"
ssh "$host" "docker rm -f '${NAME}' >/dev/null 2>&1 || true"
ssh "$host" "docker run -d --name '${NAME}' \
  --gpus all --ipc=host --memory=115g --memory-swap=115g \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -p ${PORT}:8000 \
  -v '${remote_dir}':/model:ro \
  -v ${remote_patch}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:ro \
  -v ${remote_dspark_patch}:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py:ro \
  -v ${remote_router_patch}:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:ro \
  --entrypoint vllm '${IMAGE}' serve /model \
  --served-model-name '${NAME}' --gpu-memory-utilization ${GPUUTIL} \
  --max-model-len ${MAXLEN} --max-num-seqs ${MAXSEQS} \
  --kv-cache-dtype fp8_ds_mla \
  --speculative-config '{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS}}'" >/dev/null

url="http://${host#*@}:${PORT}"
echo "== waiting for ${url}/health (up to ${READY_TIMEOUT}s; the load alone is ~9 min)"
start=$SECONDS
while ! curl -sf "${url}/health" >/dev/null 2>&1; do
  if ! ssh "$host" "docker ps -q -f name='^${NAME}\$'" | grep -q .; then
    echo "container exited during load; last 40 lines:" >&2
    ssh "$host" "docker logs --tail 40 '${NAME}'" >&2 || true
    exit 1
  fi
  if (( SECONDS - start >= READY_TIMEOUT )); then
    echo "not ready after ${READY_TIMEOUT}s; last 40 lines:" >&2
    ssh "$host" "docker logs --tail 40 '${NAME}'" >&2 || true
    exit 1
  fi
  sleep 15
done

echo "== ready after $((SECONDS - start))s: ${url}, served as '${NAME}'"
echo "== check it, and read the acceptance rate, with:"
echo "   scripts/check_serving.py --url ${url} --model ${NAME}"
