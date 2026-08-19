#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Small, fixed code challenge comparing algorithmic and API-heavy tasks.

The algorithmic half is ten LiveCodeBench v6 tasks (five medium, five hard),
evenly spaced within each difficulty in stable dataset order.  The API half is
ten dependency-available, side-effect-free tasks from BigCodeBench-Hard v0.1.4.

Generation is journaled after every response.  Scoring uses the official local
evaluators from checkouts supplied with --lcb-root and --bcb-root.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pyarrow.parquet as pq


LCB_SYSTEM = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests."
)
BCB_IDS = (
    "BigCodeBench/19",
    "BigCodeBench/177",
    "BigCodeBench/184",
    "BigCodeBench/273",
    "BigCodeBench/509",
    "BigCodeBench/763",
    "BigCodeBench/854",
    "BigCodeBench/879",
    "BigCodeBench/928",
    "BigCodeBench/990",
)


def read_lcb(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    assert len(rows) >= count > 1
    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    assert len(set(indexes)) == count
    return [rows[i] for i in indexes]


def select_lcb(rows: list[dict]) -> list[dict]:
    selected = []
    for difficulty in ("medium", "hard"):
        selected.extend(evenly_spaced(
            [row for row in rows if row["difficulty"] == difficulty], 5))
    return selected


def read_bcb(path: Path) -> list[dict]:
    by_id = {row["task_id"]: row for row in pq.read_table(path).to_pylist()}
    return [by_id[task_id] for task_id in BCB_IDS]


def lcb_messages(row: dict) -> list[dict]:
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += (
            "### Format: You will use the following starter code to write the "
            "solution to the problem and enclose your code within delimiters.\n"
            f"```python\n{row['starter_code']}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin, solve the problem, and "
            "write the answer to stdout (do not directly test on the sample "
            "inputs). Enclose your code within delimiters as follows. Ensure "
            "that the program reads input and writes output when run.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    prompt += "### Answer: (use the provided format with backticks)\n"
    return [
        {"role": "system", "content": LCB_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def bcb_messages(row: dict) -> list[dict]:
    prompt = (
        "Write a correct self-contained Python implementation for the task "
        "below. Follow every stated behavior and exception requirement. "
        "Return the complete code in one markdown code block.\n\n"
        + row["instruct_prompt"]
    )
    return [
        {"role": "system", "content": LCB_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def complete(url: str, model: str, messages: list[dict], timeout: float,
             max_tokens: int) -> tuple[str, str, dict]:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read())
            choice = payload["choices"][0]
            return (choice["message"]["content"],
                    choice.get("finish_reason", ""), payload.get("usage", {}))
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def metric_snapshot(url: str) -> dict[str, float]:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/v1')}/metrics", timeout=10) as r:
            lines = r.read().decode().splitlines()
    except Exception:
        return {}
    values = {}
    for line in lines:
        if line.startswith("#") or "spec_decode" not in line:
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            values[match.group(1)] = values.get(match.group(1), 0.0) + float(match.group(2))
    return values


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["task_id"] for line in path.read_text().splitlines()
            if line.strip()}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def generate_suite(name: str, rows: list[dict], args, out_dir: Path) -> None:
    path = out_dir / f"{name}.jsonl"
    done = completed_ids(path)
    before = metric_snapshot(args.url)
    print(f"[{name}] {len(done)}/{len(rows)} already complete", flush=True)
    for index, row in enumerate(rows, 1):
        task_id = row["question_id"] if name == "lcb" else row["task_id"]
        if task_id in done:
            continue
        messages = lcb_messages(row) if name == "lcb" else bcb_messages(row)
        started = time.perf_counter()
        raw, finish_reason, usage = complete(
            args.url, args.model, messages, args.timeout, args.max_tokens)
        append_jsonl(path, {
            "task_id": task_id,
            "difficulty": row.get("difficulty"),
            "title": row.get("question_title"),
            "libs": row.get("libs"),
            "solution": extract_code(raw),
            "raw": raw,
            "finish_reason": finish_reason,
            "usage": usage,
            "seconds": time.perf_counter() - started,
        })
        print(f"[{name}] {index:02d}/{len(rows)} {task_id} "
              f"{time.perf_counter() - started:.1f}s {finish_reason}", flush=True)
    after = metric_snapshot(args.url)
    (out_dir / f"{name}.metrics.json").write_text(json.dumps({
        "before": before,
        "after": after,
        "delta": {key: after.get(key, 0) - before.get(key, 0)
                  for key in sorted(set(before) | set(after))},
    }, indent=1) + "\n")


def decode_lcb_tests(row: dict) -> dict:
    public = json.loads(row["public_test_cases"])
    try:
        private = json.loads(row["private_test_cases"])
    except (json.JSONDecodeError, TypeError):
        private = json.loads(pickle.loads(zlib.decompress(
            base64.b64decode(row["private_test_cases"].encode()))))
    tests = public + private
    metadata = json.loads(row["metadata"])
    return {
        "input_output": json.dumps({
            "inputs": [case["input"] for case in tests],
            "outputs": [case["output"] for case in tests],
            "fn_name": metadata.get("func_name"),
        })
    }


def score_lcb(rows: list[dict], solutions: dict[str, dict], root: Path) -> list[dict]:
    sys.path.insert(0, str(root))
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    scored = []
    for row in rows:
        task_id = row["question_id"]
        if task_id not in solutions:
            continue
        result, metadata = check_correctness(
            decode_lcb_tests(row), solutions[task_id]["solution"],
            timeout=6, debug=False)
        passed = bool(result) and all(value is True for value in result)
        scored.append({"task_id": task_id, "difficulty": row["difficulty"],
                       "passed": passed, "result": result, "metadata": metadata})
        print(f"[score:lcb] {task_id} {'PASS' if passed else 'FAIL'}", flush=True)
    return scored


def bcb_check(code: str, row: dict, root: Path) -> tuple[str, dict]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from bigcodebench.eval import untrusted_check
    return untrusted_check(
        code, row["test"], row["entry_point"],
        max_as_limit=30 * 1024, max_data_limit=30 * 1024,
        max_stack_limit=10, min_time_limit=1, gt_time_limit=2)


def validate_bcb(rows: list[dict], root: Path) -> list[dict]:
    scored = []
    for row in rows:
        status, details = bcb_check(
            row["complete_prompt"] + "\n" + row["canonical_solution"], row, root)
        scored.append({"task_id": row["task_id"], "status": status,
                       "details": details})
        print(f"[validate:bcb] {row['task_id']} {status}", flush=True)
    return scored


def score_bcb(rows: list[dict], solutions: dict[str, dict], root: Path) -> list[dict]:
    scored = []
    for row in rows:
        task_id = row["task_id"]
        if task_id not in solutions:
            continue
        status, details = bcb_check(solutions[task_id]["solution"], row, root)
        scored.append({"task_id": task_id, "libs": row["libs"],
                       "passed": status == "pass", "status": status,
                       "details": details})
        print(f"[score:bcb] {task_id} {status}", flush=True)
    return scored


def load_solutions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {row["task_id"]: row for row in rows}


def summarize(scored: list[dict]) -> dict:
    passed = sum(bool(row.get("passed", row.get("status") == "pass")) for row in scored)
    return {"passed": passed, "total": len(scored),
            "pass_rate": passed / len(scored) if scored else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("generate", "score", "validate"))
    ap.add_argument("--url", default="http://192.168.100.2:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--suite", choices=("all", "lcb", "bcb"), default="all")
    ap.add_argument("--root", type=Path, default=Path("artifacts/code-challenge"))
    ap.add_argument("--lcb-data", type=Path,
                    default=Path("/tmp/livecodebench-data/test6.jsonl"))
    ap.add_argument("--bcb-data", type=Path, default=Path(
        "/tmp/bigcodebench-hard-data/data/v0.1.4-00000-of-00001.parquet"))
    ap.add_argument("--lcb-root", type=Path, default=Path("/tmp/LiveCodeBench"))
    ap.add_argument("--bcb-root", type=Path, default=Path("/tmp/bigcodebench"))
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600)
    args = ap.parse_args()

    lcb = select_lcb(read_lcb(args.lcb_data))
    bcb = read_bcb(args.bcb_data)
    manifest = {
        "lcb_version": "v6 only", "lcb_method": "5 medium + 5 hard, evenly spaced",
        "lcb": [{key: row[key] for key in
                 ("question_id", "question_title", "difficulty", "contest_date")}
                for row in lcb],
        "bcb_version": "BigCodeBench-Hard v0.1.4", "bcb_ids": list(BCB_IDS),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")

    suites = ("lcb", "bcb") if args.suite == "all" else (args.suite,)
    if args.action == "validate":
        result = validate_bcb(bcb, args.bcb_root)
        (args.root / "bcb-canonical-validation.json").write_text(
            json.dumps(result, indent=1) + "\n")
        return 0 if all(row["status"] == "pass" for row in result) else 1

    if not args.tag:
        ap.error("--tag is required for generate/score")
    out_dir = args.root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        if not args.model:
            ap.error("--model is required for generate")
        for suite in suites:
            generate_suite(suite, lcb if suite == "lcb" else bcb, args, out_dir)
        return 0

    all_results = {}
    if "lcb" in suites:
        all_results["lcb"] = score_lcb(
            lcb, load_solutions(out_dir / "lcb.jsonl"), args.lcb_root)
    if "bcb" in suites:
        all_results["bcb"] = score_bcb(
            bcb, load_solutions(out_dir / "bcb.jsonl"), args.bcb_root)
    payload = {"summary": {name: summarize(rows)
                           for name, rows in all_results.items()},
               "results": all_results}
    (out_dir / "scores.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(payload["summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
