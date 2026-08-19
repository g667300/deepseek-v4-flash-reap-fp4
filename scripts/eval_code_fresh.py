#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 g667300
"""Contamination-resistant mini code benchmark from post-release AtCoder tasks.

The six tasks were published after DeepSeek-V4-Flash-0731.  Test inputs are
generated locally from a fixed seed and are never included in prompts.  Every
generation is journaled immediately; scoring executes each solution against
small oracle-backed cases and one or more complexity cases.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


MOD = 998244353
SEED = 20260817
SYSTEM = (
    "You are an expert competitive programmer. Write the solution immediately: "
    "your response MUST begin with ```python and contain only the complete code "
    "and the closing ```. Do not explain, analyze, or restate the problem. "
    "The program must be efficient, read stdin, and write stdout."
)


@dataclass
class Case:
    input: str
    expected: str | None = None
    kind: str = "exact"


TASKS = {
    "abc469_d": {
        "date": "2026-08-01",
        "points": 400,
        "url": "https://atcoder.jp/contests/abc469/tasks/abc469_d?lang=en",
        "statement": r"""There are N players numbered 1..N. M tournaments were
held, and players A_m and B_m reached the final of tournament m. Count pairs
1 <= x < y <= N such that, for every tournament, at least one of x and y is
one of its two finalists.

Constraints: 2 <= N,M <= 2*10^5; 1 <= A_i < B_i <= N.
Input:
N M
A_1 B_1
...
A_M B_M
Output the count.""",
    },
    "abc469_e": {
        "date": "2026-08-01",
        "points": 475,
        "url": "https://atcoder.jp/contests/abc469/tasks/abc469_e?lang=en",
        "statement": r"""A length-N string S consists of 'o' (win) and 'x'
(loss), and contains at least K wins. Choose 1 <= l <= r <= N such that S[l..r]
contains at least K wins. Maximize the win rate in that substring, i.e. its
number of 'o' characters divided by its length.

Constraints: 1 <= K <= N <= 10^6. S contains at least K occurrences of 'o'.
Input:
N K
S
Output the maximum. Absolute or relative error <= 1e-6 is accepted.""",
    },
    "arc226_a": {
        "date": "2026-08-09",
        "points": 400,
        "url": "https://atcoder.jp/contests/arc226/tasks/arc226_a?lang=en",
        "statement": r"""There are N meetings. Meeting i occupies [S_i,T_i).
Assign exactly one of two people to every meeting. The same person may not be
assigned to two meetings that overlap for a positive length; equivalently they
may share a person only if T_i <= S_j or T_j <= S_i. Count valid assignments
modulo 998244353.

Constraints: 1 <= N <= 3*10^5; 1 <= S_i < T_i <= 2N; all 2N endpoints are
distinct.
Input:
N
S_1 T_1
...
S_N T_N
Output the count modulo 998244353.""",
    },
    "abc471_d": {
        "date": "2026-08-15",
        "points": 400,
        "url": "https://atcoder.jp/contests/abc471/tasks/abc471_d?lang=en",
        "statement": r"""A charger has unlimited slots and batteries have
capacity V. A plugged-in battery gains one unit of charge per unit time, capped
at V. Process Q queries with strictly increasing times. Query `1 t w` inserts
a battery whose charge is w at time t. Query `2 t` removes one battery having
the highest charge at time t and prints that charge, or prints -1 if empty.

Constraints: 1 <= Q <= 3*10^5; 1 <= V,t <= 10^9; 0 <= w <= V.
Input:
Q V
query_1
...
query_Q
Print one line per type-2 query.""",
    },
    "abc471_e": {
        "date": "2026-08-15",
        "points": 450,
        "url": "https://atcoder.jp/contests/abc471/tasks/abc471_e?lang=en",
        "statement": r"""Ball i has integer A_i. For every way to choose K of
the N balls, square the sum of the chosen values. Find the sum of these scores
over all C(N,K) choices modulo 998244353.

Constraints: 1 <= K <= N <= 2*10^5; 1 <= A_i <= 10^9.
Input:
N K
A_1 ... A_N
Output the answer.""",
    },
    "arc227_a": {
        "date": "2026-08-16",
        "points": 400,
        "url": "https://atcoder.jp/contests/arc227/tasks/arc227_a?lang=en",
        "statement": r"""A good string has length 2N and exactly N each of 0
and 1. For good strings S,T, dist(S,T) is the minimum number of adjacent swaps
needed to transform S into T. Given good strings A,B,C, find a good string X
minimizing dist(A,X)+dist(B,X)+dist(C,X), and output the minimum K and one such X.

Constraints: 1 <= N <= 2*10^5.
Input:
N
A
B
C
Output:
K
X
Any optimal X is accepted.""",
    },
}


def exact_pairs(n: int, edges: list[tuple[int, int]]) -> int:
    return sum(all(x in edge or y in edge for edge in edges)
               for x in range(1, n + 1) for y in range(x + 1, n + 1))


def cases_abc469_d(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(18):
        n = rng.randint(2, 10)
        m = rng.randint(1, 18)
        edges = [tuple(sorted(rng.sample(range(1, n + 1), 2))) for _ in range(m)]
        inp = f"{n} {m}\n" + "".join(f"{a} {b}\n" for a, b in edges)
        out.append(Case(inp, str(exact_pairs(n, edges))))
    n = 200_000
    edges = [(1, i) for i in range(2, n + 1)]
    out.append(Case(f"{n} {len(edges)}\n" + "".join(
        f"{a} {b}\n" for a, b in edges), str(n - 1)))
    out.append(Case(f"{n} 200000\n" + "1 2\n" * 200_000, str(2 * n - 3)))
    return out


def best_rate(s: str, k: int) -> float:
    best = 0.0
    for left in range(len(s)):
        wins = 0
        for right in range(left, len(s)):
            wins += s[right] == "o"
            if wins >= k:
                best = max(best, wins / (right - left + 1))
    return best


def cases_abc469_e(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(22):
        n = rng.randint(1, 30)
        s = "".join(rng.choice("ox") for _ in range(n))
        if "o" not in s:
            s = "o" + s[1:]
        k = rng.randint(1, s.count("o"))
        out.append(Case(f"{n} {k}\n{s}\n", repr(best_rate(s, k)), "float"))
    n, k = 1_000_000, 500_000
    out.append(Case(f"{n} {k}\n" + "ox" * k + "\n",
                    repr(k / (2 * k - 1)), "float"))
    out.append(Case(f"{n} {n}\n" + "o" * n + "\n", "1.0", "float"))
    return out


def charger_expected(v: int, queries: list[tuple[int, ...]]) -> str:
    import heapq
    heap: list[int] = []
    ans = []
    for query in queries:
        if query[0] == 1:
            _, t, w = query
            heapq.heappush(heap, -(w - t))
        elif not heap:
            ans.append(-1)
        else:
            _, t = query
            ans.append(min(v, t - heapq.heappop(heap)))
    return "\n".join(map(str, ans))


def cases_abc471_d(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(20):
        q, v, t, active = rng.randint(3, 35), rng.randint(1, 100), 0, 0
        queries = []
        for _ in range(q):
            t += rng.randint(1, 12)
            if active == 0 or rng.random() < 0.62:
                queries.append((1, t, rng.randint(0, v)))
                active += 1
            else:
                queries.append((2, t))
                active -= 1
        inp = f"{q} {v}\n" + "".join(" ".join(map(str, x)) + "\n" for x in queries)
        out.append(Case(inp, charger_expected(v, queries)))
    q, v = 200_000, 10**9
    queries = []
    for i in range(1, 100_001):
        queries.append((1, i, (i * 1_000_003) % (v + 1)))
    for i in range(100_001, q + 1):
        queries.append((2, i))
    inp = f"{q} {v}\n" + "".join(" ".join(map(str, x)) + "\n" for x in queries)
    out.append(Case(inp, charger_expected(v, queries)))
    return out


def comb_mod(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = den = 1
    for i in range(1, k + 1):
        num = num * (n - k + i) % MOD
        den = den * i % MOD
    return num * pow(den, MOD - 2, MOD) % MOD


def square_sum_expected(a: list[int], k: int) -> int:
    squares = sum((x % MOD) ** 2 for x in a) % MOD
    total = sum(a) % MOD
    pair_twice = (total * total - squares) % MOD
    return (comb_mod(len(a) - 1, k - 1) * squares
            + comb_mod(len(a) - 2, k - 2) * pair_twice) % MOD


def cases_abc471_e(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(22):
        n = rng.randint(1, 15)
        k = rng.randint(1, n)
        a = [rng.randint(1, 10**9) for _ in range(n)]
        out.append(Case(f"{n} {k}\n" + " ".join(map(str, a)) + "\n",
                        str(square_sum_expected(a, k))))
    n, k = 200_000, 100_000
    a = [(i * 1_000_000_007) % 10**9 + 1 for i in range(n)]
    out.append(Case(f"{n} {k}\n" + " ".join(map(str, a)) + "\n",
                    str(square_sum_expected(a, k))))
    return out


def meeting_expected(intervals: list[tuple[int, int]]) -> int:
    n = len(intervals)
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i):
            if max(intervals[i][0], intervals[j][0]) < min(intervals[i][1], intervals[j][1]):
                graph[i].append(j)
                graph[j].append(i)
    colors = [-1] * n
    components = 0
    for start in range(n):
        if colors[start] >= 0:
            continue
        components += 1
        colors[start] = 0
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in graph[node]:
                if colors[nxt] < 0:
                    colors[nxt] = colors[node] ^ 1
                    stack.append(nxt)
                elif colors[nxt] == colors[node]:
                    return 0
    return pow(2, components, MOD)


def cases_arc226_a(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(22):
        n = rng.randint(1, 11)
        endpoints = list(range(1, 2 * n + 1))
        rng.shuffle(endpoints)
        intervals = [tuple(sorted(endpoints[2 * i:2 * i + 2])) for i in range(n)]
        inp = f"{n}\n" + "".join(f"{s} {t}\n" for s, t in intervals)
        out.append(Case(inp, str(meeting_expected(intervals))))
    n = 300_000
    disjoint = [(2 * i + 1, 2 * i + 2) for i in range(n)]
    out.append(Case(f"{n}\n" + "".join(f"{s} {t}\n" for s, t in disjoint),
                    str(pow(2, n, MOD))))
    crossing = [(i, i + n) for i in range(1, n + 1)]
    out.append(Case(f"{n}\n" + "".join(f"{s} {t}\n" for s, t in crossing), "0"))
    return out


def random_good(rng: random.Random, n: int) -> str:
    chars = list("0" * n + "1" * n)
    rng.shuffle(chars)
    return "".join(chars)


def dist_good(a: str, b: str) -> int:
    pa = [i for i, char in enumerate(a) if char == "1"]
    pb = [i for i, char in enumerate(b) if char == "1"]
    return sum(abs(x - y) for x, y in zip(pa, pb))


def optimal_good(strings: list[str]) -> tuple[int, str]:
    positions = [[i for i, char in enumerate(s) if char == "1"] for s in strings]
    medians = [sorted(x)[1] for x in zip(*positions)]
    chars = ["0"] * len(strings[0])
    for pos in medians:
        chars[pos] = "1"
    x = "".join(chars)
    return sum(dist_good(s, x) for s in strings), x


def cases_arc227_a(rng: random.Random) -> list[Case]:
    out = []
    for _ in range(22):
        n = rng.randint(1, 35)
        strings = [random_good(rng, n) for _ in range(3)]
        optimum, _ = optimal_good(strings)
        out.append(Case(f"{n}\n" + "\n".join(strings) + "\n",
                        str(optimum), "special_good"))
    n = 200_000
    strings = ["01" * n, "10" * n, "0" * n + "1" * n]
    optimum, _ = optimal_good(strings)
    out.append(Case(f"{n}\n" + "\n".join(strings) + "\n",
                    str(optimum), "special_good"))
    return out


BUILDERS = {
    "abc469_d": cases_abc469_d,
    "abc469_e": cases_abc469_e,
    "arc226_a": cases_arc226_a,
    "abc471_d": cases_abc471_d,
    "abc471_e": cases_abc471_e,
    "arc227_a": cases_arc227_a,
}


def build_cases() -> dict[str, list[Case]]:
    return {task_id: BUILDERS[task_id](random.Random(SEED + i))
            for i, task_id in enumerate(TASKS)}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_manifest(root: Path, cases: dict[str, list[Case]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "frozen_at": "2026-08-17T00:00:00+09:00",
        "release_cutoff": "DeepSeek-V4-Flash-0731",
        "tasks": [{
            "task_id": task_id,
            **{k: TASKS[task_id][k] for k in ("date", "points", "url")},
            "statement_sha256": sha(TASKS[task_id]["statement"]),
            "case_count": len(cases[task_id]),
            "case_sha256": [sha(case.input) for case in cases[task_id]],
        } for task_id in TASKS],
    }
    (root / "manifest.json").write_text(json.dumps(payload, indent=1) + "\n")


def messages(task: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["statement"]}]


def complete(url: str, model: str, task: dict, timeout: float,
             max_tokens: int) -> tuple[str, str, dict]:
    body = json.dumps({"model": model, "messages": messages(task),
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read())
            choice = payload["choices"][0]
            return choice["message"]["content"], choice.get("finish_reason", ""), payload.get("usage", {})
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|Python)?\s*\n(.*?)```", text, re.DOTALL)
    return (blocks[-1] if blocks else text).strip()


def metric_snapshot(url: str) -> dict[str, float]:
    base = url[:-3] if url.endswith("/v1") else url
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/metrics", timeout=10) as response:
            lines = response.read().decode().splitlines()
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


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def generate(args, out_dir: Path) -> None:
    path = out_dir / "solutions.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["task_id"] for line in path.read_text().splitlines() if line}
    before = metric_snapshot(args.url)
    for index, (task_id, task) in enumerate(TASKS.items(), 1):
        if task_id in done:
            continue
        started = time.perf_counter()
        raw, finish, usage = complete(args.url, args.model, task, args.timeout, args.max_tokens)
        append_jsonl(path, {"task_id": task_id, "solution": extract_code(raw),
                            "raw": raw, "finish_reason": finish, "usage": usage,
                            "seconds": time.perf_counter() - started})
        print(f"[generate] {index}/{len(TASKS)} {task_id} {time.perf_counter()-started:.1f}s {finish}", flush=True)
    after = metric_snapshot(args.url)
    (out_dir / "metrics.json").write_text(json.dumps({
        "before": before, "after": after,
        "delta": {k: after.get(k, 0) - before.get(k, 0)
                  for k in sorted(set(before) | set(after))}}, indent=1) + "\n")


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))


def run_code(code: str, inp: str, timeout: float) -> tuple[str, str]:
    with tempfile.TemporaryDirectory(prefix="fresh-code-") as directory:
        path = Path(directory) / "main.py"
        path.write_text(code)
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], input=inp,
                                  text=True, capture_output=True, timeout=timeout,
                                  cwd=directory, preexec_fn=limits,
                                  env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return "", "timeout"
        if proc.returncode:
            return proc.stdout, f"exit {proc.returncode}: {proc.stderr[-500:]}"
        return proc.stdout, ""


def check_output(task_id: str, case: Case, stdout: str) -> tuple[bool, str]:
    if case.kind == "exact":
        got = " ".join(stdout.split())
        expected = " ".join((case.expected or "").split())
        return got == expected, f"expected {expected[:120]!r}, got {got[:120]!r}"
    if case.kind == "float":
        try:
            got = float(stdout.split()[0])
            expected = float(case.expected)
        except (ValueError, IndexError, TypeError):
            return False, "not a float"
        tolerance = 1e-6 * max(1.0, abs(expected))
        return math.isfinite(got) and abs(got - expected) <= tolerance, f"expected {expected}, got {got}"
    tokens = stdout.split()
    if len(tokens) != 2:
        return False, "expected two output tokens: K and X"
    try:
        claimed = int(tokens[0])
    except ValueError:
        return False, "K is not an integer"
    lines = case.input.splitlines()
    n, strings, x = int(lines[0]), lines[1:4], tokens[1]
    if len(x) != 2 * n or x.count("0") != n or x.count("1") != n:
        return False, "X is not a good string"
    actual = sum(dist_good(s, x) for s in strings)
    optimum = int(case.expected)
    ok = claimed == actual == optimum
    return ok, f"optimum={optimum}, claimed={claimed}, actual={actual}"


def score(args, out_dir: Path, cases: dict[str, list[Case]]) -> int:
    rows = [json.loads(line) for line in (out_dir / "solutions.jsonl").read_text().splitlines() if line]
    solutions = {row["task_id"]: row["solution"] for row in rows}
    results = []
    for task_id in TASKS:
        code = solutions.get(task_id)
        if code is None:
            results.append({"task_id": task_id, "passed": False, "error": "missing solution"})
            continue
        failures = []
        started = time.perf_counter()
        for index, case in enumerate(cases[task_id]):
            stdout, error = run_code(code, case.input, args.case_timeout)
            if error:
                failures.append({"case": index, "error": error})
                break
            ok, detail = check_output(task_id, case, stdout)
            if not ok:
                failures.append({"case": index, "error": detail})
                break
        row = {"task_id": task_id, "passed": not failures,
               "cases": len(cases[task_id]), "failures": failures,
               "seconds": time.perf_counter() - started}
        results.append(row)
        print(f"[score] {task_id} {'PASS' if not failures else 'FAIL'} {row['seconds']:.1f}s", flush=True)
    payload = {"summary": {"passed": sum(x["passed"] for x in results),
                            "total": len(results)}, "results": results}
    (out_dir / "scores.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["summary"]), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("manifest", "generate", "score"))
    ap.add_argument("--root", type=Path, default=Path("artifacts/code-fresh"))
    ap.add_argument("--tag")
    ap.add_argument("--url", default="http://192.168.100.2:8000/v1")
    ap.add_argument("--model")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--case-timeout", type=float, default=10)
    args = ap.parse_args()
    cases = build_cases()
    write_manifest(args.root, cases)
    if args.action == "manifest":
        print(f"wrote {args.root / 'manifest.json'}")
        return 0
    if not args.tag:
        ap.error("--tag is required")
    out_dir = args.root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        if not args.model:
            ap.error("--model is required")
        generate(args, out_dir)
        return 0
    return score(args, out_dir, cases)


if __name__ == "__main__":
    raise SystemExit(main())
