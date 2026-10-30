#!/usr/bin/env python3
"""Join KernelBench runner results with MimIR phase profiles into a per-model compile-time table.

Run the bench with ``MIMIR_PROFILE=summary MIMIR_DEBUG_DIR=<dumps>`` so every case writes a
``<dumps>/<case>/*_profile.txt``; then::

    uv run --no-sync python scripts/summarize_compile_times.py results.json --debug-dir <dumps> [--csv out.csv]

Columns (seconds): dynamo = compile_execute - cached_execute - backend_total (Dynamo tracing and guards),
translate = FX -> MimIR, mimir = MimIR phases up to LLVM IR text (optimize - ll_emit), ll_emit = LLVM IR
emission phase, clang = clang on the emitted IR plus dlopen, backend = whole backend call, pre_llvm =
dynamo + translate + mimir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

PROFILE_LINE = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+[0-9.]+%\s+\d+\s+(\S.*?)\s*$")
COLUMNS = ["dynamo", "translate", "mimir", "ll_emit", "clang", "backend", "pre_llvm"]


def read_profile(case_dir: Path) -> dict[str, float]:
    """Total milliseconds per phase name from the (first) profile summary in ``case_dir``."""
    phases: dict[str, float] = {}
    for path in sorted(case_dir.glob("*_profile.txt")):
        for line in path.read_text().splitlines():
            if match := PROFILE_LINE.match(line):
                total, _self, name = match.groups()
                phases[name] = phases.get(name, 0.0) + float(total)
        break
    return phases


def summarize(results: list[dict], debug_dir: Path | None) -> list[dict]:
    rows = []
    for result in results:
        timings = result.get("timings") or {}
        if "backend_total" not in timings:
            continue
        backend = timings["backend_total"]
        execute = timings.get("cached_execute", 0.0)
        dynamo = max(timings.get("compile_execute", 0.0) - execute - backend, 0.0)
        optimize = timings.get("backend_optimize", 0.0)
        ll_emit = math.nan
        if debug_dir is not None:
            phases = read_profile(debug_dir / Path(result["kernel"]).stem)
            if "runtime.ll_emit" in phases:
                ll_emit = phases["runtime.ll_emit"] / 1000.0
        mimir = optimize - (0.0 if math.isnan(ll_emit) else ll_emit)
        translate = timings.get("backend_translate", 0.0) + timings.get("backend_decompose", 0.0)
        rows.append(
            {
                "kernel": result["kernel"],
                "status": result["status"],
                "dynamo": dynamo,
                "translate": translate,
                "mimir": mimir,
                "ll_emit": ll_emit,
                "clang": timings.get("backend_jit", 0.0),
                "backend": backend,
                "pre_llvm": dynamo + translate + mimir,
            }
        )
    return rows


def geomean(values: list[float]) -> float:
    values = [v for v in values if v > 0 and not math.isnan(v)]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+", help="results JSON file(s) written by run_kernelbench_mimir.py")
    parser.add_argument("--debug-dir", type=Path, help="MIMIR_DEBUG_DIR used for the run (per-case profiles)")
    parser.add_argument("--csv", type=Path, help="also write the table as CSV")
    args = parser.parse_args()

    results = [r for path in args.results for r in json.loads(path.read_text())]
    rows = summarize(results, args.debug_dir)
    if not rows:
        print("no results with backend timings (is the runner up to date?)", file=sys.stderr)
        return 1

    width = max(len(row["kernel"]) for row in rows)
    print(f"{'kernel':<{width}}  status  " + "  ".join(f"{c:>9}" for c in COLUMNS))
    for row in sorted(rows, key=lambda r: r["kernel"]):
        cells = "  ".join(f"{row[c]:9.3f}" for c in COLUMNS)
        print(f"{row['kernel']:<{width}}  {row['status']:<6}  {cells}")
    passed = [row for row in rows if row["status"] == "PASS"]
    print(f"{'geomean (' + str(len(passed)) + ' PASS)':<{width}}  {'':6}  " + "  ".join(f"{geomean([r[c] for r in passed]):9.3f}" for c in COLUMNS))
    if passed:
        share = sum(r["pre_llvm"] for r in passed) / sum(r["backend"] + r["dynamo"] for r in passed)
        print(f"pre-LLVM share of total compile time (sum over PASS): {share:.1%}")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["kernel", "status", *COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
