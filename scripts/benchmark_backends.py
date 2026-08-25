#!/usr/bin/env python3
"""Benchmark eager, inductor, and MimIR in resource-bounded subprocesses."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "py"
BACKENDS = ("eager", "inductor", "mimir")
RESULT_PREFIX = "MIMIR_BENCHMARK_RESULT="


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    backend: str
    status: str
    first_seconds: float | None = None
    mean_seconds: float | None = None
    min_seconds: float | None = None
    max_abs_error: float | None = None
    detail: str = ""


def discover_model_files(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [DEFAULT_MODEL_DIR]
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(
                    item for item in path.glob("*.py") if not item.name.startswith("_")
                )
            )
        else:
            files.append(path)
    return files


def flatten_outputs(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def max_abs_error(actual: Any, expected: Any) -> float:
    actuals = flatten_outputs(actual)
    expecteds = flatten_outputs(expected)
    if len(actuals) != len(expecteds):
        raise AssertionError(
            f"output count differs: actual={len(actuals)}, expected={len(expecteds)}"
        )
    return max(
        (got - want).abs().max().item()
        for got, want in zip(actuals, expecteds, strict=True)
    )


def apply_memory_limit(max_memory_gb: int) -> None:
    if max_memory_gb <= 0 or os.name != "posix":
        return
    import resource

    limit = max_memory_gb * 1024**3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_hard = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
    new_soft = limit if soft == resource.RLIM_INFINITY else min(soft, limit)
    new_soft = min(new_soft, new_hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))


def benchmark_direct(args: argparse.Namespace) -> BenchmarkResult:
    apply_memory_limit(args.max_memory_gb)

    import torch

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass

    import mimir_frontend.backend  # noqa: F401
    from mimir_frontend.model_export import export_spec_from_module, load_python_module

    path = Path(args.model)
    try:
        spec = export_spec_from_module(load_python_module(path))
        spec.model.eval()
        inputs = [torch.randn(*shape) for shape in spec.input_shapes]
        with torch.no_grad():
            expected = spec.model(*inputs)

        if args.backend == "eager":
            function = spec.model
        else:
            torch._dynamo.reset()
            options = (
                {"cache": False} if args.backend == "mimir" and args.no_cache else None
            )
            function = torch.compile(
                spec.model,
                backend=args.backend,
                dynamic=False,
                options=options,
            )

        with torch.no_grad():
            started = time.perf_counter()
            actual = function(*inputs)
            first = time.perf_counter() - started
            timings = []
            for _ in range(args.repeat):
                started = time.perf_counter()
                actual = function(*inputs)
                timings.append(time.perf_counter() - started)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        return BenchmarkResult(
            model=spec.name,
            backend=args.backend,
            status="PASS",
            first_seconds=first,
            mean_seconds=sum(timings) / len(timings),
            min_seconds=min(timings),
            max_abs_error=max_abs_error(actual, expected),
        )
    except Exception as exc:
        message = str(exc).strip()
        return BenchmarkResult(
            model=path.stem,
            backend=args.backend,
            status="FAIL",
            detail=(message or type(exc).__name__)[-4000:],
        )


def run_subprocess(
    path: Path, backend: str, args: argparse.Namespace
) -> BenchmarkResult:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--direct",
        "--model",
        str(path),
        "--backend",
        backend,
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--max-memory-gb",
        str(args.max_memory_gb),
    ]
    if args.no_cache:
        command.append("--no-cache")

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(args.threads)
    environment["MKL_NUM_THREADS"] = str(args.threads)
    started = time.monotonic()
    try:
        child = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            path.stem,
            backend,
            "TIMEOUT",
            detail=f"exceeded {args.timeout}s",
        )

    marker = next(
        (
            line
            for line in reversed(child.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if marker is not None:
        return BenchmarkResult(**json.loads(marker.removeprefix(RESULT_PREFIX)))
    detail = (child.stdout + child.stderr).strip()[-4000:]
    return BenchmarkResult(
        path.stem,
        backend,
        "FAIL",
        detail=detail or f"subprocess exited with status {child.returncode}",
        first_seconds=time.monotonic() - started,
    )


def print_result(result: BenchmarkResult, eager_mean: float | None) -> None:
    if result.status != "PASS":
        summary = result.detail.splitlines()[-1] if result.detail else "no diagnostic"
        print(f"  {result.backend:<10} {result.status:<8} {summary}")
        return
    assert result.first_seconds is not None
    assert result.mean_seconds is not None
    assert result.min_seconds is not None
    speedup = f"{eager_mean / result.mean_seconds:.3g}x" if eager_mean else "-"
    print(
        f"  {result.backend:<10} {result.first_seconds:>10.4g} "
        f"{result.mean_seconds:>10.4g} {result.min_seconds:>10.4g} "
        f"{speedup:>10} {result.max_abs_error:>10.2e}"
    )


def write_results(path: Path, results: list[BenchmarkResult]) -> None:
    payload = json.dumps(
        [asdict(result) for result in results], indent=2, sort_keys=True
    )
    if str(path) == "-":
        print(payload)
    else:
        path.write_text(payload + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-memory-gb", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--results-json", type=Path)
    parser.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    args.backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unknown = [item for item in args.backends if item not in BACKENDS]
    if unknown:
        parser.error(f"unknown backend {unknown[0]!r}")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.max_memory_gb < 0:
        parser.error("--max-memory-gb must be >= 0")
    if args.timeout < 1:
        parser.error("--timeout must be >= 1")
    if args.direct and (args.model is None or args.backend is None):
        parser.error("--direct requires --model and --backend")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.direct:
        result = benchmark_direct(args)
        print(RESULT_PREFIX + json.dumps(asdict(result)), flush=True)
        return 0 if result.status == "PASS" else 1

    files = discover_model_files(args.paths)
    if not files:
        print("no model files found", file=sys.stderr)
        return 1

    results: list[BenchmarkResult] = []
    for path in files:
        print(f"\n{path}")
        print(
            f"  {'backend':<10} {'first[s]':>10} {'mean[s]':>10} "
            f"{'min[s]':>10} {'vs eager':>10} {'max|err|':>10}"
        )
        eager_mean = None
        for backend in sorted(args.backends, key=BACKENDS.index):
            result = run_subprocess(path, backend, args)
            results.append(result)
            if backend == "eager" and result.status == "PASS":
                eager_mean = result.mean_seconds
            print_result(result, eager_mean)
            if args.results_json:
                write_results(args.results_json, results)

    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
