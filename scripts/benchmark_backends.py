"""Benchmark models across the mimir, eager, and inductor torch.compile backends.

Model files follow the same conventions as `scripts/export_models_to_mimir.py`:
either `export_to_mim = export(...)` or `Model` + `get_inputs()` + `get_init_inputs()`
(the KernelBench layout). Every backend runs the same random inputs; outputs are
compared against eager and reported as a max-abs-error column.

Examples::

    # all backends, single-threaded PyTorch for a fair comparison with mimir
    uv run python scripts/benchmark_backends.py models/py/mlp.py --threads 1

    # a whole directory, mimir vs eager only, fresh mimir compile
    uv run python scripts/benchmark_backends.py models/py --backends mimir,eager --no-cache

`--threads N` pins PyTorch's intra/inter-op thread pools (OMP/MKL env vars are
set before torch is imported, so it applies to eager and inductor kernels). The
mimir runtime is single-threaded, so `--threads 1` levels the field.

The first call per backend is reported separately (`first[s]`): for the
compiled backends it includes compilation (for mimir a cached .so may make it
cheap; pass `--no-cache` to force a fresh compile).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "py"

BACKENDS = ("eager", "inductor", "mimir")


def discover_model_files(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [DEFAULT_MODEL_DIR]

    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.py") if not p.name.startswith("_")))
        else:
            files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model files across torch.compile backends (eager/inductor/mimir).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Model .py files or directories. Defaults to models/py.",
    )
    parser.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help=f"Comma-separated subset of {'/'.join(BACKENDS)} (default: all).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Pin PyTorch to N intra/inter-op threads (use 1 to disable parallelism).",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Timed runs per backend after the first call.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random inputs.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the mimir .so cache so its first call is a full compile.",
    )
    args = parser.parse_args()

    args.backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend in args.backends:
        if backend not in BACKENDS:
            parser.error(f"unknown backend {backend!r}, expected one of {', '.join(BACKENDS)}")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    return args


def flatten_outputs(out) -> list:
    return list(out) if isinstance(out, (tuple, list)) else [out]


def max_abs_err(got, want) -> float:
    return max((g - w).abs().max().item() for g, w in zip(flatten_outputs(got), flatten_outputs(want)))


def fmt(value: float | None, unit: float = 1.0) -> str:
    return f"{value / unit:#.4g}" if value is not None else "-"


def bench_model(path: Path, args: argparse.Namespace) -> bool:
    import torch

    import mimir_frontend.backend  # noqa: F401 - registers the "mimir" backend
    from mimir_frontend.model_export import export_spec_from_module, load_python_module

    spec = export_spec_from_module(load_python_module(path))
    torch.manual_seed(args.seed)
    inputs = [torch.randn(*shape) for shape in spec.input_shapes]
    spec.model.eval()

    with torch.no_grad():
        want = spec.model(*inputs)

    shapes = ", ".join(str(tuple(shape)) for shape in spec.input_shapes)
    print(f"\n{spec.name} ({path}) inputs [{shapes}] threads={torch.get_num_threads()} repeat={args.repeat}")
    header = f"  {'backend':<10} {'first[s]':>10} {'mean[s]':>10} {'min[s]':>10} {'vs eager':>10} {'max|err|':>10}"
    print(header)

    ok = True
    eager_mean = None
    # Canonical order: eager first, so the "vs eager" baseline exists for the others.
    for backend in sorted(args.backends, key=BACKENDS.index):
        try:
            if backend == "eager":
                fn = spec.model
            else:
                torch._dynamo.reset()
                options = {"cache": False} if args.no_cache and backend == "mimir" else None
                kwargs = {"backend": backend, "options": options} if backend == "mimir" else {"backend": backend}
                fn = torch.compile(spec.model, dynamic=False, **kwargs)

            with torch.no_grad():
                t0 = time.perf_counter()
                got = fn(*inputs)
                first = time.perf_counter() - t0
                times = []
                for _ in range(args.repeat):
                    t0 = time.perf_counter()
                    got = fn(*inputs)
                    times.append(time.perf_counter() - t0)
        except Exception as exc:
            ok = False
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            print(f"  {backend:<10} failed: {reason}")
            continue

        mean = sum(times) / len(times)
        if backend == "eager":
            eager_mean = mean
        speedup = f"{eager_mean / mean:#.3g}x" if eager_mean else "-"
        print(
            f"  {backend:<10} {fmt(first):>10} {fmt(mean):>10} {fmt(min(times)):>10}"
            f" {speedup:>10} {max_abs_err(got, want):>10.2e}"
        )
    return ok


def main() -> int:
    # Parse first: thread caps must land in the environment before torch loads
    # so OpenMP/MKL pools (used by eager and inductor kernels) respect them.
    args = parse_args()
    if args.threads is not None:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    import torch

    if args.threads is not None:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass  # inter-op pool already started; intra-op cap still applies

    files = discover_model_files(args.paths)
    if not files:
        print("no model files found", file=sys.stderr)
        return 1

    failures = 0
    for path in files:
        try:
            if not bench_model(path, args):
                failures += 1
        except Exception as exc:
            failures += 1
            print(f"\n{path}: failed to load/benchmark: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
