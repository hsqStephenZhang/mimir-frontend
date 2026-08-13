#!/usr/bin/env python3
"""Run Lighthouse KernelBench YAML cases through the MimIR JIT backend."""

from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import torch
import yaml

from mimir_frontend.backend import mimir_backend


DEFAULT_LIGHTHOUSE = Path("/workspaces/ml-compiler/lighthouse")


class InvalidCaseError(RuntimeError):
    """The scaled Lighthouse fixture is invalid before MimIR is invoked."""


def load_module(path: Path) -> ModuleType:
    name = f"_mimir_kernelbench_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scaled_extent(value: int, divisor: int) -> int:
    if divisor == 1 or value <= 4:
        return value
    # Eight preserves common channel-group divisibility and leaves enough room
    # for the suite's unscaled kernels, padding, and normalization groups.
    return max(8, (value + divisor - 1) // divisor)


def parse_shape(text: str | int, divisor: int) -> tuple[int, ...]:
    return tuple(
        scaled_extent(int(value), divisor) for value in str(text).split("x")
    )


def make_input(shape: tuple[int, ...], initialization: str) -> torch.Tensor:
    if initialization == "rnd":
        return torch.rand(shape)
    if initialization == "0":
        return torch.zeros(shape)
    if initialization == "id":
        if len(shape) != 2:
            raise ValueError(f"identity initialization requires rank 2, got {shape}")
        return torch.eye(*shape)
    raise ValueError(f"unsupported initialization {initialization!r}")


def scale_init_arg(value, divisor: int):
    if isinstance(value, int) and not isinstance(value, bool):
        return scaled_extent(value, divisor)
    if isinstance(value, list):
        return [scale_init_arg(item, divisor) for item in value]
    if isinstance(value, tuple):
        return tuple(scale_init_arg(item, divisor) for item in value)
    return value


def parse_init_args(value, module: ModuleType, divisor: int):
    if value is None or value == "None":
        value = module.get_init_inputs()
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        value = (value,)
    return [scale_init_arg(item, divisor) for item in value]


def assert_close(actual, expected) -> None:
    actuals = actual if isinstance(actual, (tuple, list)) else (actual,)
    expecteds = expected if isinstance(expected, (tuple, list)) else (expected,)
    if len(actuals) != len(expecteds):
        raise AssertionError(
            f"output count differs: compiled={len(actuals)}, eager={len(expecteds)}"
        )
    for got, want in zip(actuals, expecteds):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def run_case(
    case: dict, lighthouse: Path, size_divisor: int, max_fp_iters: int | None
) -> None:
    model_path = lighthouse / "third_party/KernelBench/KernelBench" / case["kernel"]
    module = load_module(model_path)
    model = module.Model(
        *parse_init_args(case.get("init_args"), module, size_divisor)
    ).eval()
    inputs = [
        make_input(parse_shape(shape, size_divisor), initialization)
        for shape, initialization in zip(
            case["input_shapes"], case["initializations"], strict=True
        )
    ]
    with torch.no_grad():
        try:
            expected = model(*inputs)
        except Exception as exc:
            raise InvalidCaseError(str(exc)) from exc
        options = {"max_fp_iters": max_fp_iters} if max_fp_iters else None
        compiled = torch.compile(
            model, backend=mimir_backend, fullgraph=True, dynamic=False,
            options=options,
        )
        actual = compiled(*inputs)
    assert_close(actual, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lighthouse", type=Path, default=DEFAULT_LIGHTHOUSE)
    parser.add_argument(
        "--suite",
        default="ci",
        choices=("ci", "level1", "level2", "level3", "registered"),
        help="registered combines every model described by the level YAML files",
    )
    parser.add_argument("--kernel", help="only run cases whose path contains this text")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--size-divisor", type=int, default=16,
        help="divide extents larger than four while preserving equalities; use 1 for Lighthouse sizes",
    )
    parser.add_argument(
        "--max-fp-iters", type=int, default=32,
        help="cap MimIR fixed-point phase iterations for suite diagnostics",
    )
    parser.add_argument(
        "--timeout", type=int, default=180,
        help="per-model timeout in seconds; models run in isolated subprocesses",
    )
    args = parser.parse_args()

    suite_names = (
        ("level1", "level2", "level3")
        if args.suite == "registered"
        else (args.suite,)
    )
    cases = []
    for suite_name in suite_names:
        yaml_path = args.lighthouse / "examples/KernelBench" / f"{suite_name}.yaml"
        cases.extend(yaml.safe_load(yaml_path.read_text()))
    if args.kernel:
        cases = [case for case in cases if args.kernel in case["kernel"]]
    if args.case:
        cases = [case for case in cases if args.case == case["kernel"]]
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if not args.case and args.shard_count != 1:
        cases = [
            case for index, case in enumerate(cases)
            if index % args.shard_count == args.shard_index
        ]

    failures = []
    invalid = []
    for index, case in enumerate(cases, start=1):
        name = case["kernel"]
        print(f"[{index}/{len(cases)}] {name}", flush=True)
        torch._dynamo.reset()
        try:
            if args.direct:
                run_case(
                    case, args.lighthouse, args.size_divisor, args.max_fp_iters
                )
            else:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--lighthouse", str(args.lighthouse),
                    "--suite", args.suite,
                    "--case", name,
                    "--size-divisor", str(args.size_divisor),
                    "--max-fp-iters", str(args.max_fp_iters),
                    "--direct",
                ]
                child = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                detail = (child.stdout + child.stderr).strip()
                if child.returncode == 2:
                    raise InvalidCaseError(detail[-4000:])
                if child.returncode:
                    raise RuntimeError(detail[-4000:])
        except InvalidCaseError as exc:
            invalid.append((name, exc))
            print(f"  INVALID: {exc}", flush=True)
        except Exception as exc:
            failures.append((name, exc))
            print(f"  FAIL: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                break
        else:
            print("  PASS", flush=True)

    passed = len(cases) - len(failures) - len(invalid)
    print(f"\n{passed}/{len(cases)} passed, {len(invalid)} invalid")
    for name, exc in failures:
        print(f"FAILED {name}: {type(exc).__name__}: {exc}")
    if failures:
        return 1
    if invalid and args.direct:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
