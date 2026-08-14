#!/usr/bin/env python3
"""Run KernelBench models through the MimIR JIT backend."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import torch
import yaml

from mimir_frontend.backend import mimir_backend


DEFAULT_LIGHTHOUSE = Path("/workspaces/ml-compiler/lighthouse")
RESULT_PREFIX = "MIMIR_KERNELBENCH_RESULT="


class InvalidCaseError(RuntimeError):
    """The selected fixture is invalid before MimIR is invoked."""


class PhaseError(RuntimeError):
    """A KernelBench case failed in a named execution phase."""

    def __init__(self, phase: str, cause: BaseException):
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


@dataclass(frozen=True)
class CaseResult:
    kernel: str
    status: str
    phase: str
    elapsed_seconds: float
    detail: str = ""


def load_module(path: Path) -> ModuleType:
    name = f"_mimir_kernelbench_{path.parent.name}_{path.stem}"
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


def scale_init_arg(value: Any, divisor: int) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return scaled_extent(value, divisor)
    if isinstance(value, list):
        return [scale_init_arg(item, divisor) for item in value]
    if isinstance(value, tuple):
        return tuple(scale_init_arg(item, divisor) for item in value)
    return value


def parse_init_args(value: Any, module: ModuleType, divisor: int) -> list[Any]:
    if value is None or value == "None":
        value = module.get_init_inputs()
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        value = (value,)
    return [scale_init_arg(item, divisor) for item in value]


def numeric_kernel_key(path: Path) -> tuple[int, str]:
    prefix = path.stem.split("_", 1)[0]
    return (int(prefix) if prefix.isdigit() else sys.maxsize, path.name)


def discover_cases(lighthouse: Path, levels: tuple[str, ...]) -> list[dict[str, Any]]:
    """Merge maintained YAML fixtures with the authoritative source corpus."""
    yaml_cases: dict[str, dict[str, Any]] = {}
    examples = lighthouse / "examples/KernelBench"
    for level in levels:
        yaml_path = examples / f"{level}.yaml"
        if yaml_path.exists():
            for case in yaml.safe_load(yaml_path.read_text()) or []:
                yaml_cases[case["kernel"]] = {**case, "fixture": "yaml"}

    corpus = lighthouse / "third_party/KernelBench/KernelBench"
    cases: list[dict[str, Any]] = []
    for level in levels:
        level_dir = corpus / level
        for path in sorted(level_dir.glob("*.py"), key=numeric_kernel_key):
            kernel = f"{level}/{path.name}"
            cases.append(yaml_cases.get(kernel, {"kernel": kernel, "fixture": "native"}))
    return cases


def assert_close(actual: Any, expected: Any) -> None:
    actuals = actual if isinstance(actual, (tuple, list)) else (actual,)
    expecteds = expected if isinstance(expected, (tuple, list)) else (expected,)
    if len(actuals) != len(expecteds):
        raise AssertionError(
            f"output count differs: compiled={len(actuals)}, eager={len(expecteds)}"
        )
    for got, want in zip(actuals, expecteds, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def prepare_case(
    case: dict[str, Any], lighthouse: Path, size_divisor: int
) -> tuple[torch.nn.Module, list[Any]]:
    model_path = lighthouse / "third_party/KernelBench/KernelBench" / case["kernel"]
    try:
        module = load_module(model_path)
    except Exception as exc:
        raise PhaseError("import", exc) from exc

    try:
        if case["fixture"] == "native":
            if size_divisor != 1:
                raise InvalidCaseError(
                    "no semantics-preserving scaled fixture; rerun with --size-divisor 1"
                )
            model = module.Model(*module.get_init_inputs()).eval()
            inputs = list(module.get_inputs())
        else:
            model = module.Model(
                *parse_init_args(case.get("init_args"), module, size_divisor)
            ).eval()
            inputs = [
                make_input(parse_shape(shape, size_divisor), initialization)
                for shape, initialization in zip(
                    case["input_shapes"], case["initializations"], strict=True
                )
            ]
    except InvalidCaseError:
        raise
    except Exception as exc:
        raise PhaseError("fixture", exc) from exc
    return model, inputs


def run_case(
    case: dict[str, Any], lighthouse: Path, size_divisor: int, max_fp_iters: int | None
) -> None:
    model, inputs = prepare_case(case, lighthouse, size_divisor)
    with torch.no_grad():
        try:
            expected = model(*inputs)
        except Exception as exc:
            raise InvalidCaseError(f"eager fixture failed: {exc}") from exc
        try:
            options = {"max_fp_iters": max_fp_iters} if max_fp_iters else None
            compiled = torch.compile(
                model,
                backend=mimir_backend,
                fullgraph=True,
                dynamic=False,
                options=options,
            )
            actual = compiled(*inputs)
        except Exception as exc:
            raise PhaseError("compile_execute", exc) from exc
        try:
            assert_close(actual, expected)
        except Exception as exc:
            raise PhaseError("compare", exc) from exc


def execute_direct(
    case: dict[str, Any], lighthouse: Path, size_divisor: int, max_fp_iters: int | None
) -> CaseResult:
    started = time.monotonic()
    try:
        run_case(case, lighthouse, size_divisor, max_fp_iters)
    except InvalidCaseError as exc:
        return CaseResult(
            case["kernel"],
            "INVALID",
            "fixture",
            time.monotonic() - started,
            str(exc),
        )
    except PhaseError as exc:
        return CaseResult(
            case["kernel"],
            "FAIL",
            exc.phase,
            time.monotonic() - started,
            str(exc),
        )
    except Exception as exc:
        return CaseResult(
            case["kernel"],
            "FAIL",
            "unknown",
            time.monotonic() - started,
            str(exc),
        )
    return CaseResult(case["kernel"], "PASS", "compare", time.monotonic() - started)


def write_results(path: Path, results: list[CaseResult]) -> None:
    payload = json.dumps([asdict(result) for result in results], indent=2, sort_keys=True)
    if str(path) == "-":
        print(payload)
    else:
        path.write_text(payload + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lighthouse", type=Path, default=DEFAULT_LIGHTHOUSE)
    parser.add_argument(
        "--suite",
        default="ci",
        choices=("ci", "level1", "level2", "level3", "registered", "full"),
        help="full discovers all Level 1-3 sources; registered is kept as an alias",
    )
    parser.add_argument("--kernel", help="only run cases whose path contains this text")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--direct", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--list", action="store_true", help="list selected kernels without running"
    )
    parser.add_argument(
        "--results-json", type=Path, help="write structured results, or '-' for stdout"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip PASS entries already in results-json",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--size-divisor", type=int, default=16)
    parser.add_argument("--max-fp-iters", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.suite == "ci":
        yaml_path = args.lighthouse / "examples/KernelBench/ci.yaml"
        cases = [{**case, "fixture": "yaml"} for case in yaml.safe_load(yaml_path.read_text())]
    elif args.suite == "registered":
        cases = []
        for level in ("level1", "level2", "level3"):
            yaml_path = args.lighthouse / "examples/KernelBench" / f"{level}.yaml"
            cases.extend(
                {**case, "fixture": "yaml"}
                for case in yaml.safe_load(yaml_path.read_text()) or []
            )
    else:
        levels = (
            ("level1", "level2", "level3")
            if args.suite == "full"
            else (args.suite,)
        )
        cases = discover_cases(args.lighthouse, levels)
    if args.kernel:
        cases = [case for case in cases if args.kernel in case["kernel"]]
    if args.case:
        cases = [case for case in cases if args.case == case["kernel"]]
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if not args.case and args.shard_count != 1:
        cases = [case for index, case in enumerate(cases) if index % args.shard_count == args.shard_index]

    previous: list[CaseResult] = []
    if args.resume:
        if args.results_json is None or str(args.results_json) == "-":
            parser.error("--resume requires a results-json file")
        if args.results_json.exists():
            previous = [
                CaseResult(**item)
                for item in json.loads(args.results_json.read_text())
            ]
            passed = {result.kernel for result in previous if result.status == "PASS"}
            cases = [case for case in cases if case["kernel"] not in passed]

    if args.list:
        for case in cases:
            print(f"{case['kernel']}\t{case['fixture']}")
        return 0

    results = list(previous)
    for index, case in enumerate(cases, start=1):
        name = case["kernel"]
        print(f"[{index}/{len(cases)}] {name}", flush=True)
        torch._dynamo.reset()
        if args.direct:
            result = execute_direct(case, args.lighthouse, args.size_divisor, args.max_fp_iters)
            print(RESULT_PREFIX + json.dumps(asdict(result)), flush=True)
        else:
            started = time.monotonic()
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--lighthouse", str(args.lighthouse), "--suite", args.suite,
                "--case", name, "--size-divisor", str(args.size_divisor),
                "--max-fp-iters", str(args.max_fp_iters), "--direct",
            ]
            try:
                child = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                marker = next(
                    (
                        line
                        for line in reversed(child.stdout.splitlines())
                        if line.startswith(RESULT_PREFIX)
                    ),
                    None,
                )
                if marker is None:
                    detail = (child.stdout + child.stderr).strip()[-4000:]
                    result = CaseResult(
                        name,
                        "FAIL",
                        "subprocess",
                        time.monotonic() - started,
                        detail,
                    )
                else:
                    result = CaseResult(**json.loads(marker.removeprefix(RESULT_PREFIX)))
            except subprocess.TimeoutExpired:
                result = CaseResult(
                    name,
                    "TIMEOUT",
                    "compile_execute",
                    time.monotonic() - started,
                    f"exceeded {args.timeout}s",
                )
        results = [old for old in results if old.kernel != name] + [result]
        print(f"  {result.status} [{result.phase}]: {result.detail}", flush=True)
        if args.results_json:
            write_results(args.results_json, results)
        if args.fail_fast and result.status != "PASS":
            break

    selected = {case["kernel"] for case in cases}
    selected_results = [result for result in results if result.kernel in selected]
    passed_count = sum(result.status == "PASS" for result in selected_results)
    print(f"\n{passed_count}/{len(selected_results)} passed")
    return 0 if all(result.status == "PASS" for result in selected_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
