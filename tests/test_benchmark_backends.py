from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts/benchmark_backends.py"
SPEC = spec_from_file_location("benchmark_backends", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_discover_model_files_is_sorted_and_ignores_private_files(tmp_path: Path):
    (tmp_path / "b.py").write_text("")
    (tmp_path / "a.py").write_text("")
    (tmp_path / "_helper.py").write_text("")

    assert [path.name for path in benchmark.discover_model_files([tmp_path])] == [
        "a.py",
        "b.py",
    ]


def test_max_abs_error_supports_multiple_outputs():
    expected = (torch.tensor([1.0, 2.0]), torch.tensor([4.0]))
    actual = (torch.tensor([1.0, 2.25]), torch.tensor([3.5]))

    assert benchmark.max_abs_error(actual, expected) == 0.5


def test_max_abs_error_rejects_output_count_mismatch():
    with pytest.raises(AssertionError, match="output count differs"):
        benchmark.max_abs_error([torch.ones(1)], [torch.ones(1), torch.ones(1)])


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--repeat", "0"], "repeat"),
        (["--threads", "0"], "threads"),
        (["--max-memory-gb", "-1"], "max-memory"),
        (["--timeout", "0"], "timeout"),
        (["--backends", "missing"], "unknown backend"),
    ],
)
def test_parse_args_rejects_invalid_limits(arguments: list[str], message: str, capsys):
    with pytest.raises(SystemExit):
        benchmark.parse_args(arguments)
    assert message in capsys.readouterr().err


def test_direct_mode_requires_model_and_backend(capsys):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--direct"])
    assert "requires --model and --backend" in capsys.readouterr().err
