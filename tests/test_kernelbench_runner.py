from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import resource
import sys

import pytest
import torch
import yaml


RUNNER = Path(__file__).parents[1] / "scripts/run_kernelbench_mimir.py"
SPEC = spec_from_file_location("run_kernelbench_mimir", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_discover_cases_merges_yaml_and_source_corpus(tmp_path: Path):
    corpus = tmp_path / "third_party/KernelBench/KernelBench/level3"
    corpus.mkdir(parents=True)
    (corpus / "2_B.py").write_text("")
    (corpus / "1_A.py").write_text("")
    examples = tmp_path / "examples/KernelBench"
    examples.mkdir(parents=True)
    (examples / "level3.yaml").write_text(
        yaml.safe_dump([
            {
                "kernel": "level3/1_A.py",
                "input_shapes": ["8x8"],
                "initializations": ["rnd"],
            }
        ])
    )

    cases = runner.discover_cases(tmp_path, ("level3",), tmp_path / "overlays")

    assert [case["kernel"] for case in cases] == [
        "level3/1_A.py",
        "level3/2_B.py",
    ]
    assert [case["fixture"] for case in cases] == ["yaml", "native"]


def test_discover_cases_merges_local_fixture_overlays(tmp_path: Path):
    corpus = tmp_path / "third_party/KernelBench/KernelBench/level2"
    corpus.mkdir(parents=True)
    (corpus / "3_A.py").write_text("")
    examples = tmp_path / "examples/KernelBench"
    examples.mkdir(parents=True)
    (examples / "level2.yaml").write_text(
        yaml.safe_dump([{
            "kernel": "level2/3_A.py",
            "input_shapes": ["32x32"],
            "initializations": ["rnd"],
        }])
    )
    overlays = tmp_path / "overlays"
    overlays.mkdir()
    (overlays / "level2.yaml").write_text(
        yaml.safe_dump([{
            "kernel": "level2/3_A.py",
            "scaled_input_shapes": ["8x4"],
        }])
    )

    cases = runner.discover_cases(tmp_path, ("level2",), overlays)

    assert cases[0]["input_shapes"] == ["32x32"]
    assert cases[0]["scaled_input_shapes"] == ["8x4"]


def test_native_fixture_requires_unscaled_execution(tmp_path: Path):
    case = {"kernel": "level3/1_A.py", "fixture": "native"}
    model = tmp_path / "third_party/KernelBench/KernelBench/level3/1_A.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x): return x\n"
        "def get_init_inputs(): return []\n"
        "def get_inputs(): return [torch.ones(2)]\n"
    )

    try:
        runner.prepare_case(case, tmp_path, size_divisor=16)
    except runner.InvalidCaseError as exc:
        assert "semantics-preserving scaled fixture" in str(exc)
    else:
        raise AssertionError("native fixture unexpectedly accepted scaling")


def test_fixed_fixture_is_not_rescaled(tmp_path: Path):
    case = {
        "kernel": "level3/1_A.py",
        "fixture": "yaml",
        "scalable": False,
        "init_args": [32],
        "input_shapes": ["8x32"],
        "initializations": ["rnd"],
    }
    model = tmp_path / "third_party/KernelBench/KernelBench/level3/1_A.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self, width): super().__init__(); self.width = width\n"
        "    def forward(self, x): return x\n"
    )

    instance, inputs = runner.prepare_case(case, tmp_path, size_divisor=16)

    assert instance.width == 32
    assert inputs[0].shape == (8, 32)


def test_scaled_input_shape_override_preserves_derived_shape(tmp_path: Path):
    case = {
        "kernel": "level2/26_ConvTranspose3d_Add_HardSwish.py",
        "fixture": "yaml",
        "init_args": [],
        "input_shapes": ["128x32x16x16x16", "128x64x32x32x32"],
        "scaled_input_shapes": ["8x8x8x8x8", "8x8x16x16x16"],
        "initializations": ["rnd", "rnd"],
    }
    model = tmp_path / "third_party/KernelBench/KernelBench/level2/26_ConvTranspose3d_Add_HardSwish.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self): super().__init__()\n"
        "    def forward(self, x, residual): return x + residual\n"
    )

    _, inputs = runner.prepare_case(case, tmp_path, size_divisor=16)

    assert [value.shape for value in inputs] == [
        (8, 8, 8, 8, 8),
        (8, 8, 16, 16, 16),
    ]


def test_make_input_supports_integer_zero_fixture():
    value = runner.make_input((8,), "0", "int64")

    assert value.dtype == torch.int64
    assert torch.equal(value, torch.zeros(8, dtype=torch.int64))


def test_prepare_case_applies_per_input_dtypes(tmp_path: Path):
    case = {
        "kernel": "level1/cross_entropy.py",
        "fixture": "yaml",
        "scalable": False,
        "init_args": [],
        "input_shapes": ["4x8", "4"],
        "initializations": ["rnd", "0"],
        "dtypes": ["float32", "int64"],
    }
    model = tmp_path / "third_party/KernelBench/KernelBench/level1/cross_entropy.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        "import torch\n"
        "class Model(torch.nn.Module):\n"
        "    def forward(self, logits, target): return logits\n"
    )

    _, inputs = runner.prepare_case(case, tmp_path, size_divisor=16)

    assert [value.dtype for value in inputs] == [torch.float32, torch.int64]


def test_repository_discovers_all_level1_to_level3_models():
    lighthouse = Path("/workspaces/ml-compiler/lighthouse")
    if not lighthouse.exists():
        return

    cases = runner.discover_cases(
        lighthouse,
        ("level1", "level2", "level3"),
        runner.DEFAULT_FIXTURES,
    )

    assert len(cases) == 250
    assert sum(case["fixture"] == "yaml" for case in cases) >= 205


def test_repository_discovers_level4_models_explicitly():
    lighthouse = Path("/workspaces/ml-compiler/lighthouse")
    if not lighthouse.exists():
        return

    cases = runner.discover_cases(
        lighthouse, ("level4",), runner.DEFAULT_FIXTURES
    )

    assert len(cases) == 20
    assert all(case["fixture"] == "native" for case in cases)


def test_vgg_fixtures_preserve_classifier_input_shape():
    cases = {
        case["kernel"]: case
        for case in yaml.safe_load(
            (runner.DEFAULT_FIXTURES / "level3.yaml").read_text()
        )
    }

    for kernel in ("level3/11_VGG16.py", "level3/12_VGG19.py"):
        case = cases[kernel]
        assert case["scalable"] is False
        assert case["input_shapes"] == ["1x3x224x224"]
        assert case["max_fp_iters"] == 512


def test_vanilla_rnn_fixture_preserves_loop_carried_state():
    cases = {
        case["kernel"]: case
        for case in yaml.safe_load(
            (runner.DEFAULT_FIXTURES / "level3.yaml").read_text()
        )
    }

    case = cases["level3/34_VanillaRNNHidden.py"]
    assert case["input_shapes"] == ["4x2x16", "2x16"]
    assert case["init_args"] == [16, 16, 8]
    assert case["max_fp_iters"] == 512


def test_densenet_component_fixtures_preserve_reduced_shape_contracts():
    cases = {
        case["kernel"]: case
        for case in yaml.safe_load(
            (runner.DEFAULT_FIXTURES / "level3.yaml").read_text()
        )
    }

    assert cases["level3/13_DenseNet121TransitionLayer.py"]["init_args"] == [8, 16]
    assert cases["level3/13_DenseNet121TransitionLayer.py"]["input_shapes"] == [
        "1x8x16x16"
    ]
    assert cases["level3/14_DenseNet121DenseBlock.py"]["init_args"] == [2, 8, 4]
    assert cases["level3/14_DenseNet121DenseBlock.py"]["input_shapes"] == [
        "1x8x16x16"
    ]


def test_suite_coverage_can_exclude_invalid_fixtures():
    results = [
        runner.CaseResult("pass.py", "PASS", "compare", 1.0),
        runner.CaseResult("invalid.py", "INVALID", "fixture", 0.1),
    ]

    summary = runner.evaluate_coverage(results, allow_invalid=True)

    assert summary.passed == 1
    assert summary.eligible == 1
    assert summary.invalid == 1
    assert summary.pass_rate == 1.0
    assert summary.meets(1.0)


def test_suite_coverage_counts_real_failures():
    results = [
        runner.CaseResult("pass.py", "PASS", "compare", 1.0),
        runner.CaseResult("fail.py", "FAIL", "compile_execute", 1.0),
        runner.CaseResult("timeout.py", "TIMEOUT", "compile_execute", 1.0),
    ]

    summary = runner.evaluate_coverage(results, allow_invalid=True)

    assert summary.passed == 1
    assert summary.eligible == 3
    assert summary.pass_rate == pytest.approx(1 / 3)
    assert not summary.meets(0.5)


def test_suite_coverage_rejects_invalid_or_empty_suites_by_default():
    invalid = runner.CaseResult("invalid.py", "INVALID", "fixture", 0.1)

    assert not runner.evaluate_coverage([invalid], allow_invalid=False).meets(0.0)
    assert not runner.evaluate_coverage([invalid], allow_invalid=True).meets(0.0)


def test_memory_limit_clamps_soft_limit_to_finite_hard_limit(monkeypatch):
    finite_hard_limit = 2 * 1024**3
    calls = []

    monkeypatch.setattr(
        runner.resource,
        "getrlimit",
        lambda _: (resource.RLIM_INFINITY, finite_hard_limit),
    )
    monkeypatch.setattr(
        runner.resource,
        "setrlimit",
        lambda resource_type, limits: calls.append((resource_type, limits)),
    )

    runner.apply_memory_limit(4)

    assert calls == [(resource.RLIMIT_AS, (finite_hard_limit, finite_hard_limit))]


def test_memory_limit_ignores_unsupported_macos_rlimit(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    monkeypatch.setattr(
        runner.resource,
        "getrlimit",
        lambda _: (resource.RLIM_INFINITY, resource.RLIM_INFINITY),
    )
    monkeypatch.setattr(
        runner.resource,
        "setrlimit",
        lambda *_: (_ for _ in ()).throw(ValueError("unsupported")),
    )

    runner.apply_memory_limit(4)
