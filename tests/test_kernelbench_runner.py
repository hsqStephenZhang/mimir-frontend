from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

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
    assert sum(case["fixture"] == "yaml" for case in cases) >= 203
