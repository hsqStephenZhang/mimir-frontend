"""The opt-in `nvptx` target (`options={"target": "nvptx"}` or MIMIR_TARGET).

The option checks run anywhere; the execution test needs ptxas/fatbinary (CUDA
toolkit) and a CUDA device, and is skipped otherwise.
"""

import os
import shutil
from pathlib import Path

import pytest
import torch

import mimir_frontend.backend  # noqa: F401  (registers the "mimir" backend)
from mimir_frontend.utils import TARGETS

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")

CUDA_BIN = Path(os.environ.get("CUDA_HOME") or "/opt/cuda") / "bin"


def _cuda_tools_on_path() -> bool:
    return all(shutil.which(tool) or (CUDA_BIN / tool).exists() for tool in ("ptxas", "fatbinary"))


def _compile(model, x, tmp_path, **options):
    torch._dynamo.reset()
    return torch.compile(model, backend="mimir", options={"cache_dir": str(tmp_path), **options})(x)


def mlp():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).eval()


def test_targets_are_cpu_and_nvptx():
    assert TARGETS == ("cpu", "nvptx")


def test_unknown_target_is_rejected(tmp_path):
    with pytest.raises(Exception, match="target must be one of"):
        _compile(mlp(), torch.randn(2, 16), tmp_path, target="tpu")


def test_emit_only_is_cpu_only(tmp_path):
    with pytest.raises(Exception, match="emit_only applies to the cpu target only"):
        _compile(mlp(), torch.randn(2, 16), tmp_path, target="nvptx", emit_only="mlir", debug_dir=str(tmp_path))


def test_default_target_is_cpu(tmp_path, monkeypatch):
    monkeypatch.delenv("MIMIR_TARGET", raising=False)
    model, x = mlp(), torch.randn(2, 16)
    torch.testing.assert_close(_compile(model, x, tmp_path), model(x), rtol=1e-4, atol=1e-5)
    assert not list(tmp_path.rglob("*_dev.ll")), "the default target must not emit device code"


@pytest.mark.skipif(not _cuda_tools_on_path(), reason="ptxas/fatbinary not found (CUDA toolkit)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_nvptx_target_matches_eager(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{CUDA_BIN}{os.pathsep}{os.environ['PATH']}")
    sm = "".join(map(str, torch.cuda.get_device_capability(0)))
    torch.manual_seed(0)
    model, x = mlp(), torch.randn(8, 16)
    got = _compile(model, x, tmp_path, target="nvptx", gpu_sm=sm, debug_dir=str(tmp_path / "dbg"))
    torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-5)
    host = [p for p in (tmp_path / "dbg").rglob("mimir_graph_*.ll") if not p.name.endswith("_dev.ll")]
    assert len(host) == 1 and "@.fatbin = private constant" in host[0].read_text(), "fatbin not embedded"


class CatNet(torch.nn.Module):
    """Two producers, a `cat`, and a consumer: the result of every step can stay on the device."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 4, 3)
        self.b = torch.nn.Conv2d(3, 6, 3)
        self.c = torch.nn.Conv2d(10, 5, 1)

    def forward(self, x):
        return self.c(torch.cat([self.a(x), torch.relu(self.b(x))], dim=1))


def _host_ll(debug_dir: Path) -> str:
    host = [p for p in debug_dir.rglob("mimir_graph_*.ll") if not p.name.endswith("_dev.ll")]
    assert len(host) == 1, host
    return host[0].read_text()


@pytest.mark.skipif(not _cuda_tools_on_path(), reason="ptxas/fatbinary not found (CUDA toolkit)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("dim", [1, -1], ids=["channels", "last"])
def test_cat_stays_on_the_device(dim, tmp_path, monkeypatch):
    """`torch.cat` between GPU kernels is a kernel itself: only the model's output is copied back."""
    monkeypatch.setenv("PATH", f"{CUDA_BIN}{os.pathsep}{os.environ['PATH']}")
    sm = "".join(map(str, torch.cuda.get_device_capability(0)))
    torch.manual_seed(0)
    if dim == 1:
        model, x = CatNet().eval(), torch.randn(2, 3, 9, 9)
    else:
        lin = torch.nn.Linear(6, 5)
        model = type("M", (torch.nn.Module,), {"forward": lambda self, x: lin(torch.cat([x * 2.0, torch.relu(x)], dim=-1))})()
        x = torch.randn(4, 3)
    got = _compile(model, x, tmp_path, target="nvptx", gpu_sm=sm, debug_dir=str(tmp_path / "dbg"))
    torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-5)
    text = _host_ll(tmp_path / "dbg")
    assert text.count("call i32 @cuMemcpyDtoH") == 1, "an intermediate result was copied back to the host"


PAD_MODELS = {
    "padded-convs": (
        lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 5, 3, padding=1)
        ),
        (2, 3, 8, 8),
    ),
    "pooling": (
        lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.MaxPool2d(3, stride=2, padding=1), torch.nn.Conv2d(4, 5, 1)
        ),
        (1, 3, 9, 9),
    ),
}


@pytest.mark.skipif(not _cuda_tools_on_path(), reason="ptxas/fatbinary not found (CUDA toolkit)")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("name", PAD_MODELS)
def test_padding_stays_on_the_device(name, tmp_path, monkeypatch):
    """Padding between GPU kernels is a kernel itself: only the model's output is copied back."""
    monkeypatch.setenv("PATH", f"{CUDA_BIN}{os.pathsep}{os.environ['PATH']}")
    sm = "".join(map(str, torch.cuda.get_device_capability(0)))
    torch.manual_seed(0)
    factory, shape = PAD_MODELS[name]
    model, x = factory().eval(), torch.randn(shape)
    got = _compile(model, x, tmp_path, target="nvptx", gpu_sm=sm, debug_dir=str(tmp_path / "dbg"))
    torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-5)
    text = _host_ll(tmp_path / "dbg")
    assert text.count("call i32 @cuMemcpyDtoH") == 1, "an intermediate result was copied back to the host"
