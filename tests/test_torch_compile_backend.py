"""End-to-end tests for the "mimir" torch.compile backend.

Each test lowers a Dynamo graph through MimIR to a shared library (requires
clang on PATH) and compares the JIT-compiled result against eager PyTorch.
"""

import shutil

import pytest
import torch

from mimir_frontend.backend import mimir_backend

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")


class LinearMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.fc2 = torch.nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SmallConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3)
        self.fc = torch.nn.Linear(4 * 6 * 6, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


@pytest.fixture(autouse=True)
def _reset_dynamo(monkeypatch, tmp_path_factory):
    # Hermetic per-test-run JIT cache: don't read from or pollute the user cache.
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "mimir-jit-cache"))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def check_against_eager(model, *inputs):
    with torch.no_grad():
        want = model(*inputs)
        compiled = torch.compile(model, backend=mimir_backend)
        got = compiled(*inputs)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_linear_mlp_matches_eager():
    check_against_eager(LinearMLP(), torch.randn(4, 16))


def test_conv_matches_eager():
    check_against_eager(SmallConv(), torch.randn(2, 1, 8, 8))


def test_registered_by_name():
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, backend="mimir")(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


class TwoOutputs(torch.nn.Module):
    """Two outputs of different shapes sharing an intermediate."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32)
        self.head_a = torch.nn.Linear(32, 8)
        self.head_b = torch.nn.Linear(32, 3)

    def forward(self, x):
        hidden = torch.relu(self.fc(x))
        return self.head_a(hidden), self.head_b(hidden)


class ThreeOutputs(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.relu(x - y), (x * y).sum(dim=1)


def test_two_outputs_match_eager():
    model = TwoOutputs()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, backend="mimir")(x)
    assert isinstance(got, tuple) and len(got) == 2
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_three_outputs_match_eager():
    model = ThreeOutputs()
    x, y = torch.randn(4, 6), torch.randn(4, 6)
    with torch.no_grad():
        want = model(x, y)
        got = torch.compile(model, backend="mimir")(x, y)
    assert isinstance(got, tuple) and len(got) == 3
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_cache_hit_reuses_compiled_so(tmp_path):
    x = torch.randn(4, 16)

    model_a = LinearMLP()
    with torch.no_grad():
        got_a = torch.compile(model_a, backend="mimir", options={"cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got_a, model_a(x), rtol=1e-4, atol=1e-4)

    so_files = list(tmp_path.glob("*.so"))
    assert len(so_files) == 1
    first_mtime = so_files[0].stat().st_mtime_ns

    # A second model with the same architecture but different weights hits the
    # same cache entry: weights are runtime arguments, not part of the key.
    torch._dynamo.reset()
    model_b = LinearMLP()
    with torch.no_grad():
        got_b = torch.compile(model_b, backend="mimir", options={"cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got_b, model_b(x), rtol=1e-4, atol=1e-4)

    so_files = list(tmp_path.glob("*.so"))
    assert len(so_files) == 1
    assert so_files[0].stat().st_mtime_ns == first_mtime, "cache entry was rebuilt instead of reused"
    assert not torch.allclose(got_a, got_b), "different weights must give different results"


def test_cache_can_be_disabled(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        got = torch.compile(model, backend="mimir", options={"cache": False, "cache_dir": str(tmp_path)})(x)
        torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-4)
    assert not list(tmp_path.glob("*.so"))


def test_debug_dir_dumps_artifacts(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(model, backend="mimir", options={"debug_dir": str(tmp_path)})
        got = compiled(x)
        want = model(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    for suffix in ("_pre.mim", "_post.mim", ".ll", ".so"):
        matches = list(tmp_path.glob(f"mimir_graph_*{suffix}"))
        assert matches, f"expected a mimir_graph_*{suffix} artifact in {tmp_path}"
