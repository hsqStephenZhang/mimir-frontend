"""End-to-end tests for the "mimir" torch.compile backend.

Each test lowers a Dynamo graph through MimIR to a shared library (requires
clang on PATH) and compares the JIT-compiled result against eager PyTorch.
"""

import shutil

import pytest
import torch

from mimir_frontend.backend import EXEC_PLUGINS, mimir_backend

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


class AddMM(torch.nn.Module):
    def __init__(self, *, beta=1.0, alpha=1.0):
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, self_tensor, mat1, mat2):
        return torch.addmm(
            self_tensor, mat1, mat2, beta=self.beta, alpha=self.alpha
        )


class TinyAttentionFFN(torch.nn.Module):
    """A transformer-style block without LayerNorm, for frontend coverage."""

    def __init__(self, d=8, hidden=16):
        super().__init__()
        self.q = torch.nn.Linear(d, d)
        self.k = torch.nn.Linear(d, d)
        self.v = torch.nn.Linear(d, d)
        self.proj = torch.nn.Linear(d, d)
        self.ff1 = torch.nn.Linear(d, hidden)
        self.ff2 = torch.nn.Linear(hidden, d)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        x = x + self.proj(attention @ v)
        return x + self.ff2(torch.relu(self.ff1(x)))


class TinyTransformerBlock(torch.nn.Module):
    """A complete single-head transformer block with precomputed static width."""

    def __init__(self, d=8, hidden=16):
        super().__init__()
        self.q = torch.nn.Linear(d, d)
        self.k = torch.nn.Linear(d, d)
        self.v = torch.nn.Linear(d, d)
        self.proj = torch.nn.Linear(d, d)
        self.ff1 = torch.nn.Linear(d, hidden)
        self.ff2 = torch.nn.Linear(hidden, d)
        self.norm1 = torch.nn.LayerNorm(d)
        self.norm2 = torch.nn.LayerNorm(d)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        x = self.norm1(x + self.proj(attention @ v))
        return self.norm2(x + self.ff2(torch.relu(self.ff1(x))))


@pytest.fixture(autouse=True)
def _reset_dynamo(monkeypatch, tmp_path_factory):
    # Hermetic per-test-run JIT cache: don't read from or pollute the user cache.
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path_factory.getbasetemp() / "mimir-jit-cache"))
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def check_against_eager(model, *inputs, options=None):
    with torch.no_grad():
        want = model(*inputs)
        compiled = torch.compile(model, backend=mimir_backend, options=options)
        got = compiled(*inputs)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_linear_mlp_matches_eager():
    check_against_eager(LinearMLP(), torch.randn(4, 16))


@pytest.mark.parametrize("self_shape", [(4,), (1, 4), (2, 1), (2, 4)])
def test_addmm_matches_eager(self_shape):
    check_against_eager(
        AddMM(beta=0.5, alpha=2.0),
        torch.randn(*self_shape),
        torch.randn(2, 3),
        torch.randn(3, 4),
    )


def test_addmm_beta_zero_does_not_propagate_self_nan():
    check_against_eager(
        AddMM(beta=0.0, alpha=1.5),
        torch.full((4,), float("nan")),
        torch.randn(2, 3),
        torch.randn(3, 4),
    )


@pytest.mark.parametrize("shape", [(4, 8), (2, 4, 8)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_rank_and_bias_matches_eager(shape, with_bias):
    check_against_eager(torch.nn.Linear(8, 5, bias=with_bias).eval(), torch.randn(*shape))


def test_conv_matches_eager():
    check_against_eager(SmallConv(), torch.randn(2, 1, 8, 8))


def test_batch_norm_residual_block_matches_eager():
    class ResidualBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(4)
            self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(4)

        def forward(self, x):
            identity = x
            x = torch.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += identity
            return torch.relu(x)

    check_against_eager(
        ResidualBlock().eval(),
        torch.randn(1, 4, 8, 8),
    )


def test_batch_norm_without_affine_matches_eager():
    check_against_eager(
        torch.nn.BatchNorm2d(4, affine=False).eval(),
        torch.randn(1, 4, 4, 4),
    )


@pytest.mark.slow
def test_torchvision_resnet18_matches_eager():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.resnet18(weights=None).eval()
    # A smaller static image exercises the complete architecture while
    # keeping the loop-based reference lowering suitable for a test run.
    check_against_eager(model, torch.randn(1, 3, 32, 32))


@pytest.mark.slow
@pytest.mark.parametrize(
    "constructor_name",
    ["squeezenet1_1", "densenet_slim", "mobilenet_v2"],
)
def test_torchvision_additional_classifiers_match_eager(constructor_name):
    torchvision = pytest.importorskip("torchvision")
    if constructor_name == "densenet_slim":
        # Preserve all DenseNet transitions and multi-input DenseBlock cats
        # without making this correctness test a 120-convolution stress test.
        model = torchvision.models.densenet.DenseNet(
            growth_rate=8,
            block_config=(2, 2, 2, 2),
            num_init_features=16,
            bn_size=2,
            num_classes=10,
        ).eval()
    else:
        model = getattr(torchvision.models, constructor_name)(weights=None).eval()
    check_against_eager(model, torch.randn(1, 3, 32, 32))


def test_transformer_attention_ffn_matches_eager():
    check_against_eager(TinyAttentionFFN(), torch.randn(4, 8))


def test_transformer_block_matches_eager():
    check_against_eager(TinyTransformerBlock().eval(), torch.randn(2, 4, 8))


def test_rank4_matmul_matches_eager():
    class Rank4Matmul(torch.nn.Module):
        def forward(self, lhs, rhs):
            return lhs @ rhs

    check_against_eager(
        Rank4Matmul(),
        torch.randn(2, 3, 4, 5),
        torch.randn(2, 3, 5, 6),
    )


def test_layer_norm_matches_eager():
    check_against_eager(torch.nn.LayerNorm(8).eval(), torch.randn(4, 8))


def test_native_layer_norm_three_results_match_eager():
    class NativeLayerNorm(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.native_layer_norm(x, (8,), weight, bias, 1e-5)

    check_against_eager(
        NativeLayerNorm(), torch.randn(4, 8), torch.randn(8), torch.randn(8)
    )


def test_pytorch_fallback_decomposition_matches_eager():
    class HardSwishLayerNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(8)

        def forward(self, x):
            return self.norm(torch.nn.functional.hardswish(x))

    check_against_eager(HardSwishLayerNorm().eval(), torch.randn(4, 8))


def test_rms_norm_singleton_broadcast_matches_eager():
    class RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(16))

        def forward(self, x):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(variance + 1e-6))

    check_against_eager(RMSNorm().eval(), torch.randn(1, 5, 16))


def test_embedding_with_i64_indices_matches_eager():
    check_against_eager(
        torch.nn.Embedding(32, 16).eval(),
        torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64),
    )


def test_registered_by_name():
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, backend="mimir")(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_execution_pipeline_uses_local_cfg_without_closure_conversion():
    assert "clos" not in EXEC_PLUGINS


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


def test_max_fp_iters_option_reaches_driver_flags(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(
            model,
            backend="mimir",
            options={"max_fp_iters": 64, "cache": False, "cache_dir": str(tmp_path)},
        )
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4, atol=1e-4)


def test_profile_is_emitted_when_fixed_point_fails(tmp_path, capsys):
    model = LinearMLP()
    x = torch.randn(4, 16)
    compiled = torch.compile(
        model,
        backend="mimir",
        options={
            "max_fp_iters": 1,
            "profile": "summary",
            "cache": False,
            "cache_dir": str(tmp_path),
        },
    )

    with pytest.raises(Exception, match="fixed point"):
        compiled(x)
    assert "phase profile" in capsys.readouterr().err


def test_profile_summary_prints_report(tmp_path, capsys):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(
            model, backend="mimir", options={"profile": "summary", "cache_dir": str(tmp_path)}
        )
        got = compiled(x)
        torch.testing.assert_close(got, model(x), rtol=1e-4, atol=1e-4)
    assert "Phase profile (flat):" in capsys.readouterr().err


def test_profile_trace_writes_json(tmp_path):
    model = LinearMLP()
    x = torch.randn(4, 16)
    with torch.no_grad():
        compiled = torch.compile(
            model,
            backend="mimir",
            options={"profile": "trace", "debug_dir": str(tmp_path), "cache_dir": str(tmp_path)},
        )
        compiled(x)
    traces = list(tmp_path.glob("*_profile.json"))
    assert traces and "traceEvents" in traces[0].read_text()


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
