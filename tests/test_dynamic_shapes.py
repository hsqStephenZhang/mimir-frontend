"""End-to-end tests for symbolic extents: one compiled graph serves every size.

Dynamo hands the backend a `SymInt` placeholder per symbol; the compiled function takes the symbols as
leading `Nat` arguments and evaluates output extents at call time.
"""

import shutil

import pytest
import torch
from torch import nn

import mimir_frontend.backend  # noqa: F401  (registers the "mimir" backend)
from mimir_frontend.symbols import is_sym_placeholder, placeholder_value, symbol_dim, symbol_names

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")

SIZES = (5, 3, 7)


class Elementwise(nn.Module):
    def forward(self, x):
        return torch.relu(x) * 2.0 + torch.sigmoid(x)


class SumOverAxis(nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class MeanLast(nn.Module):
    def forward(self, x):
        return x.mean(dim=-1, keepdim=True)


class Softmax(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class LayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(16)

    def forward(self, x):
        return self.norm(x)


class FlattenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4 * 5, 3)

    def forward(self, x):
        return self.fc(x.flatten(1))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.norm(self.conv(x))))


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2) + 1.0


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 32)
        self.second = nn.Linear(32, 8)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


class LinearOnSequence(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)


class MultipleOutputs(nn.Module):
    def forward(self, x):
        return torch.relu(x), x.sum(dim=1)


class CatAlongDynamicAxis(nn.Module):
    def forward(self, x):
        return torch.cat([x, x * 2.0], dim=1)


class FlattenAll(nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class BatchedMatmul(nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)


class BatchedMatmulTransposed(nn.Module):
    def forward(self, a):
        return torch.bmm(a, a.transpose(1, 2))


class Matmul(nn.Module):
    def forward(self, a, b):
        return a @ b


class SoftmaxThenMatmul(nn.Module):
    def forward(self, scores, values):
        return torch.softmax(scores, dim=-1) @ values


class ScoresByMatmul(nn.Module):
    def forward(self, x):
        return x @ x.transpose(1, 2) / 4.0


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        return torch.softmax(q @ k.transpose(1, 2) / 4.0, dim=-1) @ v


# The `3-D @ 3-D` matmul goes through the generic rank-5 `map_reduce`, whose schedule leaves a higher-order
# `wrap (d: Nat, …)` behind once an extent is only known at runtime; the LLVM backend cannot emit it.
GENERIC_MAP_REDUCE = pytest.mark.xfail(
    strict=True,
    reason="MimIR: the generic map_reduce schedule is higher-order for a runtime extent (ll backend rejects it)",
)

def case(name, factory, make_inputs, dynamic_dims, *, marks=()):
    """`make_inputs(n)` builds the inputs for size `n`; `dynamic_dims` lists `(input index, dim)` to mark dynamic."""
    return pytest.param(factory, make_inputs, dynamic_dims, id=name, marks=marks)


CASES = [
    case("elementwise-batch", Elementwise, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("elementwise-two-symbols", Elementwise, lambda n: (torch.randn(n, n + 1),), [(0, 0), (0, 1)]),
    case("sum-over-dynamic-axis", SumOverAxis, lambda n: (torch.randn(4, n),), [(0, 1)]),
    case("mean-last", MeanLast, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("softmax-batch", Softmax, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("softmax-dynamic-last-axis", Softmax, lambda n: (torch.randn(4, n),), [(0, 1)]),
    case("softmax-square", Softmax, lambda n: (torch.randn(2, n, n),), [(0, 1), (0, 2)]),
    case("layer-norm-batch", LayerNorm, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("flatten-linear-batch", FlattenLinear, lambda n: (torch.randn(n, 4, 5),), [(0, 0)]),
    case("conv-net-batch", ConvNet, lambda n: (torch.randn(n, 3, 8, 8),), [(0, 0)]),
    case("transpose-dynamic-axis", Transpose, lambda n: (torch.randn(2, n, 5),), [(0, 1)]),
    case("mlp-batch", MLP, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("linear-over-sequence", LinearOnSequence, lambda n: (torch.randn(2, n, 16),), [(0, 1)]),
    case("multiple-outputs", MultipleOutputs, lambda n: (torch.randn(n, 6),), [(0, 0)]),
    case("cat-along-dynamic-axis", CatAlongDynamicAxis, lambda n: (torch.randn(3, n),), [(0, 1)]),
    case("output-extent-is-a-product", FlattenAll, lambda n: (torch.randn(n, 16),), [(0, 0)]),
    case("bmm-batch", BatchedMatmul, lambda n: (torch.randn(n, 4, 6), torch.randn(n, 6, 5)), [(0, 0), (1, 0)]),
    case("bmm-contracted-axis", BatchedMatmul, lambda n: (torch.randn(2, 4, n), torch.randn(2, n, 5)), [(0, 2), (1, 1)]),
    case("bmm-rows", BatchedMatmul, lambda n: (torch.randn(2, n, 6), torch.randn(2, 6, 5)), [(0, 1)]),
    case("bmm-transposed-self", BatchedMatmulTransposed, lambda n: (torch.randn(2, n, 6),), [(0, 1)]),
    case("matmul-contracted-axis", Matmul, lambda n: (torch.randn(4, n), torch.randn(n, 5)), [(0, 1), (1, 0)]),
    case("softmax-then-matmul", SoftmaxThenMatmul, lambda n: (torch.randn(2, n, 4), torch.randn(2, 4, 6)), [(0, 1)]),
    case("attention-scores", ScoresByMatmul, lambda n: (torch.randn(2, n, 6),), [(0, 1)], marks=GENERIC_MAP_REDUCE),
    case("attention-block", Attention, lambda n: (torch.randn(2, n, 16),), [(0, 1)], marks=GENERIC_MAP_REDUCE),
]


def assert_matches_eager(got, want):
    if isinstance(want, tuple):
        assert len(got) == len(want)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("factory, make_inputs, dynamic_dims", CASES)
def test_one_compiled_graph_serves_every_size(factory, make_inputs, dynamic_dims, tmp_path):
    torch.manual_seed(0)
    module = factory().eval()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(module, backend="mimir", fullgraph=True, options={"cache_dir": str(tmp_path)})

    for n in SIZES:
        args = make_inputs(n)
        for index, dim in dynamic_dims:
            torch._dynamo.mark_dynamic(args[index], dim)
        with torch.no_grad():
            assert_matches_eager(compiled(*args), module(*args))

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


def test_a_static_batch_compiles_without_symbols(tmp_path):
    module = MLP().eval()
    torch._dynamo.reset()
    compiled = torch.compile(module, backend="mimir", fullgraph=True, options={"cache_dir": str(tmp_path)})
    x = torch.randn(4, 16)
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), module(x), rtol=1e-4, atol=1e-4)


def test_fallback_decompositions_refuse_symbolic_extents(tmp_path):
    class Expanded(nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardswish(x)

    module = Expanded().eval()
    torch._dynamo.reset()
    x = torch.randn(5, 16)
    torch._dynamo.mark_dynamic(x, 0)
    compiled = torch.compile(module, backend="mimir", fullgraph=True, options={"cache_dir": str(tmp_path)})
    with torch.no_grad(), pytest.raises(
        torch._dynamo.exc.BackendCompilerFailed,
        match="dynamic shapes are not supported together with fallback decompositions",
    ):
        compiled(x)


def test_symbol_dim_reads_the_symbols_dynamo_records():
    seen = {}

    def capture(gm, example_inputs):
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        seen["syms"] = [n.name for n in placeholders if is_sym_placeholder(n)]
        seen["dims"] = [
            tuple(symbol_dim(d) for d in placeholder_value(n).shape) for n in placeholders if not is_sym_placeholder(n)
        ]
        return gm.forward

    torch._dynamo.reset()
    x = torch.randn(5, 16)
    torch._dynamo.mark_dynamic(x, 0)
    torch.compile(lambda t: t * 2.0, backend=capture, fullgraph=True)(x)

    (symbol,) = seen["syms"]
    assert seen["dims"] == [(symbol, 16)]
    assert symbol_dim(7) == 7


def test_symbol_names_put_sym_placeholders_first_then_symbols_in_order_of_use():
    assert symbol_names(["s1"], [(2, "s0"), ("s0", "s1", 3), ("s2",)]) == ["s1", "s0", "s2"]
    assert symbol_names([], [(2, 3)]) == []
