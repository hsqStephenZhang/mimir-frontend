"""End-to-end tests for the "mimir" torch.compile backend.

Each test lowers a Dynamo graph through MimIR to a shared library (requires
clang on PATH) and compares the JIT-compiled result against eager PyTorch.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize("dim", [1, 2, -1])
def test_softmax_folded_singleton_axes_match_eager(dim):
    class SoftmaxFoldedSingleton(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=dim)

    check_against_eager(SoftmaxFoldedSingleton(), torch.randn(2, 1, 4))


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


def check_against_eager(model, *inputs, options=None, equal_nan=False):
    with torch.no_grad():
        want = model(*inputs)
        compiled = torch.compile(model, backend=mimir_backend, options=options)
        got = compiled(*inputs)
    torch.testing.assert_close(
        got, want, rtol=1e-4, atol=1e-4, equal_nan=equal_nan
    )


def test_linear_mlp_matches_eager():
    check_against_eager(LinearMLP(), torch.randn(4, 16))


def test_tensor_T_matches_eager():
    class MatrixTranspose(torch.nn.Module):
        def forward(self, x, y):
            # Consuming the view keeps `.T` inside Dynamo's compiled graph.
            return x.T @ y

    check_against_eager(
        MatrixTranspose(), torch.randn(3, 5), torch.randn(3, 2)
    )


def test_tril_matches_eager():
    class LowerTriangle(torch.nn.Module):
        def forward(self, x):
            return torch.tril(x, diagonal=-1)

    check_against_eager(LowerTriangle(), torch.randn(5, 5))


def test_log_softmax_flip_and_narrow_match_eager():
    class SequenceHelpers(torch.nn.Module):
        def forward(self, x):
            return torch.log_softmax(x.flip(1).narrow(1, 1, 3), dim=1)

    check_against_eager(SequenceHelpers(), torch.randn(2, 6))


def test_leaky_relu_matches_eager():
    class LeakyRelu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.leaky_relu(x, negative_slope=0.2)

    check_against_eager(LeakyRelu(), torch.randn(2, 6))


def test_gather_negative_dim_matches_eager():
    class Gather(torch.nn.Module):
        def forward(self, x, index):
            return torch.gather(x, -1, index)

    index = torch.tensor([[3, 0], [1, 2]], dtype=torch.int64)
    check_against_eager(Gather(), torch.randn(2, 4), index)


def test_scatter_src_negative_dim_matches_eager():
    class ScatterSrc(torch.nn.Module):
        def forward(self, x, index, src):
            return torch.ops.aten.scatter.src(x, -1, index, src)

    index = torch.tensor([[3, 0], [1, 2]], dtype=torch.int64)
    check_against_eager(
        ScatterSrc(), torch.randn(2, 4), index, torch.randn(2, 2)
    )


def test_scatter_value_negative_dim_matches_eager():
    class ScatterValue(torch.nn.Module):
        def forward(self, x, index):
            return torch.ops.aten.scatter.value(x, -1, index, 2.5)

    index = torch.tensor([[3, 0], [1, 2]], dtype=torch.int64)
    check_against_eager(ScatterValue(), torch.randn(2, 4), index)


@pytest.mark.parametrize(
    ("ceil_mode", "count_include_pad", "divisor_override"),
    [(False, True, None), (True, True, None), (True, False, None), (True, False, 7)],
)
def test_avg_pool2d_parameter_semantics_match_eager(
    ceil_mode, count_include_pad, divisor_override
):
    class AvgPool(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.avg_pool2d.default(
                x,
                [3, 3],
                [2, 2],
                [1, 1],
                ceil_mode,
                count_include_pad,
                divisor_override,
            )

    # The last ceil-mode window extends beyond explicit padding. This
    # distinguishes PyTorch's pool_size from the implementation-only pad.
    check_against_eager(AvgPool(), torch.randn(1, 2, 4, 4))


def test_lighthouse_max_pool1d_matches_eager():
    class MaxPool1d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.max_pool1d.default(
                x, [8], [1], [4], [3], False
            )

    check_against_eager(MaxPool1d(), torch.randn(2, 3, 32))


def test_lighthouse_avg_pool1d_matches_eager():
    class AvgPool1d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.avg_pool1d.default(
                x, [8], [1], [4], False, True
            )

    check_against_eager(AvgPool1d(), torch.randn(2, 3, 32))


def test_lighthouse_adaptive_avg_pool1d_matches_eager():
    class AdaptiveAvgPool1d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.adaptive_avg_pool1d.default(x, [1])

    check_against_eager(AdaptiveAvgPool1d(), torch.randn(2, 16, 9))


def test_lighthouse_adaptive_avg_pool3d_matches_eager():
    class AdaptiveAvgPool3d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.adaptive_avg_pool3d.default(x, [1, 1, 1])

    check_against_eager(AdaptiveAvgPool3d(), torch.randn(2, 3, 4, 5, 6))


def test_lighthouse_max_pool3d_matches_eager():
    class MaxPool3d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.max_pool3d.default(
                x, [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], True
            )

    check_against_eager(MaxPool3d(), torch.randn(2, 3, 5, 7, 9))


def test_lighthouse_avg_pool3d_matches_eager():
    class AvgPool3d(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.avg_pool3d.default(
                x, [3, 3, 3], [2, 2, 2], [1, 1, 1], True, False, None
            )

    check_against_eager(AvgPool3d(), torch.randn(2, 3, 5, 7, 9))


def test_lighthouse_constant_pad_matches_eager():
    class ConstantPad(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.pad.default(
                x, [1, 0], "constant", 0.0
            )

    check_against_eager(ConstantPad(), torch.randn(2, 3, 4, 7))


def test_lighthouse_cumprod_matches_eager():
    class Cumprod(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.cumprod.default(x, 1)

    check_against_eager(Cumprod(), torch.rand(4, 8))


def test_lighthouse_roll_matches_eager():
    class Roll(torch.nn.Module):
        def forward(self, x):
            return torch.ops.aten.roll.default(x, [1, -2], [1, 2])

    check_against_eager(Roll(), torch.randn(2, 3, 4))


def test_lighthouse_vit_double_unfold_matches_eager():
    class Patchify(torch.nn.Module):
        def forward(self, image):
            return image.unfold(2, 4, 4).unfold(3, 4, 4)

    check_against_eager(Patchify(), torch.randn(2, 3, 8, 8))


def test_sum_empty_dimensions_matches_eager_reduce_all():
    class SumEmptyDimensions(torch.nn.Module):
        def forward(self, x):
            return x + torch.sum(x, dim=[])

    check_against_eager(SumEmptyDimensions(), torch.randn(2, 3, 4))


def test_mean_empty_dimensions_matches_eager_reduce_all():
    class MeanEmptyDimensions(torch.nn.Module):
        def forward(self, x):
            return x + torch.mean(x, dim=[])

    check_against_eager(MeanEmptyDimensions(), torch.randn(2, 3, 4))


def test_amax_empty_dimensions_matches_eager_reduce_all():
    class AmaxEmptyDimensions(torch.nn.Module):
        def forward(self, x):
            return x + torch.amax(x, dim=[])

    check_against_eager(AmaxEmptyDimensions(), -torch.rand(2, 3, 4))


def test_var_mean_scalar_corrections_match_eager():
    class VarMeanCorrections(torch.nn.Module):
        def forward(self, x):
            negative, _ = torch.var_mean(x, dim=-1, correction=-1)
            equal, _ = torch.var_mean(x, dim=-1, correction=3)
            greater, _ = torch.var_mean(x, dim=-1, correction=4)
            fractional, _ = torch.var_mean(x, dim=-1, correction=0.5)
            return negative, equal, greater, fractional

    check_against_eager(
        VarMeanCorrections(), torch.randn(2, 3), equal_nan=True
    )


@pytest.mark.parametrize("unbiased", [False, True])
def test_var_mean_dim_overload_matches_eager(unbiased):
    class VarMeanDim(torch.nn.Module):
        def forward(self, x):
            variance, mean = torch.ops.aten.var_mean.dim(
                x, [-1], unbiased, True
            )
            return variance, mean

    check_against_eager(VarMeanDim(), torch.randn(2, 3), equal_nan=True)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_dim_extrema_values_indices_and_ties_match_eager(kind):
    class DimExtrema(torch.nn.Module):
        def forward(self, x):
            op = torch.max if kind == "max" else torch.min
            values, indices = op(x, dim=-1, keepdim=True)
            return values, indices.to(torch.float32)

    # Repeated extrema and NaNs both require first-index tie breaking.
    x = torch.tensor(
        [[3.0, 3.0, -2.0], [1.0, -4.0, -4.0], [float("nan"), 5.0, float("nan")]]
    )
    check_against_eager(DimExtrema(), x, equal_nan=True)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_dim_extrema_folded_singleton_axis_matches_eager(kind):
    class DimExtremaSingleton(torch.nn.Module):
        def forward(self, x):
            op = torch.max if kind == "max" else torch.min
            values, indices = op(x, dim=1)
            return values, indices.to(torch.float32)

    check_against_eager(DimExtremaSingleton(), torch.randn(2, 1, 3))


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


@pytest.mark.parametrize("torch_op,alpha", [(torch.add, 3.0), (torch.sub, 4.0)])
def test_add_sub_alpha_matches_eager(torch_op, alpha):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch_op(x, y, alpha=alpha)

    check_against_eager(
        Model(),
        torch.randn(2, 3),
        torch.randn(2, 3),
    )


@pytest.mark.parametrize("torch_op,alpha", [(torch.add, 3.0), (torch.sub, 4.0)])
def test_add_sub_scalar_alpha_matches_eager(torch_op, alpha):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch_op(x, 2.0, alpha=alpha)

    check_against_eager(Model(), torch.randn(2, 3))


@pytest.mark.parametrize("torch_op", [torch.add, torch.sub])
def test_add_sub_scalar_lhs_alpha_matches_eager(torch_op):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch_op(2.0, x, alpha=3.0)

    check_against_eager(Model(), torch.randn(2, 3))


@pytest.mark.parametrize("torch_op", [torch.addcmul, torch.addcdiv])
def test_addc_ops_broadcast_and_value_match_eager(torch_op):
    class Model(torch.nn.Module):
        def forward(self, self_tensor, tensor1, tensor2):
            return torch_op(self_tensor, tensor1, tensor2, value=-0.75)

    check_against_eager(
        Model(),
        torch.randn(2, 1),
        torch.randn(1, 3),
        torch.rand(2, 3) + 0.5,
    )


@pytest.mark.parametrize("torch_op", [torch.addcmul, torch.addcdiv])
def test_addc_ops_zero_value_preserves_ieee_nan(torch_op):
    class Model(torch.nn.Module):
        def forward(self, self_tensor, tensor1, tensor2):
            return torch_op(self_tensor, tensor1, tensor2, value=0.0)

    check_against_eager(
        Model(),
        torch.tensor([1.0, 1.0]),
        torch.tensor([float("inf"), 2.0]),
        torch.tensor([0.0, 3.0]),
        equal_nan=True,
    )
@pytest.mark.parametrize("torch_op", [torch.maximum, torch.minimum])
def test_extrema_propagates_nan_from_either_operand(torch_op):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch_op(x, y)

    check_against_eager(
        Model(),
        torch.tensor([float("nan"), 1.0]),
        torch.tensor([2.0, float("nan")]),
        equal_nan=True,
    )


def test_relu_preserves_nan():
    check_against_eager(
        torch.nn.ReLU(),
        torch.tensor([float("nan"), -1.0, 0.0, 2.0]),
        equal_nan=True,
    )


def test_mish_matches_eager():
    check_against_eager(
        torch.nn.Mish().eval(),
        torch.tensor([-20.0, -1.0, 0.0, 1.0, 20.0, float("nan")]),
        equal_nan=True,
    )


@pytest.mark.parametrize("keepdim", [False, True])
def test_logsumexp_matches_eager(keepdim):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.logsumexp(x, dim=1, keepdim=keepdim)

    check_against_eager(
        Model(),
        torch.tensor([
            [float("-inf"), float("-inf"), float("-inf")],
            [float("inf"), 1.0, -1.0],
            [1000.0, 999.0, -1000.0],
        ]),
    )


def test_torch_min_tensor_overload_matches_eager():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.min(x, torch.tensor(0.5))

    check_against_eager(Model(), torch.tensor([-1.0, 0.5, 2.0]))


def test_native_group_norm_tuple_matches_eager():
    class Model(torch.nn.Module):
        def forward(self, x, weight, bias):
            return torch.ops.aten.native_group_norm.default(
                x, weight, bias, 2, 4, 15, 2, 1e-5
            )

    check_against_eager(
        Model(), torch.randn(2, 4, 3, 5), torch.randn(4), torch.randn(4)
    )


def test_hardtanh_matches_boundaries_and_preserves_nan():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)

    check_against_eager(
        Model(),
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, float("nan")]),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "min_val,max_val",
    [
        (-1.0, None),
        (None, 1.0),
        (-1.0, 1.0),
        (float("nan"), None),
        (None, float("nan")),
    ],
)
def test_clamp_scalar_bounds_preserve_input_nan(min_val, max_val):
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.clamp(x, min=min_val, max=max_val)

    check_against_eager(
        Model(),
        torch.tensor([float("nan"), -2.0, 0.0, 2.0]),
        equal_nan=True,
    )


def test_threshold_matches_reference_and_preserves_nan():
    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.threshold(x, threshold=0.5, value=-2.0)

    check_against_eager(
        Model(),
        torch.tensor([-1.0, 0.5, 2.0, float("nan")]),
        equal_nan=True,
    )


@pytest.mark.parametrize("shape", [(4, 8), (2, 4, 8)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_rank_and_bias_matches_eager(shape, with_bias):
    check_against_eager(torch.nn.Linear(8, 5, bias=with_bias).eval(), torch.randn(*shape))


def test_conv_matches_eager():
    check_against_eager(SmallConv(), torch.randn(2, 1, 8, 8))


def test_conv1d_matches_eager():
    check_against_eager(
        torch.nn.Conv1d(4, 8, 3, stride=2, padding=1).eval(),
        torch.randn(1, 4, 16),
    )


def test_conv3d_matches_eager():
    check_against_eager(
        torch.nn.Conv3d(
            4, 8, (3, 2, 3), stride=2, padding=(1, 0, 1), groups=2
        ).eval(),
        torch.randn(2, 4, 7, 8, 9),
    )


def test_conv_transpose2d_matches_eager():
    check_against_eager(
        torch.nn.ConvTranspose2d(
            4,
            6,
            (3, 2),
            stride=(2, 3),
            padding=(1, 0),
            output_padding=(1, 2),
            dilation=(1, 2),
            groups=2,
        ).eval(),
        torch.randn(2, 4, 3, 4),
    )


def test_conv_transpose1d_matches_eager():
    check_against_eager(
        torch.nn.ConvTranspose1d(
            4,
            6,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            dilation=2,
            groups=2,
        ).eval(),
        torch.randn(2, 4, 5),
    )


def test_conv_transpose3d_unit_stride_matches_eager():
    check_against_eager(
        torch.nn.ConvTranspose3d(
            4,
            6,
            (3, 2, 4),
            padding=(1, 0, 1),
            dilation=(1, 2, 1),
            groups=2,
        ).eval(),
        torch.randn(2, 4, 3, 4, 5),
    )


def test_conv_transpose3d_strided_grouped_matches_eager():
    check_against_eager(
        torch.nn.ConvTranspose3d(
            4,
            6,
            (3, 2, 3),
            stride=(2, 2, 2),
            padding=(1, 0, 1),
            output_padding=(1, 1, 1),
            groups=2,
        ).eval(),
        torch.randn(2, 4, 3, 4, 5),
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_matches_eager(approximate):
    class Gelu(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.gelu(x, approximate=approximate)

    check_against_eager(Gelu(), torch.linspace(-3, 3, 24).reshape(2, 3, 4))


@pytest.mark.parametrize(
    "function",
    [
        lambda x: torch.selu(x),
        lambda x: torch.nn.functional.elu(x, alpha=0.5),
        lambda x: torch.nn.functional.hardsigmoid(x),
        lambda x: torch.nn.functional.softplus(x, beta=2.0, threshold=10.0),
    ],
)
def test_additional_activation_matches_eager(function):
    class Activation(torch.nn.Module):
        def forward(self, x):
            return function(x)

    check_against_eager(Activation(), torch.linspace(-4.0, 4.0, 17))


def test_scalar_tensor_output_matches_eager():
    class ScalarLoss(torch.nn.Module):
        def forward(self, x, target):
            return torch.mean((x - target) ** 2)

    check_against_eager(
        ScalarLoss(), torch.randn(3, 5), torch.randn(3, 5)
    )


@pytest.mark.parametrize("op", [torch.argmax, torch.argmin])
def test_int64_reduction_output_matches_eager(op):
    class IndexReduction(torch.nn.Module):
        def forward(self, x):
            return op(x, dim=1)

    check_against_eager(
        IndexReduction(), torch.tensor([[3.0, -1.0, 2.0], [0.5, 4.0, 1.0]])
    )


@pytest.mark.parametrize(
    "function",
    [
        lambda x: torch.norm(x, p="fro"),
        lambda x: torch.norm(x, p=2, dim=1, keepdim=True),
    ],
)
def test_norm2_matches_eager(function):
    class Norm(torch.nn.Module):
        def forward(self, x):
            return function(x)

    check_against_eager(Norm(), torch.randn(2, 3, 4))


def test_repeat_matches_eager():
    class Repeat(torch.nn.Module):
        def forward(self, x):
            return x.repeat(2, 1, 3)

    check_against_eager(Repeat(), torch.arange(20.0).reshape(4, 5))


def test_getitem_tensor_index_matches_eager():
    class Index(torch.nn.Module):
        def forward(self, weight, index):
            return weight[index]

    check_against_eager(
        Index(),
        torch.arange(32.0).reshape(8, 4),
        torch.tensor([[0, 3, 7], [2, 1, 5]]),
    )


def test_i64_tensor_scalar_sub_matches_eager():
    class Sub(torch.nn.Module):
        def forward(self, weight, x):
            return weight[x - 1]

    check_against_eager(
        Sub(),
        torch.arange(40.0).reshape(10, 4),
        torch.tensor([[1, 4, 9]], dtype=torch.int64),
    )


def test_diff_with_prepend_matches_eager():
    class Diff(torch.nn.Module):
        def forward(self, x, prepend):
            return torch.diff(x, dim=-1, prepend=prepend)

    check_against_eager(
        Diff(),
        torch.randn(2, 4),
        torch.randn(2, 1),
    )


def test_i64_tensor_scalar_ne_matches_eager():
    class Ne(torch.nn.Module):
        def forward(self, mask, x, y):
            return torch.where(mask != 1, x, y)

    check_against_eager(
        Ne(),
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
        torch.randn(1, 3),
        torch.randn(1, 3),
    )


def test_bool_cumsum_consumed_by_embedding_matches_eager():
    class Cumsum(torch.nn.Module):
        def forward(self, weight, mask):
            return weight[torch.cumsum(mask != 1, dim=-1)]

    check_against_eager(
        Cumsum(),
        torch.randn(8, 3),
        torch.tensor([[0, 1, 2, 1]], dtype=torch.int64),
    )


def test_packed_sequence_indices_consumed_by_embedding_matches_eager():
    class PackedSequence(torch.nn.Module):
        def forward(self, weight, position_ids):
            first_dummy = position_ids[:, :1] - 1
            position_diff = torch.diff(
                position_ids, prepend=first_dummy, dim=-1
            )
            packed = (position_diff != 1).cumsum(-1)
            return weight[packed]

    check_against_eager(
        PackedSequence(),
        torch.randn(4, 3),
        torch.tensor([[0, 1, 0, 1, 2, 0]], dtype=torch.int64),
    )


def test_broadcasted_packed_sequence_indexing_matches_eager():
    class PackedGrid(torch.nn.Module):
        def forward(self, weight, position_ids):
            first_dummy = position_ids[:, :1] - 1
            packed = (
                torch.diff(position_ids, prepend=first_dummy, dim=-1) != 1
            ).cumsum(-1)
            batch = torch.arange(position_ids.shape[0])[:, None, None, None]
            query = torch.arange(position_ids.shape[1])[None, None, :, None]
            key = torch.arange(position_ids.shape[1])[None, None, None, :]
            return weight[packed[batch, query]] + weight[packed[batch, key]]

    check_against_eager(
        PackedGrid(),
        torch.randn(4, 3),
        torch.tensor([[0, 1, 0, 1]], dtype=torch.int64),
    )


def test_broadcast_i64_comparison_matches_eager():
    class Compare(torch.nn.Module):
        def forward(self, lhs, rhs, x, y):
            return torch.where(lhs <= rhs, x, y)

    check_against_eager(
        Compare(),
        torch.tensor([[0], [2]], dtype=torch.int64),
        torch.tensor([[1, 3, 2]], dtype=torch.int64),
        torch.randn(2, 3),
        torch.randn(2, 3),
    )


def test_rank0_bool_tensor_broadcast_matches_eager():
    class ScalarMask(torch.nn.Module):
        def forward(self, x):
            mask = x.new_ones((), dtype=torch.bool)
            return torch.where(mask, x, -x)

    check_against_eager(ScalarMask(), torch.randn(2, 3))


def test_singleton_tensor_multiply_matches_eager():
    class MatrixSingletonMultiply(torch.nn.Module):
        def forward(self, matrix, scalar_tensor):
            return matrix * scalar_tensor

    check_against_eager(
        MatrixSingletonMultiply(),
        torch.rand(2, 8),
        torch.tensor([3.14]),
    )


def test_two_tensor_advanced_index_matches_eager():
    class Index2D(torch.nn.Module):
        def forward(self, x, rows, columns):
            return x[rows, columns]

    check_against_eager(
        Index2D(),
        torch.randn(3, 4),
        torch.tensor([[0], [2]], dtype=torch.int64),
        torch.tensor([[0, 3, 1]], dtype=torch.int64),
    )


def test_where_tensor_and_python_scalar_matches_eager():
    class WhereScalar(torch.nn.Module):
        def forward(self, mask, x):
            return torch.where(mask != 0, x, -3.5)

    check_against_eager(
        WhereScalar(),
        torch.tensor([[0, 1, 0]], dtype=torch.int64),
        torch.randn(2, 3),
    )


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


def test_tensor_matrix_matmul_matches_eager():
    class TensorMatrixMatmul(torch.nn.Module):
        def forward(self, lhs, rhs):
            return lhs @ rhs

    check_against_eager(
        TensorMatrixMatmul(),
        torch.randn(2, 3, 5),
        torch.randn(5, 7),
    )


def test_bmm_matches_eager():
    class BatchedMatmul(torch.nn.Module):
        def forward(self, lhs, rhs):
            return torch.bmm(lhs, rhs)

    check_against_eager(
        BatchedMatmul(),
        torch.randn(2, 3, 4),
        torch.randn(2, 4, 5),
    )


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape",
    [
        ((5,), (5, 7)),
        ((3, 5), (5,)),
        ((2, 3, 5), (5,)),
        ((5,), (2, 5, 7)),
        ((3, 5, 7), (2, 3, 7, 11)),
    ],
)
def test_matmul_rank_dispatch_matches_eager(lhs_shape, rhs_shape):
    class MatmulRankDispatch(torch.nn.Module):
        def forward(self, lhs, rhs):
            return lhs @ rhs

    check_against_eager(
        MatmulRankDispatch(), torch.randn(lhs_shape), torch.randn(rhs_shape)
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


def test_group_norm_matches_eager():
    check_against_eager(
        torch.nn.GroupNorm(2, 4).eval(), torch.randn(2, 4, 3, 5)
    )


def test_instance_norm_matches_eager():
    check_against_eager(
        torch.nn.InstanceNorm2d(4).eval(), torch.randn(2, 4, 3, 5)
    )


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_smooth_l1_mean_matches_eager(beta):
    class SmoothL1(torch.nn.Module):
        def forward(self, x, target):
            return torch.nn.functional.smooth_l1_loss(x, target, beta=beta)

    check_against_eager(SmoothL1(), torch.randn(3, 5), torch.randn(3, 5))


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


def test_embedding_out_of_range_reaches_compiled_runtime(tmp_path):
    code = """
import torch
from mimir_frontend.backend import mimir_backend

model = torch.nn.Embedding(4, 3).eval()
indices = torch.tensor([[0, 4]], dtype=torch.int64)
with torch.no_grad():
    torch.compile(model, backend=mimir_backend)(indices)
"""
    env = os.environ.copy()
    env["MIMIR_CACHE_DIR"] = str(tmp_path / "mimir-jit-cache")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "embedding index out of range" in output
    assert "IndexError: index out of range in self" not in output


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
