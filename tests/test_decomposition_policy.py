import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from mimir_frontend.decomposition import (
    DecompositionPolicy,
    MIMIR_DECOMPOSITION_POLICY,
    apply_decomposition_policy,
    decomposition_table,
    resolve_decomposition_policy,
)


def _targets(gm):
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def test_mimir_policy_uses_registered_fallback_and_preserves_layer_norm():
    def model(x):
        x = torch.nn.functional.hardswish(x)
        return torch.nn.functional.layer_norm(x, (4,))

    x = torch.randn(2, 4)
    gm = make_fx(model)(x)
    lowered = apply_decomposition_policy(gm, [x], MIMIR_DECOMPOSITION_POLICY)
    targets = _targets(lowered)

    assert torch.ops.aten.hardswish.default not in targets
    assert torch.ops.aten.clamp.default in targets
    assert torch.ops.aten.native_layer_norm.default in targets


def test_preserve_wins_when_an_operator_is_in_both_sets():
    op = torch.ops.aten.hardswish.default
    policy = DecompositionPolicy(fallback=(op,), preserve=(op,))
    assert op not in decomposition_table(policy)


def test_policy_selection_is_explicit():
    assert resolve_decomposition_policy(None) is MIMIR_DECOMPOSITION_POLICY
    assert resolve_decomposition_policy("mimir") is MIMIR_DECOMPOSITION_POLICY
    assert decomposition_table(resolve_decomposition_policy("none")) == {}
    with pytest.raises(ValueError, match="decomposition_policy"):
        resolve_decomposition_policy("all")


def test_policy_does_not_retrace_a_graph_without_a_fallback_trigger():
    gm = make_fx(lambda x: torch.nn.functional.layer_norm(x, (4,)))(
        torch.randn(2, 4)
    )
    assert apply_decomposition_policy(
        gm, [torch.randn(2, 4)], MIMIR_DECOMPOSITION_POLICY
    ) is gm


def test_mimir_policy_retraces_python_multihead_attention_to_aten():
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(16, 4)

        def forward(self, x):
            return self.attn(x, x, x)[0]

    x = torch.randn(8, 2, 16)
    gm, _ = torch._dynamo.export(Attention().eval())(x)

    lowered = apply_decomposition_policy(gm, [x], MIMIR_DECOMPOSITION_POLICY)
    targets = _targets(lowered)

    assert torch.nn.functional.multi_head_attention_forward not in targets
    assert torch.ops.aten.bmm.default in targets
