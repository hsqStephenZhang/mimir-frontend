"""Explicit PyTorch-to-MimIR decomposition policy.

This module only selects decompositions registered by PyTorch. Operator
semantics remain either in PyTorch's decomposition table or in MimIR's Torch
plugin; the frontend does not duplicate either implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import fx
from torch._decomp import get_decompositions, remove_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


@dataclass(frozen=True)
class DecompositionPolicy:
    """ATen operators to decompose through PyTorch and operators to preserve."""

    fallback: tuple[Any, ...] = ()
    preserve: tuple[Any, ...] = ()
    triggers: tuple[Any, ...] = ()


# Keep this deliberately small. Each entry must have a frontend regression test
# proving that PyTorch expands it only into operators accepted by MimIR.
MIMIR_DECOMPOSITION_POLICY = DecompositionPolicy(
    fallback=(
        torch.ops.aten.hardswish.default,
        torch.ops.aten.hardsigmoid.default,
    ),
    preserve=(
        torch.ops.aten.native_layer_norm.default,
    ),
    triggers=(
        torch.nn.functional.hardswish,
        torch.nn.functional.multi_head_attention_forward,
        torch.ops.aten.hardswish.default,
    ),
)

NO_DECOMPOSITIONS = DecompositionPolicy()


def resolve_decomposition_policy(value: str | DecompositionPolicy | None) -> DecompositionPolicy:
    """Resolve the public backend option without silently enabling a global table."""
    if value is None or value == "mimir":
        return MIMIR_DECOMPOSITION_POLICY
    if value == "none":
        return NO_DECOMPOSITIONS
    if isinstance(value, DecompositionPolicy):
        return value
    raise ValueError(
        "decomposition_policy must be 'mimir', 'none', or DecompositionPolicy"
    )


def decomposition_table(policy: DecompositionPolicy) -> dict[Any, Any]:
    """Build a PyTorch-registered table, with preservation taking precedence."""
    table = get_decompositions(policy.fallback)
    remove_decompositions(table, policy.preserve)
    return table


def apply_decomposition_policy(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    policy: DecompositionPolicy,
) -> fx.GraphModule:
    """Retrace ``gm`` through exactly the selected PyTorch decompositions."""
    table = decomposition_table(policy)
    triggers = policy.triggers or policy.fallback
    if not table or not any(
        node.op == "call_function" and node.target in triggers
        for node in gm.graph.nodes
    ):
        return gm
    return make_fx(gm, decomposition_table=table)(*example_inputs)
