"""Symbolic extents: Dynamo's `SymInt` placeholders and the symbols named in tensor shapes."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import fx

Shape = Sequence[int | str]


def placeholder_value(node: fx.Node):
    """The fake value Dynamo (`example_value`) or AOT (`val`) recorded for a placeholder."""
    return node.meta.get("example_value", node.meta.get("val"))


def is_sym_placeholder(node: fx.Node) -> bool:
    return node.op == "placeholder" and isinstance(placeholder_value(node), torch.SymInt)


def symbol_dim(dim) -> int | str:
    """A static extent stays an int; a plain symbol `s0` becomes its name; anything else is unsupported."""
    if isinstance(dim, torch.SymInt):
        if dim.node.expr.is_Integer:
            return int(dim.node.expr)
        if not dim.node.expr.is_Symbol:
            raise NotImplementedError(f"mimir backend supports plain symbolic extents only, got `{dim}`")
        return str(dim)
    return int(dim)


def symbol_names(sym_inputs: Sequence[str], shapes: Sequence[Shape]) -> list[str]:
    """MimIR's leading `Nat` parameters: the `SymInt` placeholders, then symbols in order of first use."""
    names = list(sym_inputs)
    for shape in shapes:
        for dim in shape:
            if isinstance(dim, str) and not dim.isdecimal() and dim not in names:
                names.append(dim)
    return names
