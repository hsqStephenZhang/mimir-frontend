"""Structured operator, partition, and timing records for model E2E runs."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import fx

from .backend import mimir_backend


def _tensor_spec(tensor: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


@dataclass
class PartitionAudit:
    index: int
    node_count: int
    operators: dict[str, int]
    inputs: list[dict[str, Any]]
    compile_seconds: float
    execution_seconds: list[float] = field(default_factory=list)


class AuditedMimirBackend:
    """Wrap ``mimir_backend`` and retain one deterministic record per partition."""

    def __init__(self, *, options: dict[str, Any] | None = None):
        self.options = dict(options or {})
        self.partitions: list[PartitionAudit] = []

    def __call__(self, gm: fx.GraphModule, example_inputs, **kwargs):
        options = dict(self.options)
        options.update(kwargs.pop("options", None) or {})
        if kwargs:
            raise TypeError(f"unknown audited backend arguments: {sorted(kwargs)}")

        operators = Counter(
            str(node.target) for node in gm.graph.nodes if node.op == "call_function"
        )
        start = time.perf_counter()
        compiled = mimir_backend(gm, example_inputs, options=options)
        compile_seconds = time.perf_counter() - start
        record = PartitionAudit(
            index=len(self.partitions),
            node_count=len(list(gm.graph.nodes)),
            operators=dict(sorted(operators.items())),
            inputs=[_tensor_spec(tensor) for tensor in example_inputs],
            compile_seconds=compile_seconds,
        )
        self.partitions.append(record)

        def measured(*args):
            start = time.perf_counter()
            result = compiled(*args)
            record.execution_seconds.append(time.perf_counter() - start)
            return result

        return measured

    def report(self, *, model: str, metadata: dict[str, Any] | None = None):
        graph_breaks = {
            str(reason): count
            for reason, count in torch._dynamo.utils.counters["graph_break"].items()
        }
        return {
            "model": model,
            "metadata": dict(metadata or {}),
            "graph_breaks": dict(sorted(graph_breaks.items())),
            "partitions": [asdict(partition) for partition in self.partitions],
        }

    def write_json(
        self,
        path: str | Path,
        *,
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        Path(path).write_text(
            json.dumps(self.report(model=model, metadata=metadata), indent=2, sort_keys=True)
            + "\n"
        )
