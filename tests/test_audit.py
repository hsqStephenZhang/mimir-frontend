import json

import torch

from mimir_frontend.audit import AuditedMimirBackend


def test_audited_backend_records_partition_inventory_and_timings(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return torch.relu(x + y)

    torch._dynamo.reset()
    audit = AuditedMimirBackend(options={"cache_dir": str(tmp_path / "cache")})
    model = Model().eval()
    x, y = torch.randn(2, 3), torch.randn(2, 3)
    with torch.no_grad():
        result = torch.compile(model, backend=audit)(x, y)
    torch.testing.assert_close(result, model(x, y))

    path = tmp_path / "audit.json"
    audit.write_json(path, model="test/model", metadata={"weights": "none"})
    report = json.loads(path.read_text())
    assert report["model"] == "test/model"
    assert report["metadata"] == {"weights": "none"}
    assert report["graph_breaks"] == {}
    assert len(report["partitions"]) == 1
    partition = report["partitions"][0]
    assert partition["node_count"] >= 4
    assert sum(partition["operators"].values()) == 2
    assert partition["compile_seconds"] > 0
    assert len(partition["execution_seconds"]) == 1
    assert partition["execution_seconds"][0] > 0
