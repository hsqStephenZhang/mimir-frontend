"""Opt-in E2E tests which download and execute official pretrained weights."""

import os
import shutil

import pytest
import torch

from mimir_frontend.audit import AuditedMimirBackend

pytestmark = [
    pytest.mark.pretrained,
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH"),
    pytest.mark.skipif(
        os.environ.get("MIMIR_RUN_PRETRAINED") != "1",
        reason="set MIMIR_RUN_PRETRAINED=1 to download official model weights",
    ),
]


def test_resnet50_imagenet_v2_matches_eager(tmp_path):
    torchvision = pytest.importorskip("torchvision")
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights).eval()
    torch.manual_seed(0)
    image = torch.rand(1, 3, 224, 224)

    torch._dynamo.reset()
    audit = AuditedMimirBackend(
        options={"cache_dir": str(tmp_path / "mimir-jit-cache")}
    )
    with torch.no_grad():
        expected = model(image)
        actual = torch.compile(model, backend=audit)(image)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    assert actual.argmax().item() == expected.argmax().item()
    report = audit.report(
        model="torchvision/resnet50",
        metadata={"weights": str(weights), "input_shape": list(image.shape)},
    )
    assert report["graph_breaks"] == {}
    assert len(report["partitions"]) == 1
