"""Optional end-to-end coverage for an official Qwen3 transformer graph."""

import shutil

import pytest
import torch

from mimir_frontend.backend import mimir_backend

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH")


def test_tiny_qwen3_causal_lm_matches_eager(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path / "mimir-jit-cache"))
    torch._dynamo.reset()

    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=16,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config).float().eval()
    mask = torch.triu(
        torch.full((5, 5), torch.finfo(torch.float32).min), diagonal=1
    )[None, None]

    class WithStaticMask(torch.nn.Module):
        def __init__(self, inner, attention_mask):
            super().__init__()
            self.inner = inner
            self.register_buffer("attention_mask", attention_mask)

        def forward(self, input_ids):
            return self.inner(
                input_ids=input_ids,
                attention_mask=self.attention_mask,
                use_cache=False,
            ).logits

    wrapped = WithStaticMask(model, mask).eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    with torch.no_grad():
        expected = wrapped(input_ids)
        actual = torch.compile(wrapped, backend=mimir_backend)(input_ids)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
