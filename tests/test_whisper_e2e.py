"""End-to-end coverage for the standard Whisper Tiny architecture."""

import shutil

import pytest
import torch

from mimir_frontend.backend import mimir_backend

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("clang") is None, reason="clang not on PATH"),
]


def _tiny_config(transformers):
    config = transformers.WhisperConfig(
        vocab_size=64,
        num_mel_bins=80,
        d_model=384,
        encoder_layers=4,
        decoder_layers=4,
        encoder_attention_heads=6,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        max_source_positions=64,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def test_whisper_tiny_short_sequence_matches_eager(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path / "mimir-jit-cache"))
    torch._dynamo.reset()

    model = transformers.WhisperForConditionalGeneration(
        _tiny_config(transformers)
    ).float().eval()

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_features, decoder_input_ids):
            return self.inner(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            ).logits

    wrapped = Logits(model).eval()
    input_features = torch.randn(1, 80, 128)
    decoder_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with torch.no_grad():
        expected = wrapped(input_features, decoder_input_ids)
        actual = torch.compile(
            wrapped,
            backend=mimir_backend,
            options={"max_fp_iters": 4096},
        )(input_features, decoder_input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_whisper_tiny_decoder_matches_eager(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path / "mimir-jit-cache"))
    torch._dynamo.reset()
    decoder = transformers.WhisperForConditionalGeneration(
        _tiny_config(transformers)
    ).model.decoder.float().eval()

    class HiddenState(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, encoder_hidden_states):
            return self.inner(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
            ).last_hidden_state

    wrapped = HiddenState(decoder).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    encoder_hidden_states = torch.randn(1, 64, 384)
    with torch.no_grad():
        expected = wrapped(input_ids, encoder_hidden_states)
        actual = torch.compile(
            wrapped,
            backend=mimir_backend,
            options={"max_fp_iters": 4096},
        )(input_ids, encoder_hidden_states)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_whisper_decoder_embeddings_match_eager(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path / "mimir-jit-cache"))
    torch._dynamo.reset()
    decoder = transformers.WhisperForConditionalGeneration(
        _tiny_config(transformers)
    ).model.decoder.float().eval()

    class Embeddings(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).repeat(
                input_ids.shape[0], 1
            )
            return self.inner.embed_tokens(input_ids) + self.inner.embed_positions(
                input_ids, position_ids=position_ids
            )

    wrapped = Embeddings(decoder).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    with torch.no_grad():
        expected = wrapped(input_ids)
        actual = torch.compile(wrapped, backend=mimir_backend)(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_whisper_causal_mask_matches_eager(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    masking_utils = pytest.importorskip("transformers.masking_utils")
    monkeypatch.setenv("MIMIR_CACHE_DIR", str(tmp_path / "mimir-jit-cache"))
    torch._dynamo.reset()

    class CausalMask(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

        def forward(self, inputs_embeds):
            position_ids = torch.arange(inputs_embeds.shape[1]).unsqueeze(0).repeat(
                inputs_embeds.shape[0], 1
            )
            mask = masking_utils.create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids,
            )
            # Keep the input as an FX placeholder while preserving the exact
            # mask values used by Whisper.
            return mask + inputs_embeds[:, :1, :4] * 0.0

    wrapped = CausalMask(_tiny_config(transformers)).eval()
    inputs_embeds = torch.randn(1, 4, 384)
    with torch.no_grad():
        expected = wrapped(inputs_embeds)
        actual = torch.compile(wrapped, backend=mimir_backend)(inputs_embeds)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
