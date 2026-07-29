"""End-to-end coverage for standard torch.nn recurrent modules."""

import shutil

import pytest
import torch

from mimir_frontend.backend import mimir_backend


pytestmark = pytest.mark.skipif(
    shutil.which("clang") is None, reason="clang not on PATH"
)


@pytest.fixture(autouse=True)
def _reset_dynamo(monkeypatch, tmp_path_factory):
    monkeypatch.setenv(
        "MIMIR_CACHE_DIR",
        str(tmp_path_factory.getbasetemp() / "mimir-recurrent-jit-cache"),
    )
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _make_case(
    kind,
    *,
    num_layers,
    bidirectional,
    bias,
    batch_first,
    batch,
    nonlinearity="tanh",
    dropout=0.0,
    explicit_state=True,
):
    input_size = 2
    hidden_size = 3
    sequence_length = 3
    common = dict(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=bias,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    if kind == "rnn":
        model = torch.nn.RNN(nonlinearity=nonlinearity, **common)
    elif kind == "gru":
        model = torch.nn.GRU(**common)
    elif kind == "lstm":
        model = torch.nn.LSTM(**common)
    else:
        raise AssertionError(f"unknown recurrent kind: {kind}")

    input_shape = (
        (batch, sequence_length, input_size)
        if batch_first
        else (sequence_length, batch, input_size)
    )
    x = torch.randn(input_shape)
    directions = 2 if bidirectional else 1
    state_shape = (num_layers * directions, batch, hidden_size)
    hidden = torch.randn(state_shape)
    state = (
        (hidden, torch.randn(state_shape))
        if kind == "lstm"
        else hidden
    )
    args = (x, state) if explicit_state else (x,)
    return model.eval(), args


@pytest.mark.parametrize(
    (
        "kind",
        "num_layers",
        "bidirectional",
        "bias",
        "batch_first",
        "batch",
        "nonlinearity",
        "dropout",
        "explicit_state",
    ),
    [
        pytest.param(
            "rnn", 1, False, True, True, 2, "tanh", 0.0, True,
            id="rnn-tanh-single"
        ),
        pytest.param(
            "rnn", 2, True, False, False, 2, "relu", 0.25, True,
            id="rnn-relu-stacked-bidir-eval-dropout"
        ),
        pytest.param(
            "gru", 1, False, True, True, 2, "tanh", 0.0, False,
            id="gru-single-implicit-state"
        ),
        pytest.param(
            "gru", 2, True, True, False, 1, "tanh", 0.0, True,
            id="gru-stacked-bidir-batch1"
        ),
        pytest.param(
            "lstm", 1, False, True, True, 2, "tanh", 0.0, False,
            id="lstm-single-implicit-state"
        ),
        pytest.param(
            "lstm", 2, True, False, False, 1, "tanh", 0.0, True,
            id="lstm-stacked-bidir-batch1"
        ),
    ],
)
def test_standard_recurrent_module_matches_eager(
    kind,
    num_layers,
    bidirectional,
    bias,
    batch_first,
    batch,
    nonlinearity,
    dropout,
    explicit_state,
):
    torch.manual_seed(0)
    model, args = _make_case(
        kind,
        num_layers=num_layers,
        bidirectional=bidirectional,
        bias=bias,
        batch_first=batch_first,
        batch=batch,
        nonlinearity=nonlinearity,
        dropout=dropout,
        explicit_state=explicit_state,
    )
    with torch.no_grad():
        expected = model(*args)
        actual = torch.compile(model, backend=mimir_backend)(*args)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
