import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode

from mimir_frontend.utils import model_to_mimir


class AddReluModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


def test_model_to_mimir_outputs_high_level_tensor_ir():
    ir = model_to_mimir(
        AddReluModel(),
        input_shapes=[("n",), ("n",)],
        compile_phase="high_level",
    )

    assert "%torch.binary.add" in ir
    assert "%torch.activation.relu" in ir


def test_model_to_mimir_signature_uses_symbolic_nat_in_input_tensor_types():
    ir = model_to_mimir(
        AddReluModel(),
        input_shapes=[("n",), ("n",)],
        compile_phase="high_level",
    )

    first_line = ir.splitlines()[0]
    assert "⊤:Nat" not in first_line
    assert ": Nat" in first_line


def test_model_to_mimir_can_build_placeholder_types_from_fake_tensor_meta():
    traced = fx.symbolic_trace(AddReluModel())
    with FakeTensorMode() as mode:
        fake_x = mode.from_tensor(torch.empty(2, 3))
        fake_y = mode.from_tensor(torch.empty(2, 3))

    placeholders = [node for node in traced.graph.nodes if node.op == "placeholder"]
    placeholders[0].meta["val"] = fake_x
    placeholders[1].meta["val"] = fake_y

    ir = model_to_mimir(
        traced,
        input_shapes=None,
        compile_phase="high_level",
    )

    first_line = ir.splitlines()[0]
    assert "«2; «3; %math.F (23, 8)»»" in first_line


def test_model_to_mimir_requires_input_shapes_when_placeholder_meta_is_missing():
    traced = fx.symbolic_trace(AddReluModel())

    try:
        model_to_mimir(
            traced,
            input_shapes=None,
            compile_phase="high_level",
        )
    except ValueError as exc:
        assert "meta['val']" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_model_to_mimir_can_use_default_compile_phase():
    ir = model_to_mimir(
        AddReluModel(),
        input_shapes=[("n",), ("n",)],
        compile_phase="default",
    )

    assert isinstance(ir, str)
    assert len(ir) > 0
    assert "extern _compile" in ir
    assert "fun extern mimir_module" in ir or "lam extern mimir_module" in ir


def test_model_to_mimir_rejects_unknown_compile_phase():
    try:
        model_to_mimir(AddReluModel(), input_shapes=[(None,), (None,)], compile_phase="unknown")
    except ValueError as exc:
        assert "compile_phase" in str(exc)
    else:
        raise AssertionError("expected ValueError")
