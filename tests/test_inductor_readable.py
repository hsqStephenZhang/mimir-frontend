import mim
import torch
from torch import fx
from torch._subclasses.fake_tensor import FakeTensorMode

from mimir_frontend.inductor_readable import make_mimir_inputs_from_annotations, make_world


def make_graph_module(source: str) -> fx.GraphModule:
    namespace = {"torch": torch}
    exec(source, namespace)
    return fx.symbolic_trace(namespace["GraphModule"]())




def tensor_shape_values(tensor_def):
    return [d.get_nat() if isinstance(d, mim.Lit) else None for d in tensor_shape_defs(tensor_def)]


def tensor_shape_defs(tensor_def):
    dims = []
    tensor_type = tensor_def.type()
    while isinstance(tensor_type, mim.Seq):
        dims.append(tensor_type.arity())
        tensor_type = tensor_type.body()
    return dims


def test_make_mimir_inputs_from_annotations_prefers_fake_tensor_meta_shape():
    graph_module = make_graph_module(
        """
class GraphModule(torch.nn.Module):
    def forward(self, x: 'f32[5]'):
        return x
"""
    )
    placeholder = next(node for node in graph_module.graph.nodes if node.op == "placeholder")

    with FakeTensorMode() as mode:
        placeholder.meta["val"] = mode.from_tensor(torch.empty(2, 3))

    world = make_world()
    inputs = make_mimir_inputs_from_annotations(world, graph_module)

    assert tensor_shape_values(inputs[0]) == [2, 3]


def test_make_mimir_inputs_from_annotations_keeps_sym_parameters_before_tensors():
    graph_module = make_graph_module(
        """
class GraphModule(torch.nn.Module):
    def forward(self, n: 'Sym(s0)', x: 'f32[5]'):
        return x
"""
    )
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]

    with FakeTensorMode() as mode:
        placeholders[1].meta["val"] = mode.from_tensor(torch.empty(2, 3))

    world = make_world()
    inputs = make_mimir_inputs_from_annotations(world, graph_module)

    assert inputs[0].type() == world.type_nat()
    assert tensor_shape_values(inputs[1]) == [2, 3]


def test_make_mimir_inputs_from_annotations_reuses_symbolic_dims_from_annotations():
    graph_module = make_graph_module(
        """
class GraphModule(torch.nn.Module):
    def forward(self, x: 'f32[s0]', y: 'f32[s0]'):
        return x
"""
    )

    world = make_world()
    x, y = make_mimir_inputs_from_annotations(world, graph_module)

    assert tensor_shape_defs(x)[0] == tensor_shape_defs(y)[0]
    assert tensor_shape_defs(x)[0] != world.top_nat()


def test_make_mimir_inputs_from_annotations_shares_sym_parameter_with_tensor_dim():
    graph_module = make_graph_module(
        """
class GraphModule(torch.nn.Module):
    def forward(self, n: 'Sym(s0)', x: 'f32[s0]'):
        return x
"""
    )

    world = make_world()
    n, x = make_mimir_inputs_from_annotations(world, graph_module)

    assert tensor_shape_defs(x)[0] == n
