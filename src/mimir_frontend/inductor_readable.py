from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
import re
from pathlib import Path

import mim
import torch
from torch import fx

from .translator import FXGraphTranslator
from .utils import ShapeEnv, _shape_from_meta_value, shape_to_mimir_dims, tensor_type_from_shape


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDUCTOR_LOG_ROOT = Path(
    os.environ.get("MIMIR_INDUCTOR_LOG_ROOT", REPO_ROOT / "inductor_logs" / "attn_debug" / "inductor")
)


@dataclass
class PartialTranslation:
    result: mim.Def | None
    frontier_node: fx.Node | None
    error: Exception | None


def make_world() -> mim.World:
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "torch", "affine"])
    return driver.world()


def _element_type_from_dtype(world: mim.World, ops, dtype) -> mim.Def:
    dtype_text = str(dtype)
    if dtype_text.startswith("b") or dtype is torch.bool:
        return world.type_bool()
    if dtype_text.startswith("i") or dtype in (torch.int64, torch.int32, torch.long):
        return world.type_idx(world.top_nat())
    return ops.F32


def load_inductor_graph_module(
    case_or_path: str | Path,
    root: Path = DEFAULT_INDUCTOR_LOG_ROOT,
) -> fx.GraphModule:
    """
    Load an inductor log generated fx_graph_readable.py as a fx.GraphModule.
    Args:
        case_or_path: name of the case in the inductor logs, e.g. "mlp_1", or the path to the fx_graph_readable.py.
    """
    if isinstance(case_or_path, str) and not case_or_path.endswith(".py"):
        path = root / case_or_path / "fx_graph_readable.py"
    else:
        path = Path(case_or_path)

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    content = path.read_text()
    content = content.replace("class <lambda>(torch.nn.Module):", "class GraphModule(torch.nn.Module):")
    namespace = {"torch": torch, "device": torch.device}
    exec(content, namespace)
    return fx.symbolic_trace(namespace["GraphModule"]())


def parse_annotation(annotation: str):
    annotation = str(annotation).strip()
    if (
        len(annotation) >= 2
        and annotation[0] == annotation[-1]
        and annotation[0] in {"'", '"'}
    ):
        annotation = annotation[1:-1]

    if annotation.startswith("Sym("):
        return "sym", annotation[4:-1]

    match = re.fullmatch(r"([a-z0-9]+)\[(.*)\]", annotation)
    if not match:
        raise ValueError(f"unsupported annotation: {annotation}")

    dtype = match.group(1)
    shape_text = match.group(2).strip()
    dims = [] if not shape_text else [part.strip() for part in shape_text.split(",")]
    return "tensor", dtype, dims


def make_mimir_inputs_from_annotations(world: mim.World, graph_module: fx.GraphModule, ops=None) -> list[mim.Def]:
    if ops is None:
        ops = FXGraphTranslator(world).ops
    inputs = []
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    parameters = list(inspect.signature(graph_module.forward).parameters.values())
    shape_env = ShapeEnv(world)

    for parameter, placeholder in zip(parameters, placeholders):
        parsed = parse_annotation(parameter.annotation)
        if parsed[0] == "sym":
            inputs.append(shape_env.symbol_def(parsed[1]))
            continue

        if "val" in placeholder.meta:
            meta_value = placeholder.meta["val"]
            elem_type = _element_type_from_dtype(world, ops, getattr(meta_value, "dtype", None))
            shape = shape_env.normalize_shape(_shape_from_meta_value(meta_value))
            tensor_type = tensor_type_from_shape(world, elem_type, shape, shape_env=shape_env, symbolic=True)
        else:
            _, dtype, dims = parsed
            elem_type = _element_type_from_dtype(world, ops, dtype)
            shape = shape_env.normalize_shape(dims)
            tensor_type = tensor_type_from_shape(world, elem_type, shape, shape_env=shape_env, symbolic=True)
        tensor = world.mut_con(tensor_type).var()
        ops._remember_shape(
            tensor,
            shape_to_mimir_dims(world, shape, shape_env=shape_env, symbolic=True),
        )
        inputs.append(tensor)

    for node in graph_module.graph.nodes:
        if node.op != "get_attr":
            continue
        attr = graph_module
        for part in node.target.split("."):
            attr = getattr(attr, part)
        if not isinstance(attr, torch.Tensor):
            raise NotImplementedError(f"get_attr {node.target} with value type {type(attr)} is not supported")
        if attr.dtype == torch.bool:
            elem_type = world.type_bool()
        else:
            elem_type = ops.F32
        tensor_type = tensor_type_from_shape(world, elem_type, attr.shape)
        tensor = world.mut_con(tensor_type).var()
        ops._remember_shape(tensor, shape_to_mimir_dims(world, attr.shape))
        inputs.append(tensor)

    return inputs


import inspect

def translate_inductor_readable(case_or_path: str | Path, root: Path = DEFAULT_INDUCTOR_LOG_ROOT, as_function: bool = False) -> mim.Def:
    graph_module = load_inductor_graph_module(case_or_path, root=root)
    world = make_world()
    translator = FXGraphTranslator(world, module=graph_module)
    inputs = make_mimir_inputs_from_annotations(world, graph_module, ops=translator.ops)
    
    if not as_function:
        return translator.translate(graph_module.graph, inputs)

    num_params = len(inputs) + 1
    dom_with_ret = world.mut_sigma(num_params)
    for i, inp in enumerate(inputs):
        dom_with_ret.set(i, inp.type())
        
    lam = world.mut_con(dom_with_ret)
    lam.set("main")
    
    actual_inputs = [lam.var().proj(num_params, i) for i in range(len(inputs))]
    
    parameters = list(inspect.signature(graph_module.forward).parameters.keys())
    for i, actual_input in enumerate(actual_inputs):
        if i < len(parameters):
            actual_input.set(parameters[i])

    result = translator.translate(graph_module.graph, actual_inputs)
    
    dom_with_ret.set(num_params - 1, world.cn([result.type()]))
    ret_cont = lam.var().proj(num_params, num_params - 1)
    lam.app(True, ret_cont, [result])
    lam.externalize()
    
    return lam


def translate_inductor_readable_prefix(
    case_or_path: str | Path,
    root: Path = DEFAULT_INDUCTOR_LOG_ROOT,
) -> PartialTranslation:
    graph_module = load_inductor_graph_module(case_or_path, root=root)
    world = make_world()
    translator = FXGraphTranslator(world, module=graph_module)
    inputs = make_mimir_inputs_from_annotations(world, graph_module, ops=translator.ops)
    graph = graph_module.graph

    translator.env = {}
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    param_nodes = [node for node in graph.nodes if node.op == "get_attr"]

    for node, arg in zip(placeholders, inputs[: len(placeholders)]):
        translator.env[node] = arg
    for node, arg in zip(param_nodes, inputs[len(placeholders) :]):
        translator.env[node] = arg

    last_result = None
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            continue
        if node.op in ("call_function", "call_method"):
            try:
                last_result = translator.convert_node(node)
                translator.env[node] = last_result
            except Exception as exc:
                return PartialTranslation(last_result, node, exc)
        elif node.op == "output":
            return PartialTranslation(last_result, None, None)
        else:
            return PartialTranslation(last_result, node, NotImplementedError(f"Op {node.op} not implemented"))

    return PartialTranslation(last_result, None, None)
