import torch
from torch import fx
import mim
from .operators import OperatorLibrary
from mim._plugins.compile import compile as mim_compile
import operator
from collections.abc import Callable

import builtins

class FXGraphTranslator:
    def __init__(self, world: mim.World, module: torch.nn.Module = None):
        self.world = world
        self.module = module
        self.ops = OperatorLibrary(world)
        self.env: dict[fx.Node, mim.Def] = {}
        self.convert_map: dict[str | Callable, Callable[[fx.Node], mim.Def]] = self.create_convert_map()

    def create_convert_map(self) -> dict:
        m = {}
        
        # Elementwise Binary
        binary_ops = [
            (torch.add, operator.add, ["aten.add.default", "aten.add.Tensor", "aten.add.Scalar"], self.ops.add),
            (torch.sub, operator.sub, ["aten.sub.default", "aten.sub.Tensor", "aten.sub.Scalar"], self.ops.sub),
            (torch.mul, operator.mul, ["aten.mul.default", "aten.mul.Tensor", "aten.mul.Scalar"], self.ops.mul),
            (torch.div, operator.truediv, ["aten.div.default", "aten.div.Tensor", "aten.div.Scalar"], self.ops.div),
            (torch.pow, operator.pow, ["aten.pow.default", "aten.pow.Tensor_Tensor", "aten.pow.Tensor_Scalar"], self.ops.pow),
            (torch.maximum, None, ["aten.maximum.default"], self.ops.maximum),
            (torch.minimum, None, ["aten.minimum.default"], self.ops.minimum),
            (torch.eq, operator.eq, ["aten.eq.default", "aten.eq.Tensor", "aten.eq.Scalar"], self.ops.eq),
            (torch.ne, operator.ne, ["aten.ne.default", "aten.ne.Tensor", "aten.ne.Scalar"], self.ops.ne),
            (torch.lt, operator.lt, ["aten.lt.default", "aten.lt.Tensor", "aten.lt.Scalar"], self.ops.lt),
            (torch.le, operator.le, ["aten.le.default", "aten.le.Tensor", "aten.le.Scalar"], self.ops.le),
            (torch.gt, operator.gt, ["aten.gt.default", "aten.gt.Tensor", "aten.gt.Scalar"], self.ops.gt),
            (torch.ge, operator.ge, ["aten.ge.default", "aten.ge.Tensor", "aten.ge.Scalar"], self.ops.ge),
            (torch.clamp_max, None, ["aten.clamp_max.default"], self.ops.clamp_max),
            (torch.clamp_min, None, ["aten.clamp_min.default"], self.ops.clamp_min),
            (torch.bitwise_and, operator.and_, ["aten.bitwise_and.default", "aten.bitwise_and.Tensor"], self.ops.bitwise_and),
        ]
        
        for t, op, names, func in binary_ops:
            wrapper = self._wrap_binary(func)
            if t: m[t] = wrapper
            if op: m[op] = wrapper
            for name in names:
                m[name] = wrapper

        m[torch.clamp] = self._wrap_clamp()
        m["aten.clamp.default"] = self._wrap_clamp()
        m["aten.clamp.Tensor"] = self._wrap_clamp()
        m[operator.floordiv] = self._wrap_nat_binary(self.ops.nat_floordiv)

        # Elementwise Unary
        unary_ops = [
            (torch.relu, "aten.relu.default", self.ops.relu),
            (torch.nn.functional.relu, "relu", self.ops.relu),
            (torch.exp, "aten.exp.default", self.ops.exp),
            (torch.tanh, "aten.tanh.default", self.ops.tanh),
            (torch.sqrt, "aten.sqrt.default", self.ops.sqrt),
            (torch.abs, "aten.abs.default", self.ops.abs),
            (torch.neg, "aten.neg.default", self.ops.neg),
            (torch.sigmoid, "aten.sigmoid.default", self.ops.sigmoid),
            (torch.reciprocal, "aten.reciprocal.default", self.ops.reciprocal),
            (torch.rsqrt, "aten.rsqrt.default", self.ops.rsqrt),
            (torch.logical_not, "aten.logical_not.default", self.ops.logical_not),
        ]
        
        for t, name, func in unary_ops:
            wrapper = self._wrap_unary(func)
            if t: m[t] = wrapper
            if name: m[name] = wrapper

        # Prims
        if hasattr(torch.ops, "prims") and hasattr(torch.ops.prims, "convert_element_type"):
            m[torch.ops.prims.convert_element_type.default] = self._wrap_convert_element_type()
        m["prims.convert_element_type.default"] = self._wrap_convert_element_type()
        m["aten.convert_element_type.default"] = self._wrap_convert_element_type()
        m["prims.fma.default"] = self._wrap_fma()

        # Injective
        m[torch.cat] = self._wrap_cat()
        m["aten.cat.default"] = self._wrap_cat()
        m[torch.permute] = self._wrap_transpose()
        m["aten.permute.default"] = self._wrap_transpose()
        m["t"] = self._wrap_t()
        
        m[torch.reshape] = self._wrap_reshape()
        m["aten.reshape.default"] = self._wrap_reshape()
        m["reshape"] = self._wrap_reshape()
        m["view"] = self._wrap_reshape()
        m["aten.view.default"] = self._wrap_reshape()
        m[torch.flatten] = self._wrap_flatten()
        m["aten.flatten.using_ints"] = self._wrap_flatten()
        m["flatten"] = self._wrap_flatten()

        m["aten.slice.Tensor"] = self._wrap_slice()
        m["aten.select.int"] = self._wrap_select()
        m["aten.split.Tensor"] = self._wrap_split()
        m["split"] = self._wrap_split()
        
        m[torch.squeeze] = self._wrap_squeeze()
        m["squeeze"] = self._wrap_squeeze()
        m["aten.squeeze.dim"] = self._wrap_squeeze()
        m["aten.squeeze.dims"] = self._wrap_squeeze()
        
        m[torch.unsqueeze] = self._wrap_unsqueeze()
        m["unsqueeze"] = self._wrap_unsqueeze()
        m["aten.unsqueeze.default"] = self._wrap_unsqueeze()
        m["contiguous"] = self._wrap_contiguous()
        m["size"] = self._wrap_size()

        m[torch.clone] = self._wrap_unary(self.ops.clone)
        m["clone"] = self._wrap_unary(self.ops.clone)
        m["aten.clone.default"] = self._wrap_unary(self.ops.clone)
        m["aten.copy.default"] = self._wrap_binary(self.ops.copy)
        m["aten.lift_fresh_copy.default"] = self._wrap_unary(self.ops.clone)
        m[torch.nn.functional.dropout] = self._wrap_dropout()
        m["aten.dropout.default"] = self._wrap_dropout()
        m["dropout"] = self._wrap_dropout()
        
        # Broadcast
        m[torch.expand_copy] = self._wrap_expand()
        m["aten.expand.default"] = self._wrap_expand()
        m["expand"] = self._wrap_expand()
        m[torch.full] = self._wrap_full()
        m["aten.full.default"] = self._wrap_full()

        # Reductions
        m[torch.sum] = self._wrap_reduction(self.ops.sum)
        m["sum"] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.default"] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.dim_IntList"] = self._wrap_reduction(self.ops.sum)
        m[torch.amax] = self._wrap_reduction(self.ops.amax)
        m["aten.amax.default"] = self._wrap_reduction(self.ops.amax)
        m[torch.max] = self._wrap_max()
        m["aten.max.default"] = self._wrap_max()
        m["aten.max.dim"] = self._wrap_max()
        m[torch.mean] = self._wrap_reduction(self.ops.mean)
        m["mean"] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.default"] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.dim"] = self._wrap_reduction(self.ops.mean)
        m[torch.var_mean] = self._wrap_var_mean()
        m["aten.var_mean.default"] = self._wrap_var_mean()
        m["aten.var_mean.correction"] = self._wrap_var_mean()

        # Linear Algebra
        m[torch.mm] = self._wrap_binary(self.ops.mm)
        m["aten.mm.default"] = self._wrap_binary(self.ops.mm)
        m["aten.addmm.default"] = self._wrap_addmm()
        if hasattr(torch, "_C") and hasattr(torch._C, "_nn") and hasattr(torch._C._nn, "linear"):
            m[torch._C._nn.linear] = self._wrap_linear()
        m["aten.linear.default"] = self._wrap_linear()
        m["torch._C._nn.linear"] = self._wrap_linear()

        # Convolution
        m[torch.convolution] = self._wrap_convolution()
        if hasattr(torch, "conv2d"):
            m[torch.conv2d] = self._wrap_conv2d()
        m[torch.nn.functional.conv2d] = self._wrap_conv2d()
        m["aten.convolution.default"] = self._wrap_convolution()
        m["aten.conv2d.default"] = self._wrap_conv2d()
        m["conv2d"] = self._wrap_conv2d()

        # Pooling
        m[torch.nn.functional.max_pool2d] = self._wrap_max_pool2d()
        m["aten.max_pool2d.default"] = self._wrap_max_pool2d()
        m["aten.max_pool2d_with_indices.default"] = self._wrap_max_pool2d(return_indices=True)
        m[torch.nn.functional.avg_pool2d] = self._wrap_avg_pool2d()
        m["aten.avg_pool2d.default"] = self._wrap_avg_pool2d()
        m[torch.nn.functional.adaptive_avg_pool2d] = self._wrap_adaptive_avg_pool2d()
        m["aten.adaptive_avg_pool2d.default"] = self._wrap_adaptive_avg_pool2d()
        m["adaptive_avg_pool2d"] = self._wrap_adaptive_avg_pool2d()

        # Indexing / Scatter
        m["aten.index.Tensor"] = self._wrap_index_tensor()
        m["aten.scatter.src"] = self._wrap_scatter()
        m["aten.scatter.default"] = self._wrap_scatter()
        m["aten.scatter_add.default"] = self._wrap_unsupported("aten.scatter_add")

        # Selection
        m[torch.where] = self._wrap_where()
        m["aten.where.self"] = self._wrap_where()

        # Tuple operations
        m[operator.getitem] = self._wrap_getitem()
        m[builtins.getattr] = self._wrap_getattr()

        return m

    def _wrap_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0], args[1])
        return convert

    def _wrap_nat_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0], args[1])
        return convert

    def _wrap_addmm(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            bias, input, mat2 = args[:3]
            out_features = self.ops.shape_of(mat2)[1]
            batch = self.ops.shape_of(input)[0]
            bias_row = self.ops.reshape(bias, [1, out_features])
            bias_expanded = self.ops.repeat(bias_row, [batch, out_features])
            return self.ops.add(self.ops.mm(input, mat2), bias_expanded)
        return convert

    def _wrap_linear(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else node.kwargs.get("bias", None)
            return self.ops.linear(input, weight, bias=bias)
        return convert

    def _wrap_convolution(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            stride = args[3] if len(args) > 3 else node.kwargs.get("stride", 1)
            padding = args[4] if len(args) > 4 else node.kwargs.get("padding", 0)
            dilation = args[5] if len(args) > 5 else node.kwargs.get("dilation", 1)
            transposed = args[6] if len(args) > 6 else node.kwargs.get("transposed", False)
            output_padding = args[7] if len(args) > 7 else node.kwargs.get("output_padding", 0)
            groups = args[8] if len(args) > 8 else node.kwargs.get("groups", 1)
            if transposed:
                raise NotImplementedError("aten.convolution with transposed=True is not implemented")
            if output_padding not in (0, [0, 0], (0, 0)):
                raise NotImplementedError("aten.convolution with output_padding is not implemented")
            return self.ops.convolution(
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        return convert

    def _wrap_conv2d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else node.kwargs.get("bias", None)
            stride = args[3] if len(args) > 3 else node.kwargs.get("stride", 1)
            padding = args[4] if len(args) > 4 else node.kwargs.get("padding", 0)
            dilation = args[5] if len(args) > 5 else node.kwargs.get("dilation", 1)
            groups = args[6] if len(args) > 6 else node.kwargs.get("groups", 1)
            return self.ops.convolution(
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        return convert

    def _wrap_max_pool2d(self, return_indices: bool = False):
        def convert(node: fx.Node):
            if return_indices:
                raise NotImplementedError("max_pool2d_with_indices tuple result is not implemented")
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            dilation = args[4] if len(args) > 4 else node.kwargs.get("dilation", 1)
            ceil_mode = args[5] if len(args) > 5 else node.kwargs.get("ceil_mode", False)
            result = self.ops.max_pool2d(
                x,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                return_indices=return_indices,
            )
            return result
        return convert

    def _wrap_avg_pool2d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            ceil_mode = args[4] if len(args) > 4 else node.kwargs.get("ceil_mode", False)
            count_include_pad = args[5] if len(args) > 5 else node.kwargs.get("count_include_pad", True)
            divisor_override = args[6] if len(args) > 6 else node.kwargs.get("divisor_override", None)
            return self.ops.avg_pool2d(
                x,
                kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
        return convert

    def _wrap_adaptive_avg_pool2d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            output_size = args[1] if len(args) > 1 else node.kwargs.get("output_size")
            if output_size == 1 or output_size == [1, 1] or output_size == (1, 1):
                return self.ops.mean(x, dim=[2, 3], keepdim=True)
            raise NotImplementedError("adaptive_avg_pool2d currently supports output_size=1 only")
        return convert

    def _wrap_index_tensor(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            indices = args[1]
            if len(indices) != 1 or indices[0] is None:
                raise NotImplementedError("aten.index.Tensor currently supports a single tensor index")
            return self.ops.index_tensor(x, indices[0])
        return convert

    def _wrap_scatter(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            dim = args[1]
            index = args[2]
            src = args[3]
            return self.ops.scatter(x, dim, index, src)
        return convert

    def _wrap_unary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0])
        return convert

    def _wrap_reduction(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", None)
            keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
            return op_func(input_def, dim=dim, keepdim=keepdim)
        return convert

    def _wrap_max(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            if len(args) > 1 or "dim" in node.kwargs:
                raise NotImplementedError("torch.max with dim is not implemented (requires tuple return)")
            return self.ops.amax(args[0], dim=None, keepdim=False)
        return convert

    def _wrap_var_mean(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", None)
            keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
            correction = args[3] if len(args) > 3 else node.kwargs.get("correction", 1)
            return self.ops.var_mean(input_def, dim=dim, keepdim=keepdim, correction=correction)
        return convert

    def _wrap_where(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.where(args[0], args[1], args[2])
        return convert

    def _wrap_t(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.transpose(args[0], [1, 0])
        return convert

    def _wrap_transpose(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            permutation = args[1]
            return self.ops.transpose(x, permutation)
        return convert

    def _wrap_getitem(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            obj = args[0]
            index = args[1]
            
            # Check if obj is a tensor by inspecting its type
            ty = obj.type()
            
            if isinstance(ty, (mim.Arr, mim.Seq)):
                # Handle tensor indexing/slicing
                if isinstance(index, int):
                    return self.ops.select(obj, 0, index)
                elif isinstance(index, slice):
                    return self.ops.slice(obj, 0, index.start or 0, index.stop, index.step or 1)
                elif isinstance(index, (tuple, list)):
                    res = obj
                    for i, idx in enumerate(index):
                        if isinstance(idx, slice):
                            res = self.ops.slice(res, i, idx.start or 0, idx.stop, idx.step or 1)
                        elif isinstance(idx, int):
                            res = self.ops.select(res, i, idx)
                    return res
            
            # Fallback to tuple projection
            if isinstance(index, int):
                return obj.proj(obj.num_projs(), index)
            raise TypeError(f"Cannot getitem from {obj} (mim_type {type(ty)}) with index {index} (type {type(index)})")
        return convert

    def _wrap_getattr(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            obj = args[0]
            attr_name = args[1] if len(args) > 1 else node.kwargs.get("name")
            if attr_name == "shape":
                return self.ops.shape_of(obj)
            else:
                raise NotImplementedError(f"getattr for {attr_name} is not implemented")
        return convert

    def _wrap_clamp(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            min_val = args[1] if len(args) > 1 else node.kwargs.get("min")
            max_val = args[2] if len(args) > 2 else node.kwargs.get("max")
            return self.ops.clamp(x, min_val=min_val, max_val=max_val)
        return convert

    def _wrap_convert_element_type(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            dtype = args[1] if len(args) > 1 else node.kwargs.get("dtype")
            return self.ops.convert_element_type(x, dtype)
        return convert

    def _wrap_fma(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.fma(args[0], args[1], args[2])
        return convert

    def _wrap_expand(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            shape = args[1:] if len(args) > 2 else args[1]
            return self.ops.expand(x, shape)
        return convert

    def _wrap_reshape(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            shape = args[1:] if len(args) > 2 else args[1]
            return self.ops.reshape(x, shape)
        return convert

    def _wrap_flatten(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            start_dim = args[1] if len(args) > 1 else node.kwargs.get("start_dim", 0)
            end_dim = args[2] if len(args) > 2 else node.kwargs.get("end_dim", -1)
            return self.ops.flatten(x, start_dim=start_dim, end_dim=end_dim)
        return convert

    def _wrap_dropout(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            p = args[1] if len(args) > 1 else node.kwargs.get("p", 0.5)
            training = args[2] if len(args) > 2 else node.kwargs.get("training", True)
            if p == 0 or training is False:
                return x
            raise NotImplementedError("dropout is only supported for p=0 or training=False")
        return convert

    def _wrap_slice(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            # aten.slice.Tensor(input, dim=0, start=0, end=9223372036854775807, step=1)
            x = args[0]
            dim = args[1] if len(args) > 1 else 0
            start = args[2] if len(args) > 2 else 0
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            return self.ops.slice(x, dim, start, end, step)
        return convert

    def _wrap_select(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            # aten.select.int(input, dim, index)
            return self.ops.select(args[0], args[1], args[2])
        return convert

    def _wrap_split(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            split_size_or_sections = args[1]
            dim = args[2] if len(args) > 2 else node.kwargs.get("dim", 0)
            return self.ops.split(x, split_size_or_sections, dim=dim)
        return convert

    def _wrap_squeeze(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            dim = args[1] if len(args) > 1 else None
            return self.ops.squeeze(x, dim)
        return convert

    def _wrap_unsqueeze(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.unsqueeze(args[0], args[1])
        return convert

    def _wrap_contiguous(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return args[0]
        return convert

    def _wrap_size(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            dims = self.ops.shape_of(x)
            if len(args) > 1:
                dim = args[1]
                if isinstance(dim, int):
                    return dims[dim]
                return dims[dim.get_nat()] if hasattr(dim, "get_nat") else dims[dim]
            return dims
        return convert

    def _wrap_cat(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            tensors = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", 0)
            return self.ops.cat(self.world.tuple(tensors), dim=dim)
        return convert

    def _wrap_full(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            shape = args[0]
            fill_value = args[1] if len(args) > 1 else node.kwargs.get("fill_value")
            dtype = args[2] if len(args) > 2 else node.kwargs.get("dtype")
            return self.ops.full(shape, fill_value, dtype=dtype)
        return convert

    def _wrap_unsupported(self, name: str):
        def convert(node: fx.Node):
            raise NotImplementedError(f"{name} is not implemented")
        return convert

    def retrieve_args(self, node: fx.Node) -> list:
        return self._retrieve_args(node.args)

    def _retrieve_args(self, args):
        if isinstance(args, fx.Node):
            return self.env[args]
        elif isinstance(args, (list, tuple)):
            return [self._retrieve_args(a) for a in args]
        elif isinstance(args, dict):
            return {k: self._retrieve_args(v) for k, v in args.items()}
        else:
            return args

    def _tensor_type_parts(self, tensor_type: mim.Def) -> tuple[list[mim.Def], mim.Def]:
        dims = []
        elem_type = tensor_type
        while isinstance(elem_type, mim.Seq):
            dims.append(elem_type.arity())
            elem_type = elem_type.body()
        return dims, elem_type

    def _rebuild_tensor_type(self, dims: list[mim.Def], elem_type: mim.Def) -> mim.Def:
        if not dims:
            return elem_type
        if len(dims) == 1:
            return self.world.arr(dims[0], elem_type)
        return self.world.arr(self.world.tuple(dims), elem_type)

    def _specialize_input_type(self, tensor_type: mim.Def, input_index: int) -> mim.Def:
        if not hasattr(self, "input_sym_names") or input_index >= len(self.input_sym_names):
            return tensor_type

        dims, elem_type = self._tensor_type_parts(tensor_type)
        if not dims:
            return tensor_type

        changed = False
        specialized_dims = []
        for dim, sym_name in zip(dims, self.input_sym_names[input_index]):
            if sym_name is not None and sym_name in self.ops.sym_map:
                specialized_dims.append(self.ops.sym_map[sym_name])
                changed = True
            else:
                specialized_dims.append(dim)
        specialized_dims.extend(dims[len(specialized_dims):])

        if not changed:
            return tensor_type
        return self._rebuild_tensor_type(specialized_dims, elem_type)

    def translate_as_function(self, graph: fx.Graph, input_types: list[mim.Def], name: str = "main", sym_names: list[str] = None) -> mim.Lam:
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        param_nodes = [node for node in graph.nodes if node.op == "get_attr"]
        num_inputs = len(placeholders) + len(param_nodes)
        num_sym = len(sym_names) if sym_names else 0

        old_sym_map = self.ops.sym_map
        num_params = len(input_types) + 1
        dom_with_ret = self.world.mut_sigma(num_params)

        for i in range(num_sym):
            dom_with_ret.set(i, self.world.type_nat())

        sigma_var = dom_with_ret.var()
        sigma_sym_params = [sigma_var.proj(num_params, i) for i in range(num_sym)]

        if sym_names:
            for sym_name, sym_param in zip(sym_names, sigma_sym_params):
                self.ops.sym_map[sym_name] = sym_param

        for i, tensor_type in enumerate(input_types[num_sym:num_sym + len(placeholders)]):
            dom_with_ret.set(num_sym + i, self._specialize_input_type(tensor_type, i))

        for i, param_type in enumerate(input_types[num_sym + len(placeholders):]):
            dom_with_ret.set(num_sym + len(placeholders) + i, param_type)

        lam = self.world.mut_con(dom_with_ret)
        lam.set(name)

        lam_sym_params = [lam.var().proj(num_params, i) for i in range(num_sym)]
        if sym_names:
            for sym_name, sym_param in zip(sym_names, lam_sym_params):
                self.ops.sym_map[sym_name] = sym_param

        actual_inputs = [lam.var().proj(num_params, i) for i in range(num_sym, num_sym + num_inputs)]
        if hasattr(self, "input_shapes"):
            for actual_input, shape in zip(actual_inputs[:len(placeholders)], self.input_shapes):
                self.ops._remember_shape(actual_input, shape)
        if hasattr(self, "param_shapes"):
            for actual_input, shape in zip(actual_inputs[len(placeholders):], self.param_shapes):
                self.ops._remember_shape(actual_input, shape)
        result = self.translate(graph, actual_inputs)

        dom_with_ret.set(num_params - 1, self.world.cn([result.type()]))
        ret_cont = lam.var().proj(num_params, num_params - 1)
        lam.app(True, ret_cont, [result])
        lam.externalize()

        self.ops.sym_map = old_sym_map
        return lam

    def translate(self, graph: fx.Graph, inputs: list[mim.Def]) -> mim.Def:
        self.env = {}
        placeholders = [node for node in graph.nodes if node.op == "placeholder"]
        param_nodes = [node for node in graph.nodes if node.op == "get_attr"]

        # Map placeholders to first part of inputs
        for node, arg in zip(placeholders, inputs[:len(placeholders)]):
            arg.set(node.name)
            self.env[node] = arg

        # Map get_attr to the rest of inputs
        for node, arg in zip(param_nodes, inputs[len(placeholders):]):
            arg.set(node.name)
            self.env[node] = arg

        for node in graph.nodes:
            if node.op in ("placeholder", "get_attr"):
                continue
            elif node.op in ("call_function", "call_method"):
                res = self.convert_node(node)
                if isinstance(res, (mim.Lam, mim.App)):
                    res.set(node.name)
                self.env[node] = res
            elif node.op == "output":
                res = node.args[0]
                if isinstance(res, fx.Node):
                    return self.env[res]
                elif isinstance(res, (list, tuple)):
                    return self.world.tuple([self.env[n] if isinstance(n, fx.Node) else n for n in res])
                else:
                    return res
            else:
                raise NotImplementedError(f"Op {node.op} not implemented")


    # def _convert_tensor_constant(self, tensor: torch.Tensor) -> mim.Def:
    #     # For now, let's treat weights as placeholders too, or real constants?
    #     # If we want a completely closed module, we should embed them or pass them as args.
    #     # Passing as args is cleaner for now.
    #     # But if they are get_attr, they are already in the graph.
        
    #     # Simple strategy: Create a MimIR constant array if it's small, 
    #     # or just a placeholder-like mutable if it's large.
    #     # Given the requirement for a "mimir_module", maybe we should let the user
    #     # decide which parameters become arguments.
        
    #     # For now, let's just create a mutable with the right type.
    #     shape = list(tensor.shape)
    #     dtype = tensor.dtype
    #     if dtype == torch.float32:
    #         elem_t = self.ops.F32
    #     elif dtype == torch.bool:
    #         elem_t = self.ops.Bool
    #     else:
    #         raise NotImplementedError(f"Tensor constant with dtype {dtype} not supported")
            
    #     mim_shape = self.world.tuple([self.world.lit_nat(d) for d in shape])
    #     return self.world.mut_con(self.world.arr(mim_shape, elem_t)).var()


    def convert_node(self, node: fx.Node) -> mim.Def:
        target = node.target
        
        if target in self.convert_map:
            return self.convert_map[target](node)
        
        target_text = str(target)
        if target_text in self.convert_map:
            return self.convert_map[target_text](node)

        if isinstance(target, str) and target in self.convert_map:
             return self.convert_map[target](node)

        if hasattr(target, "name"):
            name = target.name()
            name = name.replace("::", ".")
            if name in self.convert_map:
                return self.convert_map[name](node)
        
        raise NotImplementedError(f"Target {target} (type {type(target)}) not supported")

def get_high_level_phase(world: mim.World) -> mim.Def:
    from mim._plugins.tensor import tensor as mim_tensor
    
    internal_cleanup = world.annex(mim_compile.internal_cleanup.value)
    lower_tensor = world.annex(mim_tensor.lower_tensor.value)
    fuse_tensor = world.annex(mim_tensor.fuse_tensor.value)
    
    phases = [internal_cleanup, lower_tensor, fuse_tensor, internal_cleanup]
    return world.call(mim_compile.phases, world.lit_bool(False), phases)
