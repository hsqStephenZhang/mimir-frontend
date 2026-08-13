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
        self.env: dict[fx.Node, object] = {}
        self.convert_map: dict[str | Callable, Callable[[fx.Node], object]] = self.create_convert_map()

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
            if t and hasattr(t, "__name__"):
                m[t.__name__] = wrapper
            if op: m[op] = wrapper
            for name in names:
                m[name] = wrapper

        # `aten.add` and `aten.sub` have a keyword-only alpha parameter. Keep
        # it at the Torch boundary instead of silently dropping it in the
        # generic two-argument wrapper.
        for torch_op, python_op, names, func in binary_ops[:2]:
            wrapper = self._wrap_alpha_binary(func)
            m[torch_op] = wrapper
            m[torch_op.__name__] = wrapper
            m[python_op] = wrapper
            for name in names:
                m[name] = wrapper

        m[operator.iadd] = self._wrap_binary(self.ops.add)
        m["aten.add_.Tensor"] = self._wrap_binary(self.ops.add)

        m[torch.clamp] = self._wrap_clamp()
        m["aten.clamp.default"] = self._wrap_clamp()
        m["aten.clamp.Tensor"] = self._wrap_clamp()
        m[torch.nn.functional.hardtanh] = self._wrap_hardtanh()
        m["aten.hardtanh.default"] = self._wrap_hardtanh()
        threshold = self._wrap_threshold()
        m[torch.threshold] = threshold
        m[torch.nn.functional.threshold] = threshold
        if hasattr(torch.nn.functional, "_threshold"):
            m[torch.nn.functional._threshold] = threshold
        m["aten.threshold.default"] = threshold
        m["threshold"] = threshold
        for torch_op, name, op_func in (
            (torch.addcmul, "addcmul", self.ops.addcmul),
            (torch.addcdiv, "addcdiv", self.ops.addcdiv),
        ):
            wrapper = self._wrap_addc(op_func)
            m[torch_op] = wrapper
            m[name] = wrapper
            m[f"aten.{name}.default"] = wrapper
        m[operator.floordiv] = self._wrap_nat_binary(self.ops.nat_floordiv)

        # Elementwise Unary
        unary_ops = [
            (torch.relu, "aten.relu.default", self.ops.relu),
            (torch.nn.functional.relu, "relu", self.ops.relu),
            (torch.exp, "aten.exp.default", self.ops.exp),
            (torch.tanh, "aten.tanh.default", self.ops.tanh),
            (torch.sqrt, "aten.sqrt.default", self.ops.sqrt),
            (torch.sin, "aten.sin.default", self.ops.sin),
            (torch.cos, "aten.cos.default", self.ops.cos),
            (torch.abs, "aten.abs.default", self.ops.abs),
            (torch.neg, "aten.neg.default", self.ops.neg),
            (torch.sigmoid, "aten.sigmoid.default", self.ops.sigmoid),
            (torch.nn.functional.silu, "aten.silu.default", self.ops.silu),
            (torch.reciprocal, "aten.reciprocal.default", self.ops.reciprocal),
            (torch.rsqrt, "aten.rsqrt.default", self.ops.rsqrt),
            (torch.logical_not, "aten.logical_not.default", self.ops.logical_not),
        ]
        
        for t, name, func in unary_ops:
            wrapper = self._wrap_unary(func)
            if t: m[t] = wrapper
            if t and hasattr(t, "__name__"):
                m[t.__name__] = wrapper
            if name: m[name] = wrapper
        m[operator.neg] = self._wrap_unary(self.ops.neg)
        m["aten.relu_.default"] = self._wrap_unary(self.ops.relu)
        leaky_relu = self._wrap_leaky_relu()
        m[torch.nn.functional.leaky_relu] = leaky_relu
        m["leaky_relu"] = leaky_relu
        m["aten.leaky_relu.default"] = leaky_relu
        gelu = self._wrap_gelu()
        m[torch.nn.functional.gelu] = gelu
        m["gelu"] = gelu
        m["aten.gelu.default"] = gelu

        # Prims
        if hasattr(torch.ops, "prims") and hasattr(torch.ops.prims, "convert_element_type"):
            m[torch.ops.prims.convert_element_type.default] = self._wrap_convert_element_type()
        m["prims.convert_element_type.default"] = self._wrap_convert_element_type()
        m["aten.convert_element_type.default"] = self._wrap_convert_element_type()
        m["prims.fma.default"] = self._wrap_fma()
        m["float"] = self._wrap_to_dtype(torch.float32)
        m["to"] = self._wrap_to()
        m["aten.to.dtype"] = self._wrap_to()
        m["aten.to.dtype_layout"] = self._wrap_to()

        # Injective
        m[torch.cat] = self._wrap_cat()
        m["aten.cat.default"] = self._wrap_cat()
        m[torch.permute] = self._wrap_permute()
        m["permute"] = self._wrap_permute()
        m["aten.permute.default"] = self._wrap_permute()
        m["aten.transpose.int"] = self._wrap_transpose()
        m["t"] = self._wrap_t()
        m["transpose"] = self._wrap_transpose()
        
        m[torch.reshape] = self._wrap_reshape()
        m["aten.reshape.default"] = self._wrap_reshape()
        m["reshape"] = self._wrap_reshape()
        m["view"] = self._wrap_reshape()
        m["aten.view.default"] = self._wrap_reshape()
        m["aten._unsafe_view.default"] = self._wrap_reshape()
        m["repeat"] = self._wrap_repeat()
        m["aten.repeat.default"] = self._wrap_repeat()
        m[torch.diff] = self._wrap_diff()
        m["diff"] = self._wrap_diff()
        m["aten.diff.default"] = self._wrap_diff()
        m[torch.cumsum] = self._wrap_cumsum()
        m["cumsum"] = self._wrap_cumsum()
        m["aten.cumsum.default"] = self._wrap_cumsum()
        m[torch.cumprod] = self._wrap_cumprod()
        m["aten.cumprod.default"] = self._wrap_cumprod()
        m[torch.roll] = self._wrap_roll()
        m["aten.roll.default"] = self._wrap_roll()
        m["unfold"] = self._wrap_unfold()
        m["aten.unfold.default"] = self._wrap_unfold()
        m[torch.flatten] = self._wrap_flatten()
        m["aten.flatten.using_ints"] = self._wrap_flatten()
        m["flatten"] = self._wrap_flatten()

        m["aten.slice.Tensor"] = self._wrap_slice()
        m["aten.select.int"] = self._wrap_select()
        m["select"] = self._wrap_select()
        m["aten.split.Tensor"] = self._wrap_split()
        m["split"] = self._wrap_split()
        
        m[torch.squeeze] = self._wrap_squeeze()
        m["squeeze"] = self._wrap_squeeze()
        m["aten.squeeze.dim"] = self._wrap_squeeze()
        m["aten.squeeze.dims"] = self._wrap_squeeze()
        
        m[torch.unsqueeze] = self._wrap_unsqueeze()
        m["unsqueeze"] = self._wrap_unsqueeze()
        m["aten.unsqueeze.default"] = self._wrap_unsqueeze()
        m[torch.nn.functional.pad] = self._wrap_pad()
        m["aten.pad.default"] = self._wrap_pad()
        m["contiguous"] = self._wrap_contiguous()
        m["aten.contiguous.default"] = self._wrap_contiguous()
        m["detach_"] = self._wrap_alias()
        m["aten.detach_.default"] = self._wrap_alias()
        m["aten.detach.default"] = self._wrap_alias()
        m["size"] = self._wrap_size()

        if hasattr(torch._C, "_log_api_usage_once"):
            m[torch._C._log_api_usage_once] = lambda node: self.world.lit_tt()

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
        m[torch.tensor] = self._wrap_tensor_constant()
        m["aten.full.default"] = self._wrap_full()
        m[torch.zeros_like] = self._wrap_zeros_like()
        m["aten.zeros_like.default"] = self._wrap_zeros_like()
        m[torch.zeros] = self._wrap_zeros()
        m["aten.zeros.default"] = self._wrap_zeros()
        m[torch.ones] = self._wrap_ones()
        m["aten.ones.default"] = self._wrap_ones()
        m["new_ones"] = self._wrap_new_fill(1)
        m["aten.new_ones.default"] = self._wrap_new_fill(1)
        m["new_zeros"] = self._wrap_new_fill(0)
        m["aten.new_zeros.default"] = self._wrap_new_fill(0)
        m["aten.empty_strided.default"] = self._wrap_empty_strided()
        m["aten.fill.Scalar"] = self._wrap_fill_scalar()
        m[torch.arange] = self._wrap_arange()
        m["aten.arange.default"] = self._wrap_arange()
        m["aten.arange.start"] = self._wrap_arange()
        m["aten.arange.start_step"] = self._wrap_arange()

        # Reductions
        m[torch.sum] = self._wrap_reduction(self.ops.sum)
        m["sum"] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.default"] = self._wrap_reduction(self.ops.sum)
        m["aten.sum.dim_IntList"] = self._wrap_reduction(self.ops.sum)
        m[torch.amax] = self._wrap_reduction(self.ops.amax)
        m["aten.amax.default"] = self._wrap_reduction(self.ops.amax)
        m[torch.max] = self._wrap_max("max")
        m["aten.max.default"] = self._wrap_max("max")
        m["aten.max.dim"] = self._wrap_max("max")
        m[torch.min] = self._wrap_max("min")
        m["aten.min.default"] = self._wrap_max("min")
        m["aten.min.dim"] = self._wrap_max("min")
        m[torch.mean] = self._wrap_reduction(self.ops.mean)
        m["mean"] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.default"] = self._wrap_reduction(self.ops.mean)
        m["aten.mean.dim"] = self._wrap_reduction(self.ops.mean)
        # https://pytorch.org/docs/stable/generated/torch.var_mean.html
        # Public/correction schema: (Tensor, dim?, *, correction?, keepdim)
        m[torch.var_mean] = self._wrap_var_mean_correction()
        m["aten.var_mean.correction"] = self._wrap_var_mean_correction()
        # Native legacy schemas use `unbiased` instead of `correction`.
        # aten.var_mean: (Tensor, unbiased) -> (Tensor, Tensor)
        # aten.var_mean.dim: (Tensor, dim?, unbiased, keepdim) -> pair
        m["aten.var_mean.default"] = self._wrap_var_mean_unbiased(has_dim=False)
        m["aten.var_mean.dim"] = self._wrap_var_mean_unbiased(has_dim=True)
        m[torch.softmax] = self._wrap_softmax()
        m[torch.nn.functional.softmax] = self._wrap_softmax_int()
        m["aten._softmax.default"] = self._wrap_softmax()
        m["aten.softmax.int"] = self._wrap_softmax_int()
        m[torch.log_softmax] = self._wrap_log_softmax()
        m[torch.nn.functional.log_softmax] = self._wrap_log_softmax()
        m["aten._log_softmax.default"] = self._wrap_log_softmax()
        m["aten.log_softmax.int"] = self._wrap_log_softmax()
        m[torch.nn.functional.layer_norm] = self._wrap_layer_norm(native=False)
        m[torch.native_layer_norm] = self._wrap_layer_norm(native=True)
        m["aten.native_layer_norm.default"] = self._wrap_layer_norm(native=True)

        # Normalization
        m[torch.nn.functional.batch_norm] = self._wrap_functional_batch_norm()
        m["batch_norm"] = self._wrap_functional_batch_norm()
        m["aten.batch_norm.default"] = self._wrap_aten_batch_norm()

        # Linear Algebra
        m[torch.mm] = self._wrap_binary(self.ops.mm)
        m[torch.bmm] = self._wrap_binary(self.ops.bmm)
        m[torch.matmul] = self._wrap_binary(self.ops.matmul)
        m[operator.matmul] = self._wrap_binary(self.ops.matmul)
        m["aten.mm.default"] = self._wrap_binary(self.ops.mm)
        m["aten.bmm.default"] = self._wrap_binary(self.ops.bmm)
        m["aten.matmul.default"] = self._wrap_binary(self.ops.matmul)
        m[torch.addmm] = self._wrap_addmm()
        m["aten.addmm.default"] = self._wrap_addmm()
        if hasattr(torch, "_C") and hasattr(torch._C, "_nn") and hasattr(torch._C._nn, "linear"):
            m[torch._C._nn.linear] = self._wrap_linear()
        m["aten.linear.default"] = self._wrap_linear()

        # Standard recurrent module entry points. Dynamo emits these operators
        # when torch._dynamo.config.allow_rnn is enabled.
        for target, name, relu in (
            (getattr(torch, "rnn_tanh", None), "rnn_tanh", False),
            (getattr(torch, "rnn_relu", None), "rnn_relu", True),
        ):
            wrapper = self._wrap_recurrent("rnn", relu=relu)
            if target is not None:
                m[target] = wrapper
            m[name] = wrapper
            m[f"aten.{name}.input"] = wrapper
        for kind in ("gru", "lstm"):
            wrapper = self._wrap_recurrent(kind)
            target = getattr(torch, kind, None)
            if target is not None:
                m[target] = wrapper
            m[kind] = wrapper
            m[f"aten.{kind}.input"] = wrapper
        m["torch._C._nn.linear"] = self._wrap_linear()

        # Convolution
        m[torch.convolution] = self._wrap_convolution()
        if hasattr(torch, "conv1d"):
            m[torch.conv1d] = self._wrap_conv1d()
        if hasattr(torch, "conv2d"):
            m[torch.conv2d] = self._wrap_conv2d()
        m[torch.nn.functional.conv1d] = self._wrap_conv1d()
        m[torch.nn.functional.conv2d] = self._wrap_conv2d()
        m["aten.convolution.default"] = self._wrap_convolution()
        m["aten.conv1d.default"] = self._wrap_conv1d()
        m["aten.conv2d.default"] = self._wrap_conv2d()
        m["conv1d"] = self._wrap_conv1d()
        m["conv2d"] = self._wrap_conv2d()

        # Pooling
        m[torch.nn.functional.max_pool2d] = self._wrap_max_pool2d()
        m["aten.max_pool2d.default"] = self._wrap_max_pool2d()
        m["aten.max_pool2d_with_indices.default"] = self._wrap_max_pool2d(return_indices=True)
        m[torch.nn.functional.avg_pool2d] = self._wrap_avg_pool2d()
        m["aten.avg_pool2d.default"] = self._wrap_avg_pool2d()
        m[torch.nn.functional.max_pool1d] = self._wrap_max_pool1d()
        m["aten.max_pool1d.default"] = self._wrap_max_pool1d()
        m[torch.nn.functional.max_pool3d] = self._wrap_max_pool3d()
        m["aten.max_pool3d.default"] = self._wrap_max_pool3d()
        m[torch.nn.functional.avg_pool1d] = self._wrap_avg_pool1d()
        m["aten.avg_pool1d.default"] = self._wrap_avg_pool1d()
        m[torch.nn.functional.avg_pool3d] = self._wrap_avg_pool3d()
        m["aten.avg_pool3d.default"] = self._wrap_avg_pool3d()
        m[torch.nn.functional.adaptive_avg_pool2d] = self._wrap_adaptive_avg_pool2d()
        m["aten.adaptive_avg_pool2d.default"] = self._wrap_adaptive_avg_pool2d()
        m["adaptive_avg_pool2d"] = self._wrap_adaptive_avg_pool2d()
        m[torch.nn.functional.adaptive_avg_pool1d] = self._wrap_adaptive_avg_pool1d()
        m["aten.adaptive_avg_pool1d.default"] = self._wrap_adaptive_avg_pool1d()
        m[torch.nn.functional.adaptive_avg_pool3d] = self._wrap_adaptive_avg_pool3d()
        m["aten.adaptive_avg_pool3d.default"] = self._wrap_adaptive_avg_pool3d()

        # Indexing / Scatter
        m["aten.index.Tensor"] = self._wrap_index_tensor()
        m[torch.gather] = self._wrap_gather()
        m["aten.gather.default"] = self._wrap_gather()
        m["aten.scatter.src"] = self._wrap_scatter_src()
        m["aten.scatter.default"] = self._wrap_scatter_src()
        m["aten.scatter.value"] = self._wrap_scatter_value()
        m["aten.scatter_add.default"] = self._wrap_unsupported("aten.scatter_add")
        m[torch.nn.functional.embedding] = self._wrap_functional_embedding()
        m["aten.embedding.default"] = self._wrap_embedding()
        m["aten.alias.default"] = self._wrap_alias()
        m["aten._assert_tensor_metadata.default"] = self._wrap_assert_tensor_metadata()
        m[torch.flip] = self._wrap_flip()
        m["flip"] = self._wrap_flip()
        m["aten.flip.default"] = self._wrap_flip()
        m[torch.narrow] = self._wrap_narrow()
        m["narrow"] = self._wrap_narrow()
        m["aten.narrow.default"] = self._wrap_narrow()

        # Selection
        m[torch.where] = self._wrap_where()
        m["aten.where.self"] = self._wrap_where()
        m[torch.triu] = self._wrap_triu()
        m["aten.triu.default"] = self._wrap_triu()
        m[torch.tril] = self._wrap_tril()
        m["aten.tril.default"] = self._wrap_tril()

        # Tuple operations
        m[operator.getitem] = self._wrap_getitem()
        m[builtins.getattr] = self._wrap_getattr()

        return m

    def _wrap_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0], args[1])
        return convert

    def _wrap_alpha_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            alpha = args[2] if len(args) > 2 else node.kwargs.get("alpha", 1)
            return op_func(args[0], args[1], alpha=alpha)
        return convert

    def _wrap_nat_binary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0], args[1])
        return convert

    def _wrap_addmm(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            self_tensor, mat1, mat2 = args[:3]
            beta = args[3] if len(args) > 3 else node.kwargs.get("beta", 1)
            alpha = args[4] if len(args) > 4 else node.kwargs.get("alpha", 1)
            return self.ops.addmm(
                self_tensor, mat1, mat2, beta=beta, alpha=alpha
            )
        return convert

    def _wrap_linear(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else node.kwargs.get("bias", None)
            return self.ops.linear(input, weight, bias=bias)
        return convert

    def _wrap_recurrent(self, kind, *, relu=False):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            if len(args) != 9:
                raise NotImplementedError(
                    f"{kind} currently supports the standard 9-argument input overload"
                )
            return self.ops.recurrent(
                kind,
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                args[8],
                relu=relu,
            )
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

    def _wrap_conv1d(self):
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

    def _wrap_max_pool1d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            dilation = args[4] if len(args) > 4 else node.kwargs.get("dilation", 1)
            ceil_mode = args[5] if len(args) > 5 else node.kwargs.get("ceil_mode", False)
            return self.ops.max_pool1d(
                x, kernel_size, stride=stride, padding=padding,
                dilation=dilation, ceil_mode=ceil_mode
            )
        return convert

    def _wrap_max_pool3d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            dilation = args[4] if len(args) > 4 else node.kwargs.get("dilation", 1)
            ceil_mode = args[5] if len(args) > 5 else node.kwargs.get("ceil_mode", False)
            return self.ops.max_pool3d(
                x, kernel_size, stride=stride, padding=padding,
                dilation=dilation, ceil_mode=ceil_mode
            )
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

    def _wrap_avg_pool1d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            ceil_mode = args[4] if len(args) > 4 else node.kwargs.get("ceil_mode", False)
            count_include_pad = args[5] if len(args) > 5 else node.kwargs.get("count_include_pad", True)
            return self.ops.avg_pool1d(
                x, kernel_size, stride=stride, padding=padding,
                ceil_mode=ceil_mode, count_include_pad=count_include_pad
            )
        return convert

    def _wrap_avg_pool3d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            kernel_size = args[1] if len(args) > 1 else node.kwargs.get("kernel_size")
            stride = args[2] if len(args) > 2 else node.kwargs.get("stride", None)
            padding = args[3] if len(args) > 3 else node.kwargs.get("padding", 0)
            ceil_mode = args[4] if len(args) > 4 else node.kwargs.get("ceil_mode", False)
            count_include_pad = args[5] if len(args) > 5 else node.kwargs.get("count_include_pad", True)
            divisor_override = args[6] if len(args) > 6 else node.kwargs.get("divisor_override", None)
            return self.ops.avg_pool3d(
                x, kernel_size, stride=stride, padding=padding,
                ceil_mode=ceil_mode, count_include_pad=count_include_pad,
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

    def _wrap_adaptive_avg_pool1d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            output_size = args[1] if len(args) > 1 else node.kwargs.get("output_size")
            return self.ops.adaptive_avg_pool1d(args[0], output_size)
        return convert

    def _wrap_adaptive_avg_pool3d(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            output_size = args[1] if len(args) > 1 else node.kwargs.get("output_size")
            return self.ops.adaptive_avg_pool3d(args[0], output_size)
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

    def _wrap_gather(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.gather(args[0], args[2], dim=args[1])
        return convert

    def _wrap_scatter_src(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.scatter_src(args[0], args[1], args[2], args[3])
        return convert

    def _wrap_scatter_value(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.scatter_value(args[0], args[1], args[2], args[3])
        return convert

    def _wrap_embedding(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            padding_idx = args[2] if len(args) > 2 else -1
            scale_grad_by_freq = args[3] if len(args) > 3 else False
            sparse = args[4] if len(args) > 4 else False
            return self.ops.embedding(args[0], args[1], padding_idx, scale_grad_by_freq, sparse)
        return convert

    def _wrap_functional_embedding(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            padding_idx = args[2] if len(args) > 2 else node.kwargs.get("padding_idx", -1)
            if padding_idx is None:
                padding_idx = -1
            max_norm = args[3] if len(args) > 3 else node.kwargs.get("max_norm")
            if max_norm is not None:
                raise NotImplementedError("embedding max_norm is not implemented")
            scale_grad_by_freq = (
                args[5] if len(args) > 5
                else node.kwargs.get("scale_grad_by_freq", False)
            )
            sparse = args[6] if len(args) > 6 else node.kwargs.get("sparse", False)
            return self.ops.embedding(
                args[1], args[0], padding_idx, scale_grad_by_freq, sparse
            )
        return convert

    def _wrap_alias(self):
        def convert(node: fx.Node):
            return self.retrieve_args(node)[0]
        return convert

    def _wrap_assert_tensor_metadata(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.assert_tensor_metadata(
                args[0],
                size=args[1] if len(args) > 1 else node.kwargs.get("size"),
                stride=args[2] if len(args) > 2 else node.kwargs.get("stride"),
                dtype=args[3] if len(args) > 3 else node.kwargs.get("dtype"),
                device=node.kwargs.get("device"),
                layout=node.kwargs.get("layout"),
            )
        return convert

    def _wrap_unary(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return op_func(args[0])
        return convert

    def _wrap_gelu(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            approximate = (
                args[1]
                if len(args) > 1
                else node.kwargs.get("approximate", "none")
            )
            return self.ops.gelu(args[0], approximate=approximate)
        return convert

    def _wrap_leaky_relu(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            negative_slope = (
                args[1]
                if len(args) > 1
                else node.kwargs.get("negative_slope", 0.01)
            )
            inplace = (
                args[2] if len(args) > 2 else node.kwargs.get("inplace", False)
            )
            if inplace:
                raise NotImplementedError("in-place leaky_relu is not supported")
            return self.ops.leaky_relu(args[0], negative_slope=negative_slope)
        return convert

    def _wrap_reduction(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", None)
            keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
            return op_func(input_def, dim=dim, keepdim=keepdim)
        return convert

    def _wrap_max(self, kind):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            if len(args) > 1 or "dim" in node.kwargs:
                dim = args[1] if len(args) > 1 else node.kwargs["dim"]
                keepdim = args[2] if len(args) > 2 else node.kwargs.get("keepdim", False)
                return self.ops.dim_extrema(
                    args[0], dim, keepdim=keepdim, kind=kind
                )
            if kind == "min":
                raise NotImplementedError("torch.min without dim is not implemented")
            return self.ops.amax(args[0], dim=None, keepdim=False)
        return convert

    def _wrap_var_mean_correction(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
            return self.ops.var_mean(
                args[0],
                dim=dim,
                keepdim=kwargs.get("keepdim", False),
                correction=kwargs.get("correction", 1),
            )
        return convert

    def _wrap_var_mean_unbiased(self, *, has_dim: bool):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            if has_dim:
                dim = args[1] if len(args) > 1 else None
                unbiased = args[2] if len(args) > 2 else True
                keepdim = args[3] if len(args) > 3 else False
            else:
                dim = None
                unbiased = args[1] if len(args) > 1 else True
                keepdim = False
            return self.ops.var_mean(
                args[0],
                dim=dim,
                keepdim=keepdim,
                correction=1 if unbiased else 0,
            )
        return convert

    def _wrap_softmax(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", -1)
            half_to_float = args[2] if len(args) > 2 else node.kwargs.get("half_to_float", False)
            if half_to_float is not False:
                raise NotImplementedError("softmax half_to_float=true is not implemented")
            return self.ops.softmax(input_def, dim=dim)
        return convert

    def _wrap_softmax_int(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", -1)
            dtype = args[2] if len(args) > 2 else node.kwargs.get("dtype")
            if dtype not in (None, torch.float32, torch.float):
                raise NotImplementedError(
                    f"aten.softmax.int with dtype {dtype} is not implemented"
                )
            return self.ops.softmax(input_def, dim=dim)
        return convert

    def _wrap_log_softmax(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            dim = args[1] if len(args) > 1 else node.kwargs.get("dim", -1)
            dtype = args[2] if len(args) > 2 else node.kwargs.get("dtype")
            if dtype not in (None, torch.float32, torch.float):
                raise NotImplementedError(
                    f"log_softmax with dtype {dtype} is not implemented"
                )
            return self.ops.log_softmax(args[0], dim=dim)
        return convert

    def _wrap_flip(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            dims = args[1] if len(args) > 1 else node.kwargs["dims"]
            return self.ops.flip(args[0], dims)
        return convert

    def _wrap_narrow(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.narrow(
                args[0],
                args[1] if len(args) > 1 else node.kwargs["dim"],
                args[2] if len(args) > 2 else node.kwargs["start"],
                args[3] if len(args) > 3 else node.kwargs["length"],
            )
        return convert

    def _wrap_layer_norm(self, *, native):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            input_def = args[0]
            normalized_shape = args[1]
            weight = args[2] if len(args) > 2 else node.kwargs.get("weight")
            bias = args[3] if len(args) > 3 else node.kwargs.get("bias")
            eps = args[4] if len(args) > 4 else node.kwargs.get("eps", 1e-5)
            result = self.ops.native_layer_norm(
                input_def, normalized_shape, weight, bias, eps
            )
            return result if native else result.proj(3, 0)
        return convert

    def _wrap_functional_batch_norm(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            input_def = args[0]
            running_mean = args[1] if len(args) > 1 else kwargs.get("running_mean")
            running_var = args[2] if len(args) > 2 else kwargs.get("running_var")
            weight = args[3] if len(args) > 3 else kwargs.get("weight")
            bias = args[4] if len(args) > 4 else kwargs.get("bias")
            training = args[5] if len(args) > 5 else kwargs.get("training", False)
            eps = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)
            if training:
                raise NotImplementedError("batch_norm training mode is not implemented")
            return self.ops.batch_norm_inference(
                input_def, running_mean, running_var, weight, bias, eps
            )
        return convert

    def _wrap_aten_batch_norm(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            input_def = args[0]
            weight = args[1] if len(args) > 1 else kwargs.get("weight")
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            running_mean = args[3] if len(args) > 3 else kwargs.get("running_mean")
            running_var = args[4] if len(args) > 4 else kwargs.get("running_var")
            training = args[5] if len(args) > 5 else kwargs.get("training", False)
            eps = args[7] if len(args) > 7 else kwargs.get("eps", 1e-5)
            if training:
                raise NotImplementedError("batch_norm training mode is not implemented")
            return self.ops.batch_norm_inference(
                input_def, running_mean, running_var, weight, bias, eps
            )
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
            if len(args) == 3:
                return self.ops.transpose_int(x, args[1], args[2])
            return self.ops.transpose(x, args[1])
        return convert

    def _wrap_permute(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            # Tensor.permute exposes dimensions as variadic method arguments;
            # torch.permute and aten.permute use one tuple/list argument.
            permutation = args[1] if len(args) == 2 else list(args[1:])
            return self.ops.transpose(args[0], permutation)
        return convert

    def _wrap_repeat(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            repeats = args[1] if len(args) == 2 else args[1:]
            return self.ops.repeat(args[0], repeats)
        return convert

    def _wrap_diff(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            return self.ops.diff(
                args[0],
                n=args[1] if len(args) > 1 else kwargs.get("n", 1),
                dim=args[2] if len(args) > 2 else kwargs.get("dim", -1),
                prepend=args[3] if len(args) > 3 else kwargs.get("prepend"),
                append=args[4] if len(args) > 4 else kwargs.get("append"),
            )
        return convert

    def _wrap_cumsum(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            dim = args[1] if len(args) > 1 else kwargs.get("dim")
            dtype = args[2] if len(args) > 2 else kwargs.get("dtype")
            return self.ops.cumsum(args[0], dim, dtype=dtype)
        return convert

    def _wrap_cumprod(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            dtype = args[2] if len(args) > 2 else node.kwargs.get("dtype")
            return self.ops.cumprod(args[0], args[1], dtype=dtype)
        return convert

    def _wrap_roll(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            shifts = args[1] if len(args) > 1 else kwargs.get("shifts")
            dims = args[2] if len(args) > 2 else kwargs.get("dims")
            return self.ops.roll(args[0], shifts, dims)
        return convert

    def _wrap_unfold(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.unfold(args[0], args[1], args[2], args[3])
        return convert

    def _wrap_getitem(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            obj = args[0]
            index = args[1]

            # Multi-result Torch operators stay as Python structure until FX
            # getitem selects a result. Their computation remains in MimIR.
            if isinstance(obj, (tuple, list)):
                if isinstance(index, int):
                    return obj[index]
                raise TypeError(
                    f"structured result requires an integer index, got {index!r}"
                )
            
            # Check if obj is a tensor by inspecting its type
            ty = obj.type()
            
            if isinstance(ty, (mim.Arr, mim.Seq)) or obj in self.ops._shape_cache:
                # Handle tensor indexing/slicing
                if isinstance(index, mim.Def) and isinstance(
                    index.type(), (mim.Arr, mim.Seq)
                ):
                    return self.ops.index_tensor(obj, index)
                if isinstance(index, int):
                    return self.ops.select(obj, 0, index)
                elif isinstance(index, slice):
                    return self.ops.slice(obj, 0, index.start or 0, index.stop, index.step or 1)
                elif isinstance(index, (tuple, list)):
                    rank = len(self.ops.shape_of(obj))
                    index = list(index)
                    consumed = sum(
                        item is not None and item is not Ellipsis for item in index
                    )
                    if index.count(Ellipsis) > 1:
                        raise IndexError("an index can only have a single ellipsis")
                    if Ellipsis in index:
                        ellipsis = index.index(Ellipsis)
                        fill = rank - consumed
                        index[ellipsis:ellipsis + 1] = [slice(None)] * fill
                    else:
                        index.extend([slice(None)] * (rank - consumed))

                    if (
                        rank == 2
                        and len(index) == 2
                        and all(isinstance(item, mim.Def) for item in index)
                    ):
                        return self.ops.index_2d(obj, index[0], index[1])

                    res = obj
                    axis = 0
                    for idx in index:
                        if idx is None:
                            res = self.ops.unsqueeze(res, axis)
                            axis += 1
                            continue
                        if isinstance(idx, slice):
                            res = self.ops.slice(
                                res,
                                axis,
                                idx.start or 0,
                                idx.stop,
                                idx.step or 1,
                            )
                            axis += 1
                        elif isinstance(idx, int):
                            res = self.ops.select(res, axis, idx)
                        else:
                            raise NotImplementedError(
                                f"tensor index component {idx!r} is not implemented"
                            )
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
            if attr_name == "T":
                rank = len(self.ops.shape_of(obj))
                return self.ops.transpose(obj, list(reversed(range(rank))))
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

    def _wrap_hardtanh(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            min_val = args[1] if len(args) > 1 else node.kwargs.get("min_val", -1.0)
            max_val = args[2] if len(args) > 2 else node.kwargs.get("max_val", 1.0)
            return self.ops.hardtanh(x, min_val=min_val, max_val=max_val)
        return convert

    def _wrap_threshold(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            inplace = args[3] if len(args) > 3 else node.kwargs.get("inplace", False)
            if inplace:
                raise NotImplementedError("in-place threshold is not supported")
            return self.ops.threshold(args[0], args[1], args[2])
        return convert

    def _wrap_addc(self, op_func):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            value = args[3] if len(args) > 3 else node.kwargs.get("value", 1)
            return op_func(args[0], args[1], args[2], value=value)
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

    def _wrap_to_dtype(self, dtype):
        def convert(node: fx.Node):
            return self.ops.convert_element_type(self.retrieve_args(node)[0], dtype)
        return convert

    def _wrap_to(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            x = args[0]
            dtype = node.kwargs.get("dtype")
            device = node.kwargs.get("device")
            for arg in args[1:]:
                if isinstance(arg, torch.dtype):
                    dtype = arg
                elif isinstance(arg, (torch.device, str)):
                    device = torch.device(arg)
            if device is not None and torch.device(device).type != "cpu":
                raise NotImplementedError("to(device) currently supports CPU only")
            if dtype is None:
                return x
            if (
                dtype in (torch.float32, torch.float)
                and self.ops._tensor_element_type(x) == self.ops.F32
            ):
                return x
            if dtype in (torch.int64, torch.long) and self.ops._tensor_element_type(x) == self.ops.I64:
                return x
            return self.ops.convert_element_type(x, dtype)
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

    def _wrap_triu(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            diagonal = args[1] if len(args) > 1 else node.kwargs.get("diagonal", 0)
            return self.ops.triu(args[0], diagonal=diagonal)
        return convert

    def _wrap_tril(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            diagonal = args[1] if len(args) > 1 else node.kwargs.get("diagonal", 0)
            return self.ops.tril(args[0], diagonal=diagonal)
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

    def _wrap_pad(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            mode = args[2] if len(args) > 2 else node.kwargs.get("mode", "constant")
            value = args[3] if len(args) > 3 else node.kwargs.get("value")
            return self.ops.pad(args[0], args[1], mode=mode, value=value)
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
            return self.ops.cat(tensors, dim=dim)
        return convert

    def _wrap_full(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            shape = args[0]
            fill_value = args[1] if len(args) > 1 else node.kwargs.get("fill_value")
            dtype = args[2] if len(args) > 2 else node.kwargs.get("dtype")
            return self.ops.full(shape, fill_value, dtype=dtype)
        return convert

    def _wrap_zeros_like(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            reference = args[0]
            elem_type = self.ops._tensor_element_type(reference)
            inferred_dtype = {
                self.ops.F32: torch.float32,
                self.ops.I64: torch.int64,
                self.ops.Bool: torch.bool,
            }.get(elem_type)
            dtype = kwargs.get("dtype") or inferred_dtype
            if dtype != inferred_dtype:
                raise NotImplementedError("zeros_like dtype conversion is not implemented")
            if kwargs.get("requires_grad", False):
                raise NotImplementedError("zeros_like requires_grad=True is not implemented")
            return self.ops.full(self.ops.shape_of(reference), 0, dtype=dtype)
        return convert

    def _wrap_tensor_constant(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            value = args[0]
            if isinstance(value, (list, tuple)):
                raise NotImplementedError("torch.tensor currently supports scalar constants")
            device = kwargs.get("device")
            if device is not None and torch.device(device).type != "cpu":
                raise NotImplementedError("torch.tensor currently supports CPU tensors")
            return self.ops.full([], value, dtype=kwargs.get("dtype"))
        return convert

    def _wrap_zeros(self):
        return self._wrap_constant_fill(0, "zeros")

    def _wrap_ones(self):
        return self._wrap_constant_fill(1, "ones")

    def _wrap_constant_fill(self, fill_value, operator_name):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            shape = args[0] if len(args) == 1 else tuple(args)
            device = node.kwargs.get("device")
            layout = node.kwargs.get("layout")
            pin_memory = node.kwargs.get("pin_memory")
            requires_grad = node.kwargs.get("requires_grad", False)
            if device is not None and torch.device(device).type != "cpu":
                raise NotImplementedError(
                    f"{operator_name} currently supports CPU tensors only"
                )
            if layout not in (None, torch.strided):
                raise NotImplementedError(
                    f"{operator_name} currently supports strided layout only"
                )
            if pin_memory not in (None, False):
                raise NotImplementedError(
                    f"{operator_name} pin_memory=True is not implemented"
                )
            if requires_grad:
                raise NotImplementedError(
                    f"{operator_name} requires_grad=True is not implemented"
                )
            return self.ops.full(
                shape,
                fill_value,
                dtype=node.kwargs.get("dtype"),
            )
        return convert

    def _wrap_new_fill(self, fill_value):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            kwargs = self._retrieve_args(node.kwargs)
            reference = args[0]
            shape = args[1] if len(args) == 2 else args[1:]
            dtype = kwargs.get("dtype")
            if dtype is None:
                elem_type = self.ops._tensor_element_type(reference)
                dtype = {
                    self.ops.F32: torch.float32,
                    self.ops.I64: torch.int64,
                    self.ops.Bool: torch.bool,
                }.get(elem_type)
            if dtype is None:
                raise NotImplementedError("new_ones/new_zeros cannot infer dtype")
            return self.ops.full(shape, fill_value, dtype=dtype)
        return convert

    def _wrap_empty_strided(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.empty_strided(
                args[0],
                args[1],
                dtype=node.kwargs.get("dtype"),
                layout=node.kwargs.get("layout"),
                device=node.kwargs.get("device"),
                pin_memory=node.kwargs.get("pin_memory"),
            )
        return convert

    def _wrap_fill_scalar(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            return self.ops.fill_scalar(args[0], args[1])
        return convert

    def _wrap_arange(self):
        def convert(node: fx.Node):
            args = self.retrieve_args(node)
            if len(args) == 1:
                start, end, step = 0, args[0], 1
            else:
                start, end = args[:2]
                step = args[2] if len(args) > 2 else 1
            return self.ops.arange(
                start,
                end,
                step,
                dtype=node.kwargs.get("dtype"),
                layout=node.kwargs.get("layout"),
                device=node.kwargs.get("device"),
                pin_memory=node.kwargs.get("pin_memory"),
            )
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
            for input_index, (actual_input, shape) in enumerate(
                zip(actual_inputs[:len(placeholders)], self.input_shapes)
            ):
                sym_names_for_input = self.input_sym_names[input_index]
                bound_shape = [
                    self.ops.sym_map[sym_name]
                    if sym_name is not None else dim
                    for dim, sym_name in zip(shape, sym_names_for_input)
                ]
                self.ops._remember_shape(actual_input, bound_shape)
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
                try:
                    res = self.convert_node(node)
                except Exception as exc:
                    exc.add_note(f"while translating FX node: {node.format_node()}")
                    raise
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
    from mim._plugins.torch import torch as mim_torch
    
    internal_cleanup = world.annex(mim_compile.internal_cleanup.value)
    lower_torch = world.annex(mim_torch.lower_torch.value)
    lower_tensor = world.annex(mim_tensor.lower_tensor.value)
    fuse_tensor = world.annex(mim_tensor.fuse_tensor.value)
    
    phases = [internal_cleanup, lower_torch, lower_tensor, fuse_tensor, internal_cleanup]
    return world.call(mim_compile.phases, world.lit_bool(False), phases)
