import mim
import struct
import torch
from mim._plugins.affine import affine
from mim._plugins.math import (
    math, _math_arith, _math_extrema, _math_cmp, _math_conv,
    _math_exp, _math_tri, _math_rt,
)
from mim._plugins.tensor import tensor
from mim._plugins.torch import torch as torch_dialect
from mim._plugins.option import option
from mim._plugins.core import core, _core_bit1, _core_bit2

from .shape_rules import ShapeRules
from . import expr

class OperatorLibrary:
    @staticmethod
    def _torch_annex_id(name: str) -> int:
        family, member = name.split(".", 1)
        return getattr(getattr(torch_dialect, family), member).value

    def _lit_nat(self, value):
        return self.world.lit_nat(value)

    def __init__(self, world: mim.World):
        self.world = world
        self.rules = ShapeRules(world)
        self.f32_config = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        self.F64 = world.annex(math.F64.value)
        self.I64 = world.type_i64()
        self.Bool = world.type_bool()
        self.mode0 = world.lit_nat_0()
        self.sym_map = {} # Mapping from symbolic name to MimIR Nat variable
        self._shape_cache: dict[mim.Def, list[mim.Def]] = {}
        self._torch_semantics_cache: dict[tuple[mim.Def, bool], mim.Def] = {}
        self._flip_provenance: dict[mim.Def, tuple[mim.Def, tuple[int, ...]]] = {}
        self._cumsum_provenance: dict[mim.Def, tuple[mim.Def, int, object]] = {}
        self._narrow_provenance: dict[mim.Def, tuple[mim.Def, int, object, object]] = {}
        self._known_zero_tensors: set[mim.Def] = set()

        def bind_math_axm(axm_enum):
            axm = world.annex(axm_enum.value)
            # %_math_arith.add {pe} mode
            axm = world.app(axm, self.f32_config)
            return world.app(axm, self.mode0)

        # Arithmetic
        self.f32_add_axm = bind_math_axm(_math_arith.add)
        self.f32_sub_axm = bind_math_axm(_math_arith.sub)
        self.f32_mul_axm = bind_math_axm(_math_arith.mul)
        self.f32_div_axm = bind_math_axm(_math_arith.div)
        
        # Extrema
        self.f32_max_axm = bind_math_axm(_math_extrema.fmax)
        self.f32_min_axm = bind_math_axm(_math_extrema.fmin)
        
        # Comparisons
        self.f32_eq_axm = bind_math_axm(_math_cmp.e)
        self.f32_ne_axm = bind_math_axm(_math_cmp.ne)
        self.f32_lt_axm = bind_math_axm(_math_cmp.l)
        self.f32_le_axm = bind_math_axm(_math_cmp.le)
        self.f32_gt_axm = bind_math_axm(_math_cmp.g)
        self.f32_ge_axm = bind_math_axm(_math_cmp.ge)
        
        # Unary
        self.f32_exp_axm = bind_math_axm(_math_exp.exp)
        self.f32_log_axm = bind_math_axm(_math_exp.log)
        self.f32_tanh_axm = bind_math_axm(_math_tri.tanh)
        self.f32_sin_axm = bind_math_axm(_math_tri.sin)
        self.f32_cos_axm = bind_math_axm(_math_tri.cos)
        self.f32_sqrt_axm = bind_math_axm(_math_rt.sq)
        self.f32_pow_axm = bind_math_axm(math.pow)
        self.f32_abs_axm = bind_math_axm(math.abs)
        self.f32_neg_axm = bind_math_axm(math.minus)
        
        # Complex
        self.f32_sigmoid_axm = bind_math_axm(math.slf)
        self.f32_rsqrt_axm = bind_math_axm(math.rrt)
        self.affine_index = world.annex(affine.index.value)

        # Bitwise/Logical
        s_bool = self._lit_nat(2)
        m_scalar = world.lit_nat_0()
        
        def bind_bit_axm(axm_enum):
            axm = world.annex(axm_enum.value)
            axm = world.app(axm, s_bool)
            return world.app(axm, m_scalar)
            
        self.bool_and_axm = bind_bit_axm(_core_bit2.and_)
        self.bool_not_axm = bind_bit_axm(_core_bit1.neg)

    def _torch_semantics(self, tensor_def, *, floating=False):
        """Resolve dtype-specific Torch scalar behavior inside MimIR."""
        elem_type = self._tensor_element_type(tensor_def)
        key = (elem_type, floating)
        cached = self._torch_semantics_cache.get(key)
        if cached is not None:
            return cached
        resolver = (
            torch_dialect.resolve_floating
            if floating
            else torch_dialect.resolve_arithmetic
        )
        resolved = self.world.app(
            self.world.annex(resolver.value),
            elem_type,
        )
        self._torch_semantics_cache[key] = resolved
        return resolved

    def _rank_and_shape(self, tensor_def):
        dims = self.shape_of(tensor_def)
        return self._lit_nat(len(dims)), self.world.tuple(dims)

    def _physical_dims(self, dims):
        """Drop folded singleton axes while preserving tensor-vs-scalar rank."""
        physical = [dim for dim in dims if not self._is_lit_nat_value(dim, 1)]
        if not physical and dims:
            # Nested singleton arrays normalize to their element type, but a
            # logical tensor still needs one extent-1 loop axis. Rank zero is
            # reserved for genuine scalar tensors.
            return [dims[-1]]
        return physical

    def _rank_and_type_shape(self, tensor_def):
        dims = self._shape_dims(tensor_def)
        return self._lit_nat(len(dims)), self.world.tuple(dims)

    def shape_of(self, value):
        """
        Unified way to retrieve shape dims from a MimIR Def or PyTorch object.
        Priority:
        1. Explicitly mapped symbolic dims (for inputs)
        2. Local shape cache (for derived tensors, preserves 1-dims)
        3. MimIR Type system (Arr arity)
        4. PyTorch metadata (FakeTensor)
        """
        cached = self._shape_cache.get(value)
        if cached is not None:
            return list(cached)

        if isinstance(value, mim.Def):
            # 1. Check symbolic map for inputs
            if hasattr(self, "input_to_syms") and value in self.input_to_syms:
                sym_names = self.input_to_syms[value]
                dims = self._shape_dims(value)
                final_dims = []
                for i, name in enumerate(sym_names):
                    if name is not None and name in self.sym_map:
                        final_dims.append(self.sym_map[name])
                    else:
                        final_dims.append(dims[i])
                return final_dims

            # 2. Check cache FIRST to preserve singleton (1) dimensions
            # because MimIR type system normalizes them away.
            cached = self._shape_cache.get(value)
            if cached is not None:
                return list(cached)

            # 3. Try type system
            dims = self._shape_dims(value)
            return dims
        
        # 4. Fallback to metadata
        if hasattr(value, "meta") and isinstance(value.meta, dict) and "val" in value.meta:
            return self.shape_of(value.meta["val"])
        if hasattr(value, "shape"):
             return [self._lit_nat(d) if isinstance(d, int) else d for d in value.shape]
             
        raise TypeError(f"shape_of does not support {type(value)}")

    def _remember_shape(self, value, dims):
        normalized = []
        for dim in dims:
            if isinstance(dim, int):
                normalized.append(self._lit_nat(dim))
            else:
                normalized.append(dim)
        try:
            self._shape_cache[value] = normalized
        except TypeError:
            pass
        return value

    def _shape_dims(self, tensor_def):
        dims = []
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            arity = tensor_type.arity()
            if isinstance(arity, mim.Tuple) and arity.num_projs() == 0:
                break
            dims.append(arity)
            tensor_type = tensor_type.body()
        return dims

    def _is_lit_nat_value(self, value, expected: int) -> bool:
        return isinstance(value, mim.Lit) and value.get_nat() == expected

    def _shape_debug(self, dims):
        return [
            dim.get_nat() if isinstance(dim, mim.Lit) else str(dim)
            for dim in dims
        ]

    def _apply_grouped(self, callee, args):
        return self.world.app(callee, self.world.tuple(args))

    def _affine_projection_lam(self, total_rank, output_rank, projections):
        vec_type = self.world.arr(self._lit_nat(total_rank), self.affine_index)
        out_type = self.world.arr(self._lit_nat(output_rank), self.affine_index)
        lam = self.world.mut_lam(vec_type, out_type)
        iters = lam.var()
        lam.set_body(True, self.world.tuple([iters.proj(total_rank, index) for index in projections]))
        return lam

    def _f32_reduce_lambda(self, op):
        args_type = self.world.arr(self._lit_nat(2), self.F32)
        lam = self.world.mut_con([args_type, self.world.cn([self.F32])])
        args = lam.var().proj(2, 0)
        reduced = self.world.app(op, [args.proj(2, 0), args.proj(2, 1)])
        lam.app(True, lam.var().proj(2, 1), [reduced])
        return lam

    def _scalar_binary_lambda(self, op, arg_type, ret_type):
        args_type = self.world.arr(self._lit_nat(2), arg_type)
        lam = self.world.mut_lam(args_type, ret_type)
        args = lam.var()
        lam.set_body(True, self.world.app(op, [args.proj(2, 0), args.proj(2, 1)]))
        return lam

    def _tensor_element_type(self, tensor_def):
        tensor_type = tensor_def.type()
        while isinstance(tensor_type, mim.Seq):
            tensor_type = tensor_type.body()
        return tensor_type

    def binary(self, op, lhs, rhs, out_type=None):
        """
        Translates to MimIR elementwise binary operation:
            %tensor.binary @(T_in, T_in, T_out) op @(rank, shape) (lhs, rhs)
        Example IR:
            %tensor.binary (%math.F (23, 8)) (%_math_arith.add (23, 8) 0) (2, (10, 20)) (lhs, rhs)
        """
        if isinstance(rhs, (int, float)):
            if out_type is None:
                out_type = self._tensor_element_type(lhs)
            lam = self._f32_unary_lambda(op, lambda v: [v, self._f32_float_lit(float(rhs))], ret_type=out_type)
            return self.unary(lam, lhs, out_type=out_type)
        if isinstance(lhs, (int, float)):
            if out_type is None:
                out_type = self._tensor_element_type(rhs)
            lam = self._f32_unary_lambda(op, lambda v: [self._f32_float_lit(float(lhs)), v], ret_type=out_type)
            return self.unary(lam, rhs, out_type=out_type)
            
        in_type = self._tensor_element_type(lhs)
        if out_type is None:
            out_type = in_type
            
        # Broadcasting logic
        s_lhs_dims = self.shape_of(lhs)
        s_rhs_dims = self.shape_of(rhs)
        output_dims = self.rules.broadcast_shape(s_lhs_dims, s_rhs_dims)
        
        if not self.rules.same_shape(s_lhs_dims, s_rhs_dims):
            if not self.rules.same_shape(s_lhs_dims, output_dims):
                lhs = self.expand(lhs, output_dims)
            if not self.rules.same_shape(s_rhs_dims, output_dims):
                rhs = self.expand(rhs, output_dims)

        physical_dims = self._physical_dims(output_dims)
        rank, shape = self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)
        callee = self.world.annex(tensor.binary.value)
        callee = self._apply_grouped(callee, [in_type, in_type, out_type])
        callee = self.world.app(callee, self._scalar_binary_lambda(op, in_type, out_type))
        callee = self._apply_grouped(callee, [rank, shape])
        res = self.world.app(callee, self.world.tuple([lhs, rhs]))
        
        return self._remember_shape(res, output_dims)

    def _torch_binary(self, name, lhs, rhs, out_type=None, *, alpha=None):
        """Emit a Torch dialect binary op after frontend broadcasting."""
        if isinstance(rhs, (int, float)) or isinstance(lhs, (int, float)):
            tensor_value = lhs if isinstance(lhs, mim.Def) else rhs
            scalar_value = rhs if isinstance(lhs, mim.Def) else lhs
            if (
                name.startswith("comparison.")
                and self._tensor_element_type(tensor_value) == self.I64
            ):
                comparison = name.removeprefix("comparison.")
                if not isinstance(lhs, mim.Def):
                    comparison = {
                        "eq": "eq",
                        "ne": "ne",
                        "lt": "gt",
                        "le": "ge",
                        "gt": "lt",
                        "ge": "le",
                    }[comparison]
                return self._torch_scalar(
                    f"comparison.{comparison}_i64_scalar",
                    tensor_value,
                    scalar_value,
                )
            scalar_ops = {
                "binary.add": self.f32_add_axm, "binary.sub": self.f32_sub_axm,
                "binary.mul": self.f32_mul_axm, "binary.div": self.f32_div_axm,
                "binary.maximum": self.f32_max_axm, "binary.minimum": self.f32_min_axm,
                "comparison.eq": self.f32_eq_axm, "comparison.ne": self.f32_ne_axm,
                "comparison.lt": self.f32_lt_axm, "comparison.le": self.f32_le_axm,
                "comparison.gt": self.f32_gt_axm, "comparison.ge": self.f32_ge_axm,
            }
            return self.binary(scalar_ops[name], lhs, rhs, out_type=out_type)
        lhs_dims = self.shape_of(lhs)
        rhs_dims = self.shape_of(rhs)
        try:
            output_dims = self.rules.broadcast_shape(lhs_dims, rhs_dims)
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"{name} cannot broadcast logical shapes "
                f"{self._shape_debug(lhs_dims)} and {self._shape_debug(rhs_dims)}"
            ) from exc
        if lhs_dims != output_dims:
            lhs = self.expand(lhs, output_dims)
        if rhs_dims != output_dims:
            rhs = self.expand(rhs, output_dims)
        physical_dims = self._physical_dims(output_dims)
        rank = self._lit_nat(len(physical_dims))
        shape = self.world.tuple(physical_dims)
        elem_type = self._tensor_element_type(lhs)
        if elem_type == self.I64:
            i64_comparisons = {
                "comparison.eq": "comparison.eq_i64", "comparison.ne": "comparison.ne_i64",
                "comparison.lt": "comparison.lt_i64", "comparison.le": "comparison.le_i64",
                "comparison.gt": "comparison.gt_i64", "comparison.ge": "comparison.ge_i64",
            }
            if name not in i64_comparisons:
                raise NotImplementedError(f"{name} is not implemented for int64 tensors")
            callee = self.world.annex(
                self._torch_annex_id(i64_comparisons[name])
            )
            callee = self._apply_grouped(callee, [rank, shape])
            result = self.world.app(callee, self.world.tuple([lhs, rhs]))
            return self._remember_shape(result, output_dims)
        callee = self.world.annex(self._torch_annex_id(name))
        callee = self.world.app(callee, self._torch_semantics(lhs))
        callee = self._apply_grouped(callee, [rank, shape])
        operands = [lhs, rhs]
        if alpha is not None:
            operands.append(self._float_lit(elem_type, alpha))
        result = self.world.app(callee, self.world.tuple(operands))
        return self._remember_shape(result, output_dims)

    def _torch_unary(self, name, input, *, floating=False, out_type=None):
        """Emit a Torch dialect unary op with the appropriate dictionary."""
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        rank = self._lit_nat(len(physical_dims))
        shape = self.world.tuple(physical_dims)
        callee = self.world.annex(self._torch_annex_id(name))
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=floating)
        )
        callee = self._apply_grouped(callee, [rank, shape])
        result = self.world.app(callee, input)
        return self._remember_shape(result, dims)

    def _torch_scalar(self, name, input, scalar, *, floating=False, alpha=None):
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(input)
        if elem_type == self.I64:
            if name not in (
                "binary.add_scalar",
                "binary.sub_scalar",
                "comparison.eq_i64_scalar",
                "comparison.ne_i64_scalar",
                "comparison.lt_i64_scalar",
                "comparison.le_i64_scalar",
                "comparison.gt_i64_scalar",
                "comparison.ge_i64_scalar",
            ):
                raise NotImplementedError(
                    f"{name} is not implemented for int64 tensors"
                )
            name = {
                "binary.add_scalar": "binary.add_i64_scalar",
                "binary.sub_scalar": "binary.sub_i64_scalar",
                "comparison.eq_i64_scalar": "comparison.eq_i64_scalar",
                "comparison.ne_i64_scalar": "comparison.ne_i64_scalar",
                "comparison.lt_i64_scalar": "comparison.lt_i64_scalar",
                "comparison.le_i64_scalar": "comparison.le_i64_scalar",
                "comparison.gt_i64_scalar": "comparison.gt_i64_scalar",
                "comparison.ge_i64_scalar": "comparison.ge_i64_scalar",
            }[name]
        callee = self.world.annex(self._torch_annex_id(name))
        if elem_type != self.I64:
            callee = self.world.app(
                callee, self._torch_semantics(input, floating=floating)
            )
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        if isinstance(scalar, mim.Def):
            scalar_def = scalar
        elif elem_type == self.I64:
            scalar_def = self.world.lit_i64(int(scalar))
        else:
            scalar_def = self._float_lit(elem_type, scalar)
        operands = [input, scalar_def]
        if alpha is not None:
            if elem_type == self.I64:
                if not isinstance(alpha, (int, bool)):
                    raise TypeError("integer add/sub alpha must be integral")
                alpha_def = self.world.lit_i64(int(alpha))
            else:
                alpha_def = self._float_lit(elem_type, alpha)
            operands.append(alpha_def)
        result = self.world.app(callee, self.world.tuple(operands))
        return self._remember_shape(result, dims)

    def _torch_scalar_lhs(self, name, scalar, tensor_value, *, alpha=1):
        dims = self.shape_of(tensor_value)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(tensor_value)
        callee = self.world.annex(self._torch_annex_id(name))
        callee = self.world.app(callee, self._torch_semantics(tensor_value))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple([
                self._float_lit(elem_type, scalar),
                tensor_value,
                self._float_lit(elem_type, alpha),
            ]),
        )
        return self._remember_shape(result, dims)

    def _broadcast_metadata(self, dims, output_dims):
        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and dims:
            physical_axes = [len(dims) - 1]
        physical_dims = [dims[axis] for axis in physical_axes]

        output_physical_axes = [
            axis for axis, extent in enumerate(output_dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not output_physical_axes and output_dims:
            output_physical_axes = [len(output_dims) - 1]
        output_rank = len(output_physical_axes)
        idx_type = self.world.type_idx(self._lit_nat(output_rank))
        offset = len(output_dims) - len(dims)
        mapped = []
        for axis in physical_axes:
            output_axis = offset + axis
            if output_axis in output_physical_axes:
                physical_output_axis = output_physical_axes.index(output_axis)
            elif self._is_lit_nat_value(dims[axis], 1) and output_rank:
                physical_output_axis = output_rank - 1
            else:
                raise ValueError("cannot represent broadcast after singleton folding")
            mapped.append(self.world.lit(idx_type, physical_output_axis))
        return physical_dims, self.world.tuple(mapped)

    def _torch_addc(self, name, self_tensor, tensor1, tensor2, *, value=1):
        tensors = (self_tensor, tensor1, tensor2)
        elem_types = [self._tensor_element_type(tensor) for tensor in tensors]
        if len(set(elem_types)) != 1:
            raise NotImplementedError(
                f"{name} mixed-dtype promotion is not implemented"
            )
        elem_type = elem_types[0]
        if elem_type not in (self.F32, self.F64):
            if name == "binary.addcdiv":
                raise TypeError("addcdiv does not support integer inputs")
            raise NotImplementedError(f"{name} dtype {elem_type} is not implemented")

        logical_shapes = [self.shape_of(tensor) for tensor in tensors]
        output_dims = logical_shapes[0]
        for dims in logical_shapes[1:]:
            output_dims = self.rules.broadcast_shape(output_dims, dims)
        output_physical_dims = self._physical_dims(output_dims)
        metadata = []
        ranks = []
        for dims in logical_shapes:
            physical_dims, axis_map = self._broadcast_metadata(dims, output_dims)
            ranks.append(self._lit_nat(len(physical_dims)))
            metadata.extend([self.world.tuple(physical_dims), axis_map])
        metadata.append(self.world.tuple(output_physical_dims))

        callee = self.world.annex(self._torch_annex_id(name))
        callee = self.world.app(callee, self._torch_semantics(self_tensor))
        callee = self._apply_grouped(
            callee, [*ranks, self._lit_nat(len(output_physical_dims))]
        )
        callee = self.world.app(callee, self.world.tuple(metadata))
        result = self.world.app(
            callee,
            self.world.tuple([
                self_tensor,
                tensor1,
                tensor2,
                self._float_lit(elem_type, value),
            ]),
        )
        return self._remember_shape(result, output_dims)

    def compare(self, op, lhs, rhs):
        return self.binary(op, lhs, rhs, out_type=self.Bool)

    def unary(self, op, input, out_type=None):
        """
        Translates to MimIR elementwise unary operation:
            %tensor.unary @(T_in, T_out) op @(rank, shape) input
        """
        in_type = self._tensor_element_type(input)
        if out_type is None:
            out_type = in_type
        dims = self._physical_dims(self.shape_of(input))
        rank, shape = self._lit_nat(len(dims)), self.world.tuple(dims)
        res = self._unary_with_types(in_type, out_type, op, input, rank, shape)
        return self._remember_shape(res, self.shape_of(input))

    def _unary_with_types(self, input_type, output_type, op, input, rank, shape):
        callee = self.world.annex(tensor.unary.value)
        callee = self._apply_grouped(callee, [input_type, output_type])
        callee = self.world.app(callee, op)
        callee = self._apply_grouped(callee, [rank, shape])
        return self.world.app(callee, input)

    def _f32_float_lit(self, value):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        return self.world.lit(self.F32, bits)

    def _float_lit(self, elem_type, value):
        if isinstance(value, mim.Def):
            return value
        if elem_type == self.F32:
            return self._f32_float_lit(value)
        if elem_type == self.F64:
            bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
            return self.world.lit(self.F64, bits)
        raise NotImplementedError(f"floating scalar for element type {elem_type}")

    def _f32_unary_lambda(self, callee, args_fn, ret_type=None):
        """
        Constructs a MimIR lambda (anonymous function) to map over elements.
        Example: `lam v: %_math_extrema.fmax (v, 0.0)`
        """
        if ret_type is None:
            ret_type = self.F32
        lam = self.world.mut_lam(self.F32, ret_type)
        v = lam.var()
        lam.app(True, callee, args_fn(v))
        return lam

    def _f32_pair_to_mean_lambda(self, pair_type):
        lam = self.world.mut_lam(pair_type, self.F32)
        pair = lam.var()
        lam.set_body(True, self.world.app(self.f32_div_axm, [pair.proj(2, 0), pair.proj(2, 1)]))
        return lam

    # Arithmetic
    def add(self, lhs, rhs, *, alpha=1):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar("binary.add_scalar", lhs, rhs, alpha=alpha)
        if isinstance(lhs, (int, float)):
            return self._torch_scalar_lhs(
                "binary.add_scalar_lhs", lhs, rhs, alpha=alpha
            )
        return self._torch_binary("binary.add", lhs, rhs, alpha=alpha)
    def sub(self, lhs, rhs, *, alpha=1):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar("binary.sub_scalar", lhs, rhs, alpha=alpha)
        if isinstance(lhs, (int, float)):
            return self._torch_scalar_lhs(
                "binary.sub_scalar_lhs", lhs, rhs, alpha=alpha
            )
        return self._torch_binary("binary.sub", lhs, rhs, alpha=alpha)
    def addcmul(self, self_tensor, tensor1, tensor2, *, value=1):
        return self._torch_addc(
            "binary.addcmul", self_tensor, tensor1, tensor2, value=value
        )
    def addcdiv(self, self_tensor, tensor1, tensor2, *, value=1):
        return self._torch_addc(
            "binary.addcdiv", self_tensor, tensor1, tensor2, value=value
        )
    def mul(self, lhs, rhs):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar("binary.mul_scalar", lhs, rhs)
        if isinstance(lhs, (int, float)):
            return self._torch_scalar("binary.mul_scalar", rhs, lhs)
        return self._torch_binary("binary.mul", lhs, rhs)
    def div(self, lhs, rhs): return self._torch_binary("binary.div", lhs, rhs)
    def pow(self, lhs, rhs):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar(
                "binary.pow_tensor_scalar", lhs, rhs, floating=True
            )
        return self.binary(self.f32_pow_axm, lhs, rhs)
    
    # Comparison
    def eq(self, lhs, rhs): return self._torch_binary("comparison.eq", lhs, rhs, out_type=self.Bool)
    def ne(self, lhs, rhs):
        if (
            isinstance(rhs, (int, float))
            and self._tensor_element_type(lhs) == self.I64
        ):
            return self._torch_scalar("comparison.ne_i64_scalar", lhs, rhs)
        return self._torch_binary("comparison.ne", lhs, rhs, out_type=self.Bool)
    def lt(self, lhs, rhs): return self._torch_binary("comparison.lt", lhs, rhs, out_type=self.Bool)
    def le(self, lhs, rhs): return self._torch_binary("comparison.le", lhs, rhs, out_type=self.Bool)
    def gt(self, lhs, rhs): return self._torch_binary("comparison.gt", lhs, rhs, out_type=self.Bool)
    def ge(self, lhs, rhs): return self._torch_binary("comparison.ge", lhs, rhs, out_type=self.Bool)

    # Extrema
    def maximum(self, lhs, rhs): return self._torch_binary("binary.maximum", lhs, rhs)
    def minimum(self, lhs, rhs): return self._torch_binary("binary.minimum", lhs, rhs)

    def _clamp_scalar_bounds(self, x, min_val, max_val):
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(x)

        def optional_bound(value):
            if value is None:
                return self.world.app(
                    self.world.annex(option.none.value), elem_type
                )
            return self.world.implicit_app(
                self.world.annex(option.some.value),
                self._float_lit(elem_type, value),
            )

        callee = self.world.annex(torch_dialect.activation.clamp.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple([
                x,
                optional_bound(min_val),
                optional_bound(max_val),
            ]),
        )
        return self._remember_shape(result, dims)
    
    def clamp_max(self, x, max_val):
        if isinstance(max_val, (int, float)):
            return self._clamp_scalar_bounds(x, None, max_val)
        return self.minimum(x, max_val)

    def clamp_min(self, x, min_val):
        if isinstance(min_val, (int, float)):
            return self._clamp_scalar_bounds(x, min_val, None)
        return self.maximum(x, min_val)

    def clamp(self, x, min_val=None, max_val=None):
        if all(
            value is None or isinstance(value, (int, float))
            for value in (min_val, max_val)
        ):
            return self._clamp_scalar_bounds(x, min_val, max_val)
        res = x
        if min_val is not None:
            res = self.clamp_min(res, min_val)
        if max_val is not None:
            res = self.clamp_max(res, max_val)
        return res

    # Unary
    def exp(self, x): return self._torch_unary("unary.exp", x, floating=True)
    def log(self, x):
        # A logical rank-0 Torch tensor is the scalar itself in MimIR's
        # physical representation.  Applying the rank-polymorphic tensor
        # schema would wrap that scalar in a non-assignable empty shape.
        if not self.shape_of(x) and not isinstance(x.type(), mim.Seq):
            if self._tensor_element_type(x) != self.F32:
                raise NotImplementedError("rank-0 log currently requires float32")
            return self._remember_shape(self.world.app(self.f32_log_axm, x), [])
        return self._torch_unary("unary.log", x, floating=True)
    def tanh(self, x): return self._torch_unary("activation.tanh", x, floating=True)
    def sqrt(self, x): return self._torch_unary("unary.sqrt", x, floating=True)
    def sin(self, x): return self._torch_unary("unary.sin", x, floating=True)
    def cos(self, x): return self._torch_unary("unary.cos", x, floating=True)
    def abs(self, x): return self._torch_unary("unary.abs", x)
    def neg(self, x): return self._torch_unary("unary.neg", x)
    def sigmoid(self, x): return self._torch_unary("activation.sigmoid", x, floating=True)
    def silu(self, x): return self._torch_unary("activation.silu", x, floating=True)
    def gelu(self, x, *, approximate="none"):
        """Map PyTorch GELU's string mode to a PE-visible Torch dialect flag."""
        if approximate not in ("none", "tanh"):
            raise ValueError(f"unsupported GELU approximation mode: {approximate!r}")
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        callee = self.world.annex(torch_dialect.activation.gelu.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple([self.world.lit_bool(approximate == "tanh"), x]),
        )
        return self._remember_shape(result, dims)

    def elu(self, x, *, alpha=1.0, scale=1.0, input_scale=1.0):
        """Map `aten.elu`; piecewise scalar semantics live in MimIR."""
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.activation.elu.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [
                    x,
                    self._float_lit(elem_type, alpha),
                    self._float_lit(elem_type, scale),
                    self._float_lit(elem_type, input_scale),
                ]
            ),
        )
        return self._remember_shape(result, dims)

    def selu(self, x):
        return self._torch_unary("activation.selu", x, floating=True)

    def hardsigmoid(self, x):
        return self._torch_unary("activation.hardsigmoid", x, floating=True)

    def softplus(self, x, *, beta=1.0, threshold=20.0):
        """Map `aten.softplus`; threshold selection remains PE-visible in MimIR."""
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.activation.softplus.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [
                    x,
                    self._float_lit(elem_type, beta),
                    self._float_lit(elem_type, threshold),
                ]
            ),
        )
        return self._remember_shape(result, dims)
    def mish(self, x):
        """Map Mish directly; softplus/tanh/multiply semantics live in MimIR."""
        return self._torch_unary("activation.mish", x, floating=True)
    def rsqrt(self, x): return self._torch_unary("unary.rsqrt", x, floating=True)
    
    def relu(self, x):
        return self._torch_unary("activation.relu", x)

    def leaky_relu(self, x, *, negative_slope=0.01):
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.activation.leaky_relu.value)
        callee = self.world.app(
            callee, self._torch_semantics(x, floating=True)
        )
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [x, self._float_lit(elem_type, negative_slope)]
            ),
        )
        return self._remember_shape(result, dims)

    def threshold(self, x, threshold, value):
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.activation.threshold.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple([
                x,
                self._float_lit(elem_type, threshold),
                self._float_lit(elem_type, value),
            ]),
        )
        return self._remember_shape(result, dims)

    def reciprocal(self, x):
        lam = self._f32_unary_lambda(
            self.f32_div_axm,
            lambda v: [self._f32_float_lit(1.0), v],
        )
        return self._torch_unary("unary.reciprocal", x, floating=True)

    def bitwise_and(self, lhs, rhs):
        return self.binary(self.bool_and_axm, lhs, rhs, out_type=self.Bool)

    def logical_not(self, x):
        return self.unary(self.bool_not_axm, x, out_type=self.Bool)

    def fma(self, a, b, c):
        return self.add(self.mul(a, b), c)

    def convert_element_type(self, x, dtype):
        import torch
        in_type = self._tensor_element_type(x)
        out_type = None
        if dtype in (torch.float32, torch.float):
            out_type = self.F32
        elif dtype == torch.bool:
            out_type = self.Bool
        else:
            raise NotImplementedError(f"Conversion to {dtype} is not implemented")
            
        if in_type == out_type:
            return x
            
        if in_type == self.Bool and out_type == self.F32:
            lam = self.world.mut_lam(self.Bool, self.F32)
            v = lam.var()
            callee = self.world.annex(core.select.value)
            callee = self.world.app(callee, self.F32)
            res = self._apply_grouped(callee, [v, self._f32_float_lit(1.0), self._f32_float_lit(0.0)])
            lam.set_body(True, res)
            return self.unary(lam, x, out_type=self.F32)

        if in_type == self.I64 and out_type == self.F32:
            lam = self.world.mut_lam(self.I64, self.F32)
            value = lam.var()
            callee = self.world.annex(_math_conv.s2f.value)
            callee = self.world.implicit_app(callee, self.f32_config)
            lam.set_body(True, self.world.app(callee, value))
            return self.unary(lam, x, out_type=self.F32)
            
        if in_type == self.F32 and out_type == self.Bool:
            lam = self._f32_unary_lambda(
                self.f32_ne_axm,
                lambda v: [v, self._f32_float_lit(0.0)],
                ret_type=self.Bool
            )
            return self.unary(lam, x, out_type=self.Bool)
            
        raise NotImplementedError(f"Conversion from {in_type} to {out_type} is not implemented")

    # Logical
    def where(self, cond, x, y):
        """
        Translates torch.where(cond, x, y) into MimIR's ternary selection:
            %tensor.select @T @(rank, shape) (cond, x, y)
        """
        cond_dims = self.shape_of(cond)
        x_dims = [] if isinstance(x, (int, float, bool)) else self.shape_of(x)
        y_dims = [] if isinstance(y, (int, float, bool)) else self.shape_of(y)
        output_dims = self.rules.broadcast_shape(self.rules.broadcast_shape(cond_dims, x_dims), y_dims)

        value = x if isinstance(x, mim.Def) else y
        if not isinstance(value, mim.Def):
            raise TypeError("torch.where requires at least one tensor value branch")
        elem_type = self._tensor_element_type(value)
        dtype = {
            self.F32: torch.float32,
            self.F64: torch.float64,
            self.I64: torch.int64,
            self.Bool: torch.bool,
        }.get(elem_type)
        if dtype is None:
            raise NotImplementedError(f"torch.where element type {elem_type}")
        if isinstance(x, (int, float, bool)):
            x = self.full(output_dims, x, dtype=dtype)
            x_dims = output_dims
        if isinstance(y, (int, float, bool)):
            y = self.full(output_dims, y, dtype=dtype)
            y_dims = output_dims

        if not self.rules.same_shape(cond_dims, output_dims):
            cond = self.expand(cond, output_dims)
        if not self.rules.same_shape(x_dims, output_dims):
            x = self.expand(x, output_dims)
        if not self.rules.same_shape(y_dims, output_dims):
            y = self.expand(y, output_dims)

        rank = self._lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.pointwise.where_.value)
        callee = self._apply_grouped(callee, [elem_type, rank, shape])
        result = self.world.app(callee, self.world.tuple([cond, x, y]))
        return self._remember_shape(result, output_dims)

    def bool_reduce_dims(self, input, dim, *, keepdim=False, reduce_all=False):
        input_dims = self.shape_of(input)
        dims = self.rules.normalize_reduce_dims(dim, len(input_dims))
        callee = self.world.annex(
            torch_dialect.reduction.all_dims.value
            if reduce_all
            else torch_dialect.reduction.any_dims.value
        )
        callee = self._apply_grouped(
            callee,
            [
                self._lit_nat(len(input_dims)),
                self._lit_nat(len(dims)),
                self.world.tuple(input_dims),
            ],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([self.world.lit_i64(axis) for axis in dims]),
        )
        result = self.world.app(callee, input)
        reduced_dims = self.rules.reduce_shape(input_dims, dims, False)
        self._remember_shape(result, reduced_dims)
        if keepdim:
            kept_dims = self.rules.reduce_shape(input_dims, dims, True)
            return self.reshape(result, kept_dims)
        return result

    def masked_fill_scalar(self, input, mask, value):
        input_dims = self.shape_of(input)
        mask_dims = self.shape_of(mask)
        physical_input_dims = self._physical_dims(input_dims)
        physical_mask_dims = self._physical_dims(mask_dims)
        rank = len(physical_input_dims)
        mask_rank = len(physical_mask_dims)
        if mask_rank > rank:
            raise ValueError("masked_fill mask rank exceeds input rank")

        elem_type = self._tensor_element_type(input)
        if isinstance(value, mim.Def):
            scalar = value
        elif elem_type in (self.F32, self.F64):
            scalar = self._float_lit(elem_type, value)
        elif elem_type == self.I64:
            scalar = self.world.lit_i64(int(value))
        elif elem_type == self.Bool:
            scalar = self.world.lit_tt() if value else self.world.lit_ff()
        else:
            raise NotImplementedError(
                f"masked_fill scalar for element type {elem_type}"
            )

        rank_def = self._lit_nat(rank)
        mask_rank_def = self._lit_nat(mask_rank)
        idx_t = self.world.type_idx(rank_def)
        mask_to_input = self.world.tuple(
            [
                self.world.lit(idx_t, rank - mask_rank + axis)
                for axis in range(mask_rank)
            ]
        )
        callee = self.world.annex(
            self._torch_annex_id("pointwise.masked_fill_scalar")
        )
        callee = self._apply_grouped(
            callee, [elem_type, mask_rank_def, rank_def]
        )
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple(physical_mask_dims),
                self.world.tuple(physical_input_dims),
                mask_to_input,
            ],
        )
        result = self.world.app(
            callee, self.world.tuple([input, mask, scalar])
        )
        return self._remember_shape(result, input_dims)

        tensor_type = self._tensor_element_type(x)
        rank = self._lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        callee = self.world.annex(tensor.select.value)
        callee = self.world.app(callee, tensor_type)
        callee = self._apply_grouped(callee, [rank, shape])
        result = self._apply_grouped(callee, [cond, x, y])
        return self._remember_shape(result, output_dims)

    def _extract_shape(self, shape_arg):
        out_shape_list = []
        for d in shape_arg:
            if isinstance(d, int):
                out_shape_list.append(self._lit_nat(d))
            elif isinstance(d, mim.Def):
                out_shape_list.append(d)
            else:
                raise ValueError(f"Unsupported shape dimension type: {type(d)}")
        return self.world.tuple(out_shape_list), len(shape_arg)

    def expand(self, input, shape):
        """
        Translates torch.expand to %tensor.broadcast_in_dim or %tensor.broadcast.
        """
        in_dims = self.shape_of(input)
        # Tensor values use nested arrays, where literal singleton axes are
        # erased from the physical type.  Keep the frontend's logical shape
        # for broadcast semantics, but pass the folded physical shape/rank to
        # the tensor dialect together with the original logical output-axis
        # mapping.
        physical_axes = [i for i, dim in enumerate(in_dims) if not self._is_lit_nat_value(dim, 1)]
        physical_dims = [in_dims[i] for i in physical_axes]
        in_rank_val = len(physical_dims)

        shape = list(shape)
        out_rank_val = len(shape)
        rank_offset = out_rank_val - in_rank_val
        logical_rank_offset = out_rank_val - len(in_dims)
        resolved_shape = []
        for i, dim in enumerate(shape):
            if not (isinstance(dim, int) and dim == -1):
                resolved_shape.append(dim)
                continue
            input_index = i - logical_rank_offset
            if input_index < 0 or input_index >= len(in_dims):
                raise ValueError(
                    f"cannot infer expand dimension {i} from input rank {len(in_dims)}"
                )
            resolved_shape.append(in_dims[input_index])

        shape = resolved_shape
        out_shape_tuple, _ = self._extract_shape(shape)
        out_rank = self._lit_nat(len(shape))
        in_rank = self._lit_nat(in_rank_val)
        in_shape_tuple = self.world.tuple(physical_dims)
        elem_type = self._tensor_element_type(input)

        # A scalar tensor has no index tuple.  Keep the established tensor
        # map path for this case; `%torch.shape.expand` requires an array input.
        if in_rank_val == 0:
            if input.type() == elem_type:
                scalar = input
            else:
                getter = self.world.annex(tensor.get.value)
                getter = self._apply_grouped(
                    getter, [elem_type, self._lit_nat(0), self.world.tuple([])]
                )
                scalar = self.world.app(
                    getter, self.world.tuple([self.world.tuple([]), input])
                )
            callee = self.world.annex(torch_dialect.creation.full.value)
            callee = self._apply_grouped(callee, [elem_type, out_rank])
            result = self.world.app(callee, [out_shape_tuple, scalar])
            return self._remember_shape(result, shape)

        if in_dims == shape:
            return input

        idx_t = self.world.type_idx(out_rank)
        # Physical input axes are right-aligned to the logical output shape.
        # This matters for e.g. [1, hidden] -> [batch, sequence, hidden].
        mapped_axes = [logical_rank_offset + i for i in physical_axes]
        if any(axis < 0 or axis >= out_rank_val for axis in mapped_axes):
            raise ValueError(
                f"broadcast axis mismatch: input={len(in_dims)}:{physical_axes}, "
                f"output={len(shape)}, offset={logical_rank_offset}"
            )
        index_tuple = self.world.tuple([self.world.lit(idx_t, i) for i in mapped_axes])
        callee = self.world.annex(torch_dialect.shape.expand.value)
        callee = self._apply_grouped(callee, [elem_type, in_rank, out_rank])
        callee = self.world.app(callee, [in_shape_tuple, out_shape_tuple])
        result = self.world.app(callee, [input, index_tuple])
        return self._remember_shape(result, shape)

        # Legacy tensor implementation retained below for comparison.
        out_shape_tuple, _ = self._extract_shape(shape)
        out_rank = self._lit_nat(out_rank_val)
        
        if in_dims == shape:
            return input

        elem_type = self._tensor_element_type(input)
        _, in_shape_tuple = self._rank_and_shape(input)

        if in_rank_val == 0:
            callee = self.world.annex(tensor.map.value)
            callee = self._apply_grouped(callee, [elem_type, self._lit_nat(0), self.world.tuple([])])
            lam = self.world.mut_lam(self.world.sigma([]), elem_type)
            lam.set_body(True, input)
            callee = self.world.app(callee, lam)
            callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
            result = self.world.app(callee, self.world.tuple([]))
            return self._remember_shape(result, shape)

        if in_rank_val == out_rank_val:
            callee = self.world.annex(tensor.broadcast.value)
            callee = self._apply_grouped(callee, [elem_type, out_rank])
            result = self.world.app(callee, self.world.tuple([in_shape_tuple, out_shape_tuple, input]))
            return self._remember_shape(result, shape)
        else:
            callee = self.world.annex(tensor.broadcast_in_dim.value)
            callee = self._apply_grouped(callee, [elem_type, self._lit_nat(in_rank_val), out_rank])
            idx_t = self.world.type_idx(out_rank)
            offset = out_rank_val - in_rank_val
            index_mapping = [self.world.lit(idx_t, offset + i) for i in range(in_rank_val)]
            index_tuple = self.world.tuple(index_mapping)
            result = self.world.app(callee, self.world.tuple([in_shape_tuple, out_shape_tuple, input, index_tuple]))
            return self._remember_shape(result, shape)

    def full(self, shape, fill_value, dtype=None):
        """
        Translates torch.full to a 0-input map (%tensor.map with ni=0).
        """
        import torch
        if dtype is None:
            dtype = torch.float32
            
        if dtype in (torch.float32, torch.float, None):
            elem_type = self.F32
            scalar_def = self._f32_float_lit(float(fill_value))
        elif dtype == torch.bool:
            elem_type = self.Bool
            scalar_def = self.world.lit_tt() if fill_value else self.world.lit_ff()
        elif dtype in (torch.int64, torch.long):
            elem_type = self.I64
            scalar_def = self.world.lit_i64(int(fill_value))
        else:
            raise NotImplementedError(f"full with dtype {dtype} is not implemented")

        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        if out_rank_val == 0:
            return self._remember_shape(scalar_def, [])
        callee = self.world.annex(torch_dialect.creation.full.value)
        callee = self._apply_grouped(callee, [elem_type, self._lit_nat(out_rank_val)])
        callee = self.world.app(callee, [out_shape_tuple, scalar_def])
        return self._remember_shape(callee, shape)

    def zeros_like(self, reference, dtype):
        result = self.full(self.shape_of(reference), 0, dtype=dtype)
        self._known_zero_tensors.add(result)
        return result
            
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self._lit_nat(out_rank_val)
        
        callee = self.world.annex(tensor.map.value)
        ni = self._lit_nat(0)
        Is = self.world.tuple([])
        callee = self.world.app(callee, self.world.tuple([elem_type, ni, Is]))
        
        lam = self.world.mut_lam(self.world.sigma([]), elem_type)
        lam.set_body(True, scalar_def)
        
        callee = self.world.app(callee, lam)
        callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
        
        input_is = self.world.tuple([])
        result = self.world.app(callee, input_is)
        return self._remember_shape(result, shape)

    def empty_strided(
        self, shape, stride, dtype=None, layout=None, device=None, pin_memory=None
    ):
        if dtype not in (None, torch.float32, torch.float):
            raise NotImplementedError(f"empty_strided with dtype {dtype} is not implemented")
        if layout not in (None, torch.strided):
            raise NotImplementedError("empty_strided currently supports strided layout only")
        if device is not None and torch.device(device).type != "cpu":
            raise NotImplementedError("empty_strided currently supports CPU tensors only")
        if pin_memory not in (None, False):
            raise NotImplementedError("empty_strided pin_memory=True is not implemented")

        expected_stride = []
        running = 1
        for dim in reversed(shape):
            expected_stride.append(running)
            if not isinstance(dim, int):
                raise NotImplementedError("dynamic non-contiguous strides are not represented yet")
            running *= dim
        expected_stride.reverse()
        if list(stride) != expected_stride:
            raise NotImplementedError(
                "empty_strided currently supports contiguous row-major strides only"
            )

        shape_tuple, rank_value = self._extract_shape(shape)
        stride_tuple, _ = self._extract_shape(stride)
        callee = self.world.annex(torch_dialect.creation.empty_strided.value)
        callee = self._apply_grouped(callee, [self.F32, self._lit_nat(rank_value)])
        result = self.world.app(callee, self.world.tuple([shape_tuple, stride_tuple]))
        return self._remember_shape(result, list(shape))

    def scalar_tensor(
        self, value, dtype=None, layout=None, device=None, pin_memory=None
    ):
        if layout not in (None, torch.strided):
            raise NotImplementedError("scalar_tensor requires strided layout")
        if device is not None and torch.device(device).type != "cpu":
            raise NotImplementedError("scalar_tensor currently supports CPU only")
        if pin_memory not in (None, False):
            raise NotImplementedError("scalar_tensor pin_memory=True is unsupported")

        if dtype in (None, torch.float32, torch.float):
            elem_type = self.F32
            scalar = self._f32_float_lit(float(value))
        elif dtype in (torch.int64, torch.long):
            elem_type = self.I64
            scalar = self.world.lit_i64(int(value))
        elif dtype == torch.bool:
            elem_type = self.Bool
            scalar = self.world.lit_tt() if value else self.world.lit_ff()
        else:
            raise NotImplementedError(
                f"scalar_tensor with dtype {dtype} is not implemented"
            )

        callee = self.world.annex(torch_dialect.creation.scalar_tensor.value)
        result = self.world.app(self.world.app(callee, elem_type), scalar)
        return self._remember_shape(result, [])

    def scalar_value(self, value):
        """Extract a logical rank-0 tensor at the native function boundary."""
        if self.shape_of(value):
            return value
        value_type = value.type()
        if not isinstance(value_type, mim.Seq):
            return value
        elem_type = self._tensor_element_type(value)
        getter = self.world.annex(tensor.get.value)
        getter = self._apply_grouped(
            getter, [elem_type, self._lit_nat(0), self.world.tuple([])]
        )
        return self.world.app(
            getter, self.world.tuple([self.world.tuple([]), value])
        )

    def fill_scalar(self, input, value):
        dims = self.shape_of(input)
        elem_type = self._tensor_element_type(input)
        if elem_type == self.F32:
            scalar = self._f32_float_lit(float(value))
        elif elem_type == self.Bool:
            scalar = self.world.lit_tt() if value else self.world.lit_ff()
        else:
            raise NotImplementedError("fill.Scalar currently supports float32 and bool tensors")
        callee = self.world.annex(torch_dialect.creation.fill_scalar.value)
        callee = self._apply_grouped(
            callee, [elem_type, self._lit_nat(len(dims)), self.world.tuple(dims)]
        )
        result = self.world.app(callee, self.world.tuple([input, scalar]))
        return self._remember_shape(result, dims)

    def arange(self, start, end=None, step=1, dtype=None, layout=None, device=None, pin_memory=None):
        if end is None:
            start, end = 0, start
        if not all(isinstance(value, int) for value in (start, end, step)):
            raise NotImplementedError("arange currently requires integer scalar bounds")
        if step == 0:
            raise ValueError("arange step must be nonzero")
        if dtype not in (None, torch.int64, torch.long):
            raise NotImplementedError("arange currently supports the integer-default/int64 dtype")
        if layout not in (None, torch.strided):
            raise NotImplementedError("arange currently supports strided layout only")
        if device is not None and torch.device(device).type != "cpu":
            raise NotImplementedError("arange currently supports CPU tensors only")
        if pin_memory not in (None, False):
            raise NotImplementedError("arange pin_memory=True is not implemented")

        size = len(range(start, end, step))
        callee = self.world.annex(torch_dialect.creation.arange_i64.value)
        callee = self.world.app(callee, self._lit_nat(size))
        result = self.world.app(
            callee,
            self.world.tuple(
                [
                    self.world.lit(self.I64, start),
                    self.world.lit(self.I64, end),
                    self.world.lit(self.I64, step),
                ]
            ),
        )
        return self._remember_shape(result, [self._lit_nat(size)])

    def _reduce_aff(self, input, output_type, reducer, init, dim=None, keepdim=False, return_shape=False):
        """
        Translates reduction operations into `%tensor.map_reduce`.
        Uses ShapeRules.reduce_shape_spec as the canonical source for reduce shape invariants.
        """
        input_dims = self.shape_of(input)
        input_rank = len(input_dims)
        spec = self.rules.reduce_shape_spec(input_dims, dim=dim, keepdim=keepdim)

        output_rank = len(spec.output_dims)
        reduce_rank = len(spec.reduce_dims)
        total_rank = output_rank + reduce_rank

        callee = self.world.annex(tensor.map_reduce.value)
        callee = self.world.app(callee, self._lit_nat(1)) # nis = 1 input tensor
        callee = self._apply_grouped(callee, [output_type, self._lit_nat(output_rank), self._lit_nat(reduce_rank)])
        callee = self._apply_grouped(callee, [self.world.tuple(spec.output_dims), self.world.tuple(spec.loop_dims)])
        
        in_elem_type = self._tensor_element_type(input)
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([in_elem_type]),
                self.world.tuple([self._lit_nat(input_rank)]),
                self.world.tuple([self.world.tuple(input_dims)]),
            ],
        )
        callee = self._apply_grouped(callee, [reducer, init])
        callee = self.world.app(
            callee,
            self._affine_projection_lam(total_rank, output_rank, list(range(output_rank))),
        )
        callee = self.world.app(
            callee,
            self.world.tuple(
                [self._affine_projection_lam(total_rank, input_rank, spec.input_projections)]
            ),
        )
        result = self.world.app(callee, self.world.tuple([input]))
        
        # 2. Record the truth of the resulting shape
        self._remember_shape(result, spec.output_dims)
        
        if return_shape:
            return result, spec.output_dims
        return result

    def sum(self, input, dim=None, keepdim=False):
        """
        Translates to the Torch reduction schema; its implementation lowers
        to tensor.map_reduce after dimension validation and PE.
        """
        return self._torch_reduce("sum", input, dim, keepdim)

    def amax(self, input, dim=None, keepdim=False):
        """
        Translates to maximum reduction via `%tensor.map_reduce`.
        """
        if keepdim:
            return self._reduce_aff(input, self.F32, self._f32_reduce_lambda(self.f32_max_axm), self._f32_float_lit(-float("inf")), dim=dim, keepdim=True)
        return self._torch_reduce("amax", input, dim, keepdim)

    def norm(self, input, p="fro", dim=None, keepdim=False, dtype=None):
        """Map the L2/Frobenius cases of `torch.norm` to MimIR semantics."""
        if p not in (2, 2.0, "fro"):
            raise NotImplementedError(f"torch.norm p={p!r} is not implemented")
        if dtype is not None:
            raise NotImplementedError("torch.norm dtype conversion is not implemented")
        logical_dims = self.shape_of(input)
        logical_rank = len(logical_dims)
        dimensions = (
            list(range(logical_rank))
            if dim is None
            else list(dim) if isinstance(dim, (tuple, list)) else [dim]
        )
        canonical = [d + logical_rank if d < 0 else d for d in dimensions]
        if (
            not canonical
            or any(d < 0 or d >= logical_rank for d in canonical)
            or len(set(canonical)) != len(canonical)
        ):
            raise ValueError("norm dimensions must be non-empty, unique, and in range")
        output_dims = self.rules.reduce_shape_spec(
            logical_dims, dim=canonical, keepdim=keepdim
        ).output_dims
        physical_axes = [
            axis for axis, extent in enumerate(logical_dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [logical_dims[axis] for axis in physical_axes]
        reduced_axes = [
            physical_axes.index(axis) for axis in canonical if axis in physical_axes
        ]
        if not reduced_axes:
            return self._remember_shape(self.abs(input), output_dims)

        if not keepdim and len(reduced_axes) == len(physical_dims):
            callee = self.world.annex(torch_dialect.reduction.norm2_all.value)
            callee = self.world.app(
                callee, self._torch_semantics(input, floating=True)
            )
            callee = self._apply_grouped(
                callee,
                [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
            )
            result = self.world.app(callee, input)
            return self._remember_shape(result, [])

        op = (
            torch_dialect.reduction.norm2_dims_keepdim
            if keepdim
            else torch_dialect.reduction.norm2_dims
        )
        callee = self.world.annex(op.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee,
            [
                self._lit_nat(len(physical_dims)),
                self._lit_nat(len(reduced_axes)),
                self.world.tuple(physical_dims),
            ],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([self.world.lit_i64(axis) for axis in reduced_axes]),
        )
        result = self.world.app(callee, input)
        return self._remember_shape(result, output_dims)

    def normalize(self, input, p=2.0, dim=1, eps=1e-12):
        """Map ``torch.nn.functional.normalize`` through existing reductions.

        PyTorch computes ``input / max(vector_norm(input, p, dim, keepdim=True),
        eps)``.  Keeping the denominator's reduced axes makes the ordinary
        Torch binary broadcasting path express the API semantics without a
        new tensor primitive.
        """
        if p not in (2, 2.0):
            raise NotImplementedError(
                f"functional.normalize p={p!r} is not implemented"
            )
        if not isinstance(eps, (int, float)) or eps < 0:
            raise NotImplementedError(
                "functional.normalize requires a non-negative static eps"
            )
        denominator = self.norm(input, p=2, dim=dim, keepdim=True)
        denominator = self.clamp_min(denominator, float(eps))
        return self.div(input, denominator)

    def dim_extrema(self, input, dim, keepdim=False, *, kind="max"):
        """Map value+index max/min; tie and NaN semantics live in MimIR."""
        dims = self.shape_of(input)
        rank = len(dims)
        canonical = dim + rank if dim < 0 else dim
        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and dims:
            physical_axes = [rank - 1]
        physical_dims = self._physical_dims(dims)
        reduced_dims = dims[:canonical] + dims[canonical + 1:]

        if 0 <= canonical < rank and canonical not in physical_axes:
            callee = self.world.annex(torch_dialect.creation.full.value)
            callee = self._apply_grouped(
                callee, [self.I64, self._lit_nat(len(physical_dims))]
            )
            indices = self.world.app(
                callee,
                self.world.tuple(
                    [self.world.tuple(physical_dims), self.world.lit_i64(0)]
                ),
            )
            result_dims = list(dims) if keepdim else reduced_dims
            return (
                self._remember_shape(input, result_dims),
                self._remember_shape(indices, result_dims),
            )

        physical_dim = (
            physical_axes.index(canonical)
            if 0 <= canonical < rank and canonical in physical_axes
            else len(physical_dims)
        )
        callee = self.world.annex(
            torch_dialect.reduction.max_dim.value
            if kind == "max"
            else torch_dialect.reduction.min_dim.value
        )
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        callee = self.world.app(callee, self.world.lit_i64(physical_dim))
        result = self.world.app(callee, input)
        outputs = [result.proj(3, 1), result.proj(3, 2)]
        for output in outputs:
            self._remember_shape(output, reduced_dims)
        if keepdim:
            keep_dims = list(dims)
            keep_dims[canonical] = self._lit_nat(1)
            outputs = [self.reshape(output, keep_dims) for output in outputs]
        return tuple(outputs)

    def _softmax(self, input, dim, op, singleton_value):
        """Emit common softmax semantics while preserving logical singleton axes."""
        dims = self.shape_of(input)
        logical_rank = len(dims)
        canonical_dim = dim + logical_rank if dim < 0 else dim
        valid_logical_dim = 0 <= canonical_dim < logical_rank

        physical_axes = [
            axis
            for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and dims:
            physical_axes = [logical_rank - 1]
        physical_dims = self._physical_dims(dims)

        # Reduction over a folded singleton is known statically: softmax is
        # one and log_softmax is zero, so PE can eliminate the full operator.
        if valid_logical_dim and canonical_dim not in physical_axes:
            elem_type = self._tensor_element_type(input)
            callee = self.world.annex(torch_dialect.creation.full.value)
            callee = self._apply_grouped(
                callee, [elem_type, self._lit_nat(len(physical_dims))]
            )
            result = self.world.app(
                callee,
                self.world.tuple(
                    [
                        self.world.tuple(physical_dims),
                        self._float_lit(elem_type, singleton_value),
                    ]
                ),
            )
            return self._remember_shape(result, dims)

        # Preserve invalid dimensions as invalid MimIR inputs. The Torch
        # plugin's precondition then owns static diagnostics/runtime checks.
        physical_dim = (
            physical_axes.index(canonical_dim)
            if valid_logical_dim
            else len(physical_dims)
        )
        callee = self.world.annex(op.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        callee = self.world.app(
            callee,
            self.world.tuple(
                [self.world.lit_i64(physical_dim), self.world.lit_ff()]
            ),
        )
        result = self.world.app(callee, input)
        return self._remember_shape(result, dims)

    def softmax(self, input, dim=-1):
        """Emit API-level softmax; stabilization and reduction live in MimIR."""
        return self._softmax(input, dim, torch_dialect.normalization.softmax, 1.0)

    def log_softmax(self, input, dim=-1):
        """Emit API-level log_softmax; stabilization and reduction live in MimIR."""
        return self._softmax(input, dim, torch_dialect.normalization.log_softmax, 0.0)

    def flip(self, input, dims):
        """Map logical Torch axes to the singleton-folded physical tensor rank."""
        logical_shape = self.shape_of(input)
        logical_rank = len(logical_shape)
        dims = [dims] if isinstance(dims, int) else list(dims)
        canonical = [dim + logical_rank if dim < 0 else dim for dim in dims]
        valid = (
            all(0 <= dim < logical_rank for dim in canonical)
            and len(set(canonical)) == len(canonical)
        )
        if valid and len(canonical) == 1:
            cumsum = self._cumsum_provenance.get(input)
            if cumsum is not None:
                cumsum_input, cumsum_dim, dtype = cumsum
                inner_flip = self._flip_provenance.get(cumsum_input)
                if (
                    cumsum_dim == canonical[0]
                    and inner_flip is not None
                    and inner_flip[1] == tuple(canonical)
                ):
                    return self.cumsum(
                        inner_flip[0], cumsum_dim, dtype=dtype, reverse=True
                    )
        physical_axes = [
            axis for axis, extent in enumerate(logical_shape)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_shape = self._physical_dims(logical_shape)
        if valid:
            physical_dims = [
                physical_axes.index(dim)
                for dim in canonical
                if dim in physical_axes
            ]
            if not physical_dims:
                return input
        else:
            # Keep malformed axes malformed so `%runtime.require` owns the
            # diagnostic instead of silently accepting them in the frontend.
            physical_dims = [len(physical_shape)] * max(1, len(dims))

        elem_type = self._tensor_element_type(input)
        callee = self.world.annex(torch_dialect.indexing.flip.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_type,
                self._lit_nat(len(physical_shape)),
                self._lit_nat(len(physical_dims)),
                self.world.tuple(physical_shape),
            ],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([self.world.lit_i64(dim) for dim in physical_dims]),
        )
        result = self.world.app(callee, input)
        result = self._remember_shape(result, logical_shape)
        if valid:
            self._flip_provenance[result] = (input, tuple(canonical))
        return result

    def narrow(self, input, dim, start, length):
        """Lower Torch narrow while retaining its checked signed index semantics."""
        logical_shape = self.shape_of(input)
        logical_rank = len(logical_shape)
        if not isinstance(dim, int):
            raise NotImplementedError("dynamic narrow dim is not implemented")
        canonical = dim + logical_rank if dim < 0 else dim
        physical_axes = [
            axis for axis, extent in enumerate(logical_shape)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_shape = self._physical_dims(logical_shape)
        if 0 <= canonical < logical_rank and canonical in physical_axes:
            physical_dim = physical_axes.index(canonical)
        else:
            physical_dim = len(physical_shape)

        elem_type = self._tensor_element_type(input)
        callee = self.world.annex(torch_dialect.indexing.narrow.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_type,
                self._lit_nat(len(physical_shape)),
                self.world.tuple(physical_shape),
            ],
        )
        provenance_start = start
        provenance_length = length
        start = self.world.lit_i64(start) if isinstance(start, int) else start
        length = self._to_nat(length)
        callee = self.world.app(
            callee,
            self.world.tuple(
                [self.world.lit_i64(physical_dim), start, length]
            ),
        )
        result = self.world.app(callee, input)
        output_shape = list(logical_shape)
        if 0 <= canonical < logical_rank:
            output_shape[canonical] = self._to_nat(length)
        result = self._remember_shape(result, output_shape)
        if 0 <= canonical < logical_rank:
            self._narrow_provenance[result] = (
                input, canonical, provenance_start, provenance_length
            )
        return result

    def _triangular(self, input, diagonal, op, name):
        """Instantiate a Torch triangular axiom with a dtype-correct zero."""
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(input)
        if elem_type == self.F32:
            zero = self._f32_float_lit(0.0)
        elif elem_type == self.I64:
            zero = self.world.lit(self.I64, 0)
        elif elem_type == self.Bool:
            zero = self.world.lit_ff()
        else:
            raise NotImplementedError(
                f"{name} with element type {elem_type} is not implemented"
            )
        if isinstance(diagonal, int):
            diagonal = self.world.lit_i64(diagonal)

        callee = self.world.annex(op.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_type,
                self._lit_nat(len(physical_dims)),
                self.world.tuple(physical_dims),
            ],
        )
        callee = self.world.app(callee, self.world.tuple([zero, diagonal]))
        result = self.world.app(callee, input)
        return self._remember_shape(result, dims)

    def triu(self, input, diagonal=0):
        return self._triangular(
            input, diagonal, torch_dialect.linalg.triu, "triu"
        )

    def tril(self, input, diagonal=0):
        return self._triangular(
            input, diagonal, torch_dialect.linalg.tril, "tril"
        )

    def native_layer_norm(self, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Map functional/ATen LayerNorm to `%torch.normalization.native_layer_norm`."""
        input_dims = self.shape_of(input)
        normalized_shape = tuple(int(d) for d in normalized_shape)
        rn = len(normalized_shape)
        if rn == 0 or rn > len(input_dims):
            raise ValueError("normalized_shape must be a non-empty suffix of input shape")
        rb = len(input_dims) - rn
        normalized_dims = input_dims[rb:]
        if any(
            not self.rules._same_dim(actual, self._lit_nat(expected))
            for actual, expected in zip(normalized_dims, normalized_shape)
        ):
            raise ValueError("normalized_shape must match the input suffix")

        elem_t = self._tensor_element_type(input)
        normalized_type = self.world.arr(
            self.world.tuple(normalized_dims), elem_t
        )

        def optional(value):
            if value is None:
                return self.world.app(
                    self.world.annex(option.none.value), normalized_type
                )
            value_dims = self.shape_of(value)
            if not self.rules.same_shape(value_dims, normalized_dims):
                raise ValueError("LayerNorm weight and bias must match normalized_shape")
            return self.world.implicit_app(
                self.world.annex(option.some.value), value
            )

        callee = self.world.annex(torch_dialect.normalization.native_layer_norm.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee, [self._lit_nat(rb), self._lit_nat(rn)]
        )
        callee = self._apply_grouped(
            callee,
            [self.world.tuple(input_dims[:rb]), self.world.tuple(normalized_dims)],
        )
        callee = self.world.app(callee, input)
        callee = self._apply_grouped(callee, [optional(weight), optional(bias)])
        result = self.world.app(callee, self._f32_float_lit(eps))

        stat_dims = input_dims[:rb] + [self._lit_nat(1)] * rn
        self._remember_shape(result.proj(3, 0), input_dims)
        self._remember_shape(result.proj(3, 1), stat_dims)
        self._remember_shape(result.proj(3, 2), stat_dims)
        return result

    def group_norm(self, input, groups, weight=None, bias=None, eps=1e-5):
        """Map GroupNorm/InstanceNorm; statistics and affine semantics live in MimIR."""
        dims = self.shape_of(input)
        elem_type = self._tensor_element_type(input)
        channels = dims[1]
        channel_type = self.world.arr(channels, elem_type)

        def optional(value):
            if value is None:
                return self.world.app(
                    self.world.annex(option.none.value), channel_type
                )
            return self.world.implicit_app(
                self.world.annex(option.some.value), value
            )

        callee = self.world.annex(torch_dialect.normalization.group_norm.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(dims)), self.world.tuple(dims)]
        )
        groups_def = groups if isinstance(groups, mim.Def) else self._lit_nat(groups)
        result = self.world.app(
            callee,
            self.world.tuple(
                [input, optional(weight), optional(bias), groups_def,
                 self._f32_float_lit(eps)]
            ),
        )
        return self._remember_shape(result, dims)

    def native_group_norm(
        self, input, weight, bias, n, channels, inner, groups, eps
    ):
        """Map the full Aten `(output, mean, rstd)` group-normalization schema."""
        dims = self.shape_of(input)
        elem_type = self._tensor_element_type(input)
        channel_type = self.world.arr(dims[1], elem_type)

        def optional(value):
            if value is None:
                return self.world.app(
                    self.world.annex(option.none.value), channel_type
                )
            return self.world.implicit_app(
                self.world.annex(option.some.value), value
            )

        def nat(value):
            if isinstance(value, mim.Def):
                return value
            if isinstance(value, int):
                return self._lit_nat(value)
            raise NotImplementedError(
                "native_group_norm currently requires static integer shape arguments"
            )

        n_def = nat(n)
        groups_def = nat(groups)
        callee = self.world.annex(
            torch_dialect.normalization.native_group_norm.value
        )
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=True)
        )
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(dims)), self.world.tuple(dims)]
        )
        callee = self.world.app(
            callee, self.world.tuple([input, optional(weight), optional(bias)])
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [
                    n_def,
                    nat(channels),
                    nat(inner),
                    groups_def,
                    self._float_lit(elem_type, eps),
                ]
            ),
        )
        stat_dims = [n_def, groups_def]
        self._remember_shape(result.proj(3, 0), dims)
        self._remember_shape(result.proj(3, 1), stat_dims)
        self._remember_shape(result.proj(3, 2), stat_dims)
        return result

    def smooth_l1_mean(self, input, target, beta=1.0):
        """Map mean-reduced SmoothL1; piecewise and reduction semantics live in MimIR."""
        dims = self.shape_of(input)
        callee = self.world.annex(torch_dialect.loss.smooth_l1_mean.value)
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=True)
        )
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(dims)), self.world.tuple(dims)]
        )
        callee = self.world.app(callee, self.world.tuple([input, target]))
        result = self.world.app(callee, self._f32_float_lit(beta))
        return self._remember_shape(result, [])

    def cross_entropy_mean_2d(self, input, target):
        """Map default rank-2 cross entropy; its formula lives in MimIR."""
        input_dims = self.shape_of(input)
        target_dims = self.shape_of(target)
        if len(input_dims) != 2 or len(target_dims) != 1:
            raise NotImplementedError(
                "cross_entropy currently requires (N, C) logits and (N) target"
            )
        if not self.rules.same_shape([input_dims[0]], target_dims):
            raise ValueError("cross_entropy target batch must match input batch")
        if self._tensor_element_type(target) != self.I64:
            raise TypeError("cross_entropy class target must have dtype int64")
        if len(self._physical_dims(input_dims)) != 2 or len(
            self._physical_dims(target_dims)
        ) != 1:
            raise NotImplementedError(
                "cross_entropy with folded singleton batch/class axes is not implemented"
            )

        callee = self.world.annex(
            torch_dialect.loss.cross_entropy_mean_2d.value
        )
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=True)
        )
        callee = self._apply_grouped(callee, input_dims)
        result = self.world.app(callee, self.world.tuple([input, target]))
        return self._remember_shape(result, [])

    def kl_div_reduced(self, input, target, reduction, log_target=False):
        """Map reduced KLDiv; pointwise and reduction branches live in MimIR."""
        dims = self.shape_of(input)
        target_dims = self.shape_of(target)
        if not self.rules.same_shape(dims, target_dims):
            raise ValueError("kl_div input and target must have the same shape")
        if not dims:
            raise NotImplementedError("reduced scalar kl_div is not implemented")
        reduction_tags = {"sum": 0, "mean": 1, "batchmean": 2}
        if reduction not in reduction_tags:
            raise NotImplementedError(
                "kl_div currently supports sum, mean, and batchmean reduction"
            )

        physical_dims = self._physical_dims(dims)
        callee = self.world.annex(torch_dialect.loss.kl_div_reduced.value)
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=True)
        )
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        callee = self.world.app(
            callee,
            self.world.tuple(
                [
                    self.world.lit_bool(log_target),
                    self._lit_nat(reduction_tags[reduction]),
                ]
            ),
        )
        result = self.world.app(callee, self.world.tuple([input, target]))
        return self._remember_shape(result, [])

    def triplet_margin_reduced(
        self,
        anchor,
        positive,
        negative,
        margin=1.0,
        p=2.0,
        eps=1e-6,
        swap=False,
        reduction="mean",
    ):
        """Map rank-2 TripletMarginLoss; all formulas and branches stay in MimIR."""
        dims = self.shape_of(anchor)
        if len(dims) != 2 or len(self._physical_dims(dims)) != 2:
            raise NotImplementedError(
                "triplet_margin_loss currently requires physical rank 2"
            )
        for name, value in (("positive", positive), ("negative", negative)):
            if not self.rules.same_shape(dims, self.shape_of(value)):
                raise ValueError(
                    f"triplet_margin_loss {name} must match anchor shape"
                )
        if not all(isinstance(value, (int, float)) for value in (margin, p, eps)):
            raise NotImplementedError(
                "dynamic triplet margin, p, and eps are not implemented"
            )
        if not isinstance(swap, bool):
            raise NotImplementedError("dynamic triplet swap is not implemented")
        reduction_tags = {"sum": 0, "mean": 1}
        if reduction not in reduction_tags:
            raise NotImplementedError(
                "triplet_margin_loss currently supports sum and mean reduction"
            )

        physical_dims = self._physical_dims(dims)
        callee = self.world.annex(torch_dialect.loss.triplet_margin_reduced.value)
        callee = self.world.app(
            callee, self._torch_semantics(anchor, floating=True)
        )
        callee = self._apply_grouped(callee, physical_dims)
        callee = self.world.app(
            callee,
            self.world.tuple(
                [
                    self._f32_float_lit(float(margin)),
                    self._f32_float_lit(float(p)),
                    self._f32_float_lit(float(eps)),
                    self.world.lit_bool(swap),
                    self._lit_nat(reduction_tags[reduction]),
                ]
            ),
        )
        result = self.world.app(
            callee, self.world.tuple([anchor, positive, negative])
        )
        return self._remember_shape(result, [])


    def _torch_reduce(self, kind, input, dim, keepdim):
        dims = self.shape_of(input)
        logical_rank = len(dims)
        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [dims[axis] for axis in physical_axes]
        rank = len(physical_dims)
        reduce_all = dim is None or (
            isinstance(dim, (tuple, list)) and len(dim) == 0
        )
        if reduce_all and kind == "sum" and not keepdim:
            callee = self.world.annex(torch_dialect.reduction.sum_all.value)
            callee = self.world.app(callee, self._torch_semantics(input))
            callee = self._apply_grouped(
                callee,
                [self._lit_nat(rank), self.world.tuple(physical_dims)],
            )
            result = self.world.app(callee, input)
            return self._remember_shape(result, [])
        if reduce_all and kind == "mean" and not keepdim:
            callee = self.world.annex(torch_dialect.reduction.mean_all.value)
            callee = self.world.app(
                callee, self._torch_semantics(input, floating=True)
            )
            callee = self._apply_grouped(
                callee, [self._lit_nat(rank), self.world.tuple(physical_dims)]
            )
            result = self.world.app(callee, input)
            return self._remember_shape(result, [])

        if reduce_all:
            dim_list = list(range(logical_rank))
        else:
            dim_list = list(dim) if isinstance(dim, (tuple, list)) else [dim]
        canonical = [d if d >= 0 else d + logical_rank for d in dim_list]
        if (
            not canonical
            or any(axis < 0 or axis >= logical_rank for axis in canonical)
            or len(set(canonical)) != len(canonical)
        ):
            raise ValueError(
                f"{kind} dimensions must be non-empty, unique, and in range"
            )
        output_dims = self.rules.reduce_shape_spec(
            dims, dim=canonical, keepdim=keepdim
        ).output_dims
        physical_reduction_dims = [
            physical_axes.index(axis)
            for axis in canonical
            if axis in physical_axes
        ]
        # Reducing an extent-one dimension is an identity for sum, mean, and
        # extrema. Its logical removal/retention remains visible in the cache.
        if not physical_reduction_dims:
            return self._remember_shape(input, output_dims)
        dim_values = [
            self.world.lit_i64(axis) for axis in physical_reduction_dims
        ]
        dim_tuple = self.world.tuple(dim_values)
        nr = self._lit_nat(len(dim_values))
        shape = self.world.tuple(physical_dims)
        if len(dim_values) == 1 and not keepdim and kind != "logsumexp":
            callee = self.world.annex(
                self._torch_annex_id(f"reduction.{kind}_dim")
            )
            dictionary = self._torch_semantics(
                input, floating=kind in ("amax", "mean", "logsumexp")
            )
            callee = self.world.app(callee, dictionary)
            callee = self._apply_grouped(callee, [self._lit_nat(rank), shape])
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        if kind == "sum" and len(dim_values) == 1 and keepdim:
            callee = self.world.annex(torch_dialect.reduction.sum_dim_keepdim.value)
            callee = self.world.app(callee, self._torch_semantics(input))
            callee = self._apply_grouped(callee, [self._lit_nat(rank), shape])
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        if kind == "mean" and len(dim_values) == 1 and keepdim:
            callee = self.world.annex(torch_dialect.reduction.mean_dim_keepdim.value)
            callee = self.world.app(
                callee, self._torch_semantics(input, floating=True)
            )
            callee = self._apply_grouped(
                callee, [self._lit_nat(rank), shape]
            )
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        name = (
            f"reduction.{kind}_dims_keepdim"
            if keepdim
            else f"reduction.{kind}_dims"
        )
        if keepdim and kind == "amax":
            name = "reduction.amax_dims"
        callee = self.world.annex(self._torch_annex_id(name))
        dictionary = self._torch_semantics(
            input, floating=kind in ("amax", "mean", "logsumexp")
        )
        callee = self.world.app(callee, dictionary)
        callee = self._apply_grouped(callee, [self._lit_nat(rank), nr, shape])
        callee = self.world.app(callee, dim_tuple)
        result = self.world.app(callee, input)
        if keepdim and kind == "amax":
            reduced_dims = self.rules.reduce_shape_spec(
                dims, dim=canonical, keepdim=False
            ).output_dims
            self._remember_shape(result, reduced_dims)
            return self.reshape(result, output_dims)
        return self._remember_shape(result, output_dims)

    def _f32_pair_reduce_lambda(self, pair_type):
        args_type = self.world.sigma([pair_type, self.F32])
        lam = self.world.mut_con([args_type, self.world.cn([pair_type])])
        args = lam.var().proj(2, 0)
        pair = args.proj(2, 0)
        value = args.proj(2, 1)
        sum_next = self.world.app(self.f32_add_axm, [pair.proj(2, 0), value])
        count_next = self.world.app(self.f32_add_axm, [pair.proj(2, 1), self._f32_float_lit(1.0)])
        lam.app(True, lam.var().proj(2, 1), [self.world.tuple([sum_next, count_next])])
        return lam

    def mean(self, input, dim=None, keepdim=False):
        """
        Translates to mean reduction.
        """
        return self._torch_reduce("mean", input, dim, keepdim)

    def logsumexp(self, input, dim, keepdim=False):
        """Map stable multi-axis logsumexp; its numerical semantics live in MimIR."""
        if dim is None:
            raise ValueError("logsumexp requires an explicit dim")
        return self._torch_reduce("logsumexp", input, dim, keepdim)

        # Legacy pair reduction retained below for reference.
        pair_type = self.world.arr(self._lit_nat(2), self.F32)
        reduced, output_dims = self._reduce_aff(
            input,
            pair_type,
            self._f32_pair_reduce_lambda(pair_type),
            self.world.tuple([self._f32_float_lit(0.0), self._f32_float_lit(0.0)]),
            dim=dim,
            keepdim=keepdim,
            return_shape=True,
        )
        rank = self._lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        res = self._unary_with_types(
            pair_type,
            self.F32,
            self._f32_pair_to_mean_lambda(pair_type),
            reduced,
            rank,
            shape,
        )
        return self._remember_shape(res, output_dims)

    def _f32_var_mean_reduce_lambda(self, acc_type):
        """
        Creates the reduction lambda for var_mean which maintains (sum, sum_sq, count).
        """
        args_type = self.world.sigma([acc_type, self.F32])
        lam = self.world.mut_con([args_type, self.world.cn([acc_type])])
        args = lam.var()
        acc = args.proj(2, 0)
        value = args.proj(2, 1)
        
        sum_acc = acc.proj(3, 0)
        sum_sq_acc = acc.proj(3, 1)
        count_acc = acc.proj(3, 2)
        
        sum_next = self.world.app(self.f32_add_axm, [sum_acc, value])
        
        val_sq = self.world.app(self.f32_mul_axm, [value, value])
        sum_sq_next = self.world.app(self.f32_add_axm, [sum_sq_acc, val_sq])
        
        count_next = self.world.app(self.f32_add_axm, [count_acc, self._f32_float_lit(1.0)])
        
        lam.app(True, lam.var().proj(2, 1), [self.world.tuple([sum_next, sum_sq_next, count_next])])
        return lam

    def _f32_acc_to_var_mean(self, acc_type, extract_var=True):
        """
        Finalizer map step for var_mean.
        """
        lam = self.world.mut_lam(acc_type, self.F32)
        acc = lam.var()
        s = acc.proj(3, 0)
        s_sq = acc.proj(3, 1)
        c = acc.proj(3, 2)
        
        mean = self.world.app(self.f32_div_axm, [s, c])
        
        if extract_var:
            mean_sq = self.world.app(self.f32_mul_axm, [mean, mean])
            e_x_sq = self.world.app(self.f32_div_axm, [s_sq, c])
            var = self.world.app(self.f32_sub_axm, [e_x_sq, mean_sq])
            lam.set_body(True, var)
        else:
            lam.set_body(True, mean)
            
        return lam

    def var_mean(self, input, dim=None, keepdim=False, correction=1):
        """Emit the Torch var_mean decomposition and restore optional keepdim."""
        if correction is None:
            correction = 1
        dims = self.shape_of(input)
        logical_rank = len(dims)
        dim_list = (
            list(range(logical_rank))
            if dim is None
            else list(dim) if isinstance(dim, (tuple, list)) else [dim]
        )
        canonical = [d if d >= 0 else d + logical_rank for d in dim_list]
        if (
            not canonical
            or any(axis < 0 or axis >= logical_rank for axis in canonical)
            or len(set(canonical)) != len(canonical)
        ):
            raise ValueError(
                "var_mean dimensions must be non-empty, unique, and in range"
            )

        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [dims[axis] for axis in physical_axes]
        physical_reduction_dims = [
            physical_axes.index(axis)
            for axis in canonical
            if axis in physical_axes
        ]
        if len(physical_reduction_dims) == len(physical_dims):
            raise NotImplementedError(
                "var_mean reduction to a rank-0 tensor is not implemented"
            )
        if not physical_reduction_dims:
            raise NotImplementedError(
                "var_mean over only folded singleton axes is not implemented"
            )

        dim_values = [
            self.world.lit_i64(axis) for axis in physical_reduction_dims
        ]
        callee = self.world.annex(torch_dialect.reduction.var_mean_dims.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee,
            [
                self._lit_nat(len(physical_dims)),
                self._lit_nat(len(dim_values)),
                self.world.tuple(physical_dims),
            ],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([
                self.world.tuple(dim_values),
                self._float_lit(
                    self._tensor_element_type(input), correction
                ),
            ]),
        )
        result = self.world.app(callee, input)
        reduced_dims = self.rules.reduce_shape_spec(
            dims, dim=canonical, keepdim=False
        ).output_dims
        outputs = []
        for index in range(2):
            # The Torch axiom carries an internal Bool tag to keep two
            # equal-shaped tensors from folding into one rank-higher tensor.
            output = result.proj(3, index + 1)
            self._remember_shape(output, reduced_dims)
            if keepdim:
                keep_dims = self.rules.reduce_shape_spec(
                    dims, dim=canonical, keepdim=True
                ).output_dims
                output = self.reshape(output, keep_dims)
            outputs.append(output)
        # Keep multiple results as frontend structure. Constructing a MimIR
        # tuple here would normalize equal-shaped tensors into one tensor with
        # an extra leading dimension, changing tuple projection into slicing.
        return tuple(outputs)

    # Linear Algebra
    def mm(self, lhs, rhs):
        """Translate `aten.mm`, whose operands are required to be matrices."""
        lhs_dims = self.shape_of(lhs)
        rhs_dims = self.shape_of(rhs)
        if len(lhs_dims) != 2 or len(rhs_dims) != 2:
            raise NotImplementedError("aten.mm requires two rank-2 operands")
        if not self.rules._same_dim(lhs_dims[-1], rhs_dims[-2]):
            raise ValueError("aten.mm contracting dimensions must match")

        callee = self.world.annex(torch_dialect.linalg.mm.value)
        callee = self.world.app(callee, self._torch_semantics(lhs))
        callee = self._apply_grouped(callee, [lhs_dims[0], lhs_dims[1], rhs_dims[1]])
        result = self.world.app(callee, self.world.tuple([lhs, rhs]))
        return self._remember_shape(result, [lhs_dims[0], rhs_dims[1]])

    def addmm(self, self_tensor, mat1, mat2, *, beta=1, alpha=1):
        """Map the complete `aten.addmm` contract to `%torch.linalg.addmm`."""
        self_dims = self.shape_of(self_tensor)
        mat1_dims = self.shape_of(mat1)
        mat2_dims = self.shape_of(mat2)
        if len(mat1_dims) != 2 or len(mat2_dims) != 2:
            raise NotImplementedError("aten.addmm requires two matrix operands")
        if len(self_dims) > 2:
            raise ValueError("aten.addmm self must be broadcastable to a matrix")
        if not self.rules._same_dim(mat1_dims[1], mat2_dims[0]):
            raise ValueError("aten.addmm contracting dimensions must match")

        # MimIR tensor types fold literal singleton dimensions. Preserve the
        # corresponding logical output axis in the in-dimension map.
        physical_axes = [
            axis
            for axis, extent in enumerate(self_dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_self_dims = [self_dims[axis] for axis in physical_axes]
        output_offset = 2 - len(self_dims)
        idx2 = self.world.type_idx(self._lit_nat(2))
        broadcast_dims = self.world.tuple(
            [self.world.lit(idx2, output_offset + axis) for axis in physical_axes]
        )

        m, k, n = mat1_dims[0], mat1_dims[1], mat2_dims[1]
        elem_type = self._tensor_element_type(mat1)
        callee = self.world.annex(torch_dialect.linalg.addmm.value)
        callee = self.world.app(callee, self._torch_semantics(mat1))
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(physical_self_dims)), m, k, n]
        )
        callee = self.world.app(
            callee, self.world.tuple([self.world.tuple(physical_self_dims), broadcast_dims])
        )
        callee = self.world.app(
            callee, self.world.tuple([self_tensor, mat1, mat2])
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [self._float_lit(elem_type, beta), self._float_lit(elem_type, alpha)]
            ),
        )
        return self._remember_shape(result, [m, n])

    def matmul(self, lhs, rhs):
        """Map `aten.matmul`; rank dispatch and broadcasting live in MimIR."""
        lhs_dims = self._physical_dims(self.shape_of(lhs))
        rhs_dims = self._physical_dims(self.shape_of(rhs))
        callee = self.world.annex(torch_dialect.linalg.matmul.value)
        callee = self.world.app(callee, self._torch_semantics(lhs))
        callee = self._apply_grouped(
            callee,
            [
                self._lit_nat(len(lhs_dims)),
                self._lit_nat(len(rhs_dims)),
                self.world.tuple(lhs_dims),
                self.world.tuple(rhs_dims),
            ],
        )
        return self.world.app(callee, self.world.tuple([lhs, rhs]))

    def bmm(self, lhs, rhs):
        """Map `aten.bmm`, normalizing composite high-rank uses through matmul.

        Native ``torch.bmm`` is rank-3-only.  PyTorch decompositions used by
        attention kernels can nevertheless leave a higher-rank ``aten.bmm``
        in the FX graph before the batch dimensions have been folded.  The
        Torch layer owns that normalization; the tensor layer remains a
        strict rank-3 primitive.
        """
        lhs_dims = self._physical_dims(self.shape_of(lhs))
        rhs_dims = self._physical_dims(self.shape_of(rhs))
        if len(lhs_dims) != 3 or len(rhs_dims) != 3:
            if len(lhs_dims) > 3 and len(rhs_dims) > 3:
                return self.matmul(lhs, rhs)
            raise ValueError("aten.bmm expects two rank-3 tensors")

        batch, rows, contract = lhs_dims
        rhs_batch, rhs_contract, cols = rhs_dims
        if not self.rules._same_dim(batch, rhs_batch):
            raise ValueError("aten.bmm batch dimensions must match")
        if not self.rules._same_dim(contract, rhs_contract):
            raise ValueError("aten.bmm contracting dimensions must match")

        callee = self.world.annex(torch_dialect.linalg.bmm.value)
        callee = self.world.app(callee, self._torch_semantics(lhs))
        callee = self._apply_grouped(callee, [batch, rows, contract, cols])
        result = self.world.app(callee, self.world.tuple([lhs, rhs]))
        return self._remember_shape(result, [batch, rows, cols])

    def linear(self, input, weight, bias=None):
        input_dims = self.shape_of(input)
        weight_dims = self.shape_of(weight)
        if len(input_dims) < 2 or len(weight_dims) != 2:
            raise NotImplementedError("linear requires input rank >= 2 and 2D weight")

        in_features = input_dims[-1]
        out_features = weight_dims[0]
        if not self.rules._same_dim(weight_dims[1], in_features):
            raise NotImplementedError("linear weight in_features must match input features")

        batch_dims = input_dims[:-1]
        elem_t = self._tensor_element_type(input)
        callee = self.world.annex(torch_dialect.linalg.linear.value)
        callee = self.world.app(callee, self._torch_semantics(input))
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(batch_dims)), in_features, out_features]
        )
        callee = self.world.app(callee, self.world.tuple(batch_dims))
        callee = self.world.app(callee, self.world.tuple([input, weight]))
        if bias is None:
            bias_t = self.world.arr(out_features, elem_t)
            optional_bias = self.world.app(self.world.annex(option.none.value), bias_t)
        else:
            optional_bias = self.world.implicit_app(self.world.annex(option.some.value), bias)
        result = self.world.app(callee, optional_bias)
        output_dims = batch_dims + [out_features]
        return self._remember_shape(result, output_dims)

    def _optional_recurrent_bias(self, bias, size, elem_t):
        if bias is not None:
            return self.world.implicit_app(self.world.annex(option.some.value), bias)
        bias_t = self.world.arr(size, elem_t)
        return self.world.app(self.world.annex(option.none.value), bias_t)

    def _recurrent_direction(
        self,
        kind,
        input,
        hidden,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        *,
        reverse,
        relu=False,
        cell=None,
    ):
        """Map one standard recurrent direction to a Torch dialect axiom."""
        seq, batch, input_size = self.shape_of(input)
        hidden_dims = self.shape_of(hidden)
        if len(hidden_dims) != 2:
            raise ValueError("recurrent hidden state must have shape [batch, hidden]")
        hidden_size = hidden_dims[1]
        gate_size = self.shape_of(weight_ih)[0]
        elem_t = self._tensor_element_type(input)

        axiom = {
            "rnn": torch_dialect.recurrent.rnn_direction,
            "gru": torch_dialect.recurrent.gru_direction,
            "lstm": torch_dialect.recurrent.lstm_direction,
        }[kind]
        callee = self.world.annex(axiom.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee, [seq, batch, input_size, hidden_size]
        )
        if kind == "rnn":
            callee = self._apply_grouped(
                callee,
                [self.world.lit_bool(relu), self.world.lit_bool(reverse)],
            )
        else:
            callee = self.world.app(callee, self.world.lit_bool(reverse))

        args = [input, hidden]
        if kind == "lstm":
            if cell is None:
                raise ValueError("LSTM requires an initial cell state")
            args.append(cell)
        args.extend([
            weight_ih,
            weight_hh,
            self._optional_recurrent_bias(bias_ih, gate_size, elem_t),
            self._optional_recurrent_bias(bias_hh, gate_size, elem_t),
        ])
        result = self.world.app(callee, self.world.tuple(args))

        output = result.proj(3 if kind == "lstm" else 2, 0)
        final_hidden = result.proj(3 if kind == "lstm" else 2, 1)
        self._remember_shape(output, [seq, batch, hidden_size])
        self._remember_shape(final_hidden, [batch, hidden_size])
        if kind != "lstm":
            return output, final_hidden
        final_cell = result.proj(3, 2)
        self._remember_shape(final_cell, [batch, hidden_size])
        return output, final_hidden, final_cell

    def _stack_recurrent_states(self, states):
        if len(states) == 1:
            return self.unsqueeze(states[0], 0)
        batch, hidden = self.shape_of(states[0])
        # A literal batch extent of one is erased from MimIR's physical array
        # type. Concatenating the remaining hidden axis still lays out complete
        # states consecutively, and the final reshape restores [D*L, B, H].
        cat_dim = 1 if self._is_lit_nat_value(batch, 1) else 0
        flattened = self.cat(states, dim=cat_dim)
        return self.reshape(flattened, [len(states), batch, hidden])

    def recurrent(
        self,
        kind,
        input,
        hx,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        *,
        relu=False,
    ):
        """Translate the standard ATen RNN/GRU/LSTM input overload.

        Python handles only the statically configured layer/direction parameter
        layout. Every recurrent direction, including its time loop and gate
        semantics, remains a single `%torch.*_direction_op`.
        """
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("recurrent num_layers must be a positive integer")
        if train and dropout != 0:
            raise NotImplementedError(
                "training recurrent dropout requires an RNG/effectful dropout operator"
            )

        input_dims = self.shape_of(input)
        if len(input_dims) != 3:
            raise NotImplementedError(
                "standard recurrent input overload requires a rank-3 tensor"
            )
        if batch_first:
            input = self.transpose_int(input, 0, 1)
            batch, seq, _ = input_dims
        else:
            seq, batch, _ = input_dims

        if kind == "lstm":
            hidden_state, cell_state = hx
        else:
            hidden_state, cell_state = hx, None
        hidden_dims = self.shape_of(hidden_state)
        if len(hidden_dims) != 3:
            raise ValueError("recurrent hidden state must have rank 3")
        hidden_size = hidden_dims[2]
        directions = 2 if bidirectional else 1
        params_per_direction = 4 if has_biases else 2
        expected_params = num_layers * directions * params_per_direction
        if len(params) != expected_params:
            raise ValueError(
                f"recurrent parameter list has {len(params)} entries, "
                f"expected {expected_params}"
            )

        layer_input = input
        final_hidden = []
        final_cells = []
        cursor = 0
        for layer in range(num_layers):
            direction_outputs = []
            for direction in range(directions):
                state_index = layer * directions + direction
                h0 = self.select(hidden_state, 0, state_index)
                c0 = (
                    self.select(cell_state, 0, state_index)
                    if cell_state is not None
                    else None
                )
                weight_ih, weight_hh = params[cursor:cursor + 2]
                cursor += 2
                if has_biases:
                    bias_ih, bias_hh = params[cursor:cursor + 2]
                    cursor += 2
                else:
                    bias_ih = bias_hh = None

                result = self._recurrent_direction(
                    kind,
                    layer_input,
                    h0,
                    weight_ih,
                    weight_hh,
                    bias_ih,
                    bias_hh,
                    reverse=direction == 1,
                    relu=relu,
                    cell=c0,
                )
                direction_outputs.append(result[0])
                final_hidden.append(result[1])
                if kind == "lstm":
                    final_cells.append(result[2])

            layer_input = (
                direction_outputs[0]
                if directions == 1
                else self.cat(direction_outputs, dim=2)
            )

        output = (
            self.transpose_int(layer_input, 0, 1)
            if batch_first
            else layer_input
        )
        output_dims = (
            [batch, seq, directions * self.rules._dim_literal_value(hidden_size)]
            if batch_first and self.rules._dim_literal_value(hidden_size) is not None
            else self.shape_of(output)
        )
        self._remember_shape(output, output_dims)
        hidden_out = self._stack_recurrent_states(final_hidden)
        if kind != "lstm":
            return self.world.tuple([output, hidden_out])
        cell_out = self._stack_recurrent_states(final_cells)
        return self.world.tuple([output, hidden_out, cell_out])

    def batch_norm_inference(
        self, input, running_mean, running_var, weight=None, bias=None, eps=1e-5
    ):
        dims = self.shape_of(input)
        if len(dims) < 2:
            raise NotImplementedError("batch_norm expects input rank >= 2")
        channels = dims[1]
        for name, value in (
            ("running_mean", running_mean),
            ("running_var", running_var),
            ("weight", weight),
            ("bias", bias),
        ):
            if value is None:
                if name.startswith("running_"):
                    raise NotImplementedError(
                        f"batch_norm inference requires {name}"
                    )
                continue
            value_dims = self.shape_of(value)
            if len(value_dims) != 1 or not self.rules._same_dim(
                value_dims[0], channels
            ):
                raise NotImplementedError(
                    f"batch_norm {name} must have shape [channels]"
                )

        callee = self.world.annex(torch_dialect.normalization.batch_norm_inference.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(dims)), self.world.tuple(dims)]
        )
        channel_type = self.world.arr(channels, self._tensor_element_type(input))

        def optional(value):
            if value is None:
                return self.world.app(
                    self.world.annex(option.none.value), channel_type
                )
            return self.world.implicit_app(
                self.world.annex(option.some.value), value
            )

        result = self.world.app(
            callee,
            self.world.tuple([
                input, optional(weight), optional(bias), running_mean,
                running_var, self._f32_float_lit(eps),
            ]),
        )
        return self._remember_shape(result, dims)

    def _tensor_reshape(self, x, shape):
        in_rank, in_shape_tuple = self._rank_and_shape(x)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self._lit_nat(out_rank_val)
        elem_t = self._tensor_element_type(x)
        callee = self.world.annex(tensor.reshape.value)
        callee = self._apply_grouped(callee, [elem_t, in_rank, out_rank])
        callee = self.world.app(callee, in_shape_tuple)
        callee = self.world.app(callee, out_shape_tuple)
        result = self.world.app(callee, x)
        return self._remember_shape(result, list(shape))

    def convolution(
        self,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        transposed=False,
        output_padding=0,
    ):
        semantic_in_dims = self.shape_of(x)
        weight_dims = self.shape_of(weight)
        if transposed:
            if len(semantic_in_dims) == 3 and len(weight_dims) == 3:
                return self.convolution_transpose1d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                )
            if len(semantic_in_dims) == 4 and len(weight_dims) == 4:
                return self.convolution_transpose2d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                )
            if len(semantic_in_dims) == 5 and len(weight_dims) == 5:
                return self.convolution_transpose3d(
                    x,
                    weight,
                    bias=bias,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    groups=groups,
                )
            raise NotImplementedError(
                "transposed convolution currently supports rank-3 NCL and "
                "rank-4 NCHW, and rank-5 NCDHW inputs"
            )
        if len(semantic_in_dims) == 3 and len(weight_dims) == 3:
            return self.convolution1d(
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if len(semantic_in_dims) == 5 and len(weight_dims) == 5:
            return self.convolution3d(
                x,
                weight,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
        if len(semantic_in_dims) != 4 or len(weight_dims) != 4:
            raise NotImplementedError(
                "aten.convolution currently supports rank-3 NCL, rank-4 NCHW, "
                "and rank-5 NCDHW inputs"
            )

        type_in_dims = self._shape_dims(x)
        if len(type_in_dims) == 3 and self._is_lit_nat_value(semantic_in_dims[0], 1):
            in_dims = [semantic_in_dims[0], type_in_dims[0], type_in_dims[1], type_in_dims[2]]
        else:
            in_dims = semantic_in_dims

        stride = self._pair(stride, "stride")
        padding = self._pair(padding, "padding")
        dilation = self._pair(dilation, "dilation")

        n, cin, h, w = in_dims
        cout, cin_per_group, kh, kw = weight_dims
        out_spatial = self._conv2d_spatial_shape((h, w), (kh, kw), stride, dilation, padding)
        out_dims = [n, cout, out_spatial[0], out_spatial[1]]

        callee = self.world.annex(torch_dialect.conv.general.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee, [n, cin, cout, cin_per_group, h, w, kh, kw]
        )
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([self._to_nat(stride[0]), self._to_nat(stride[1])]),
                self.world.tuple([self._to_nat(padding[0]), self._to_nat(padding[1])]),
                self.world.tuple([self._to_nat(dilation[0]), self._to_nat(dilation[1])]),
                self.world.lit_bool(False),
                self.world.tuple([self._lit_nat(0), self._lit_nat(0)]),
                self._to_nat(groups),
            ],
        )
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def convolution_transpose2d(
        self,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    ):
        in_dims = self.shape_of(x)
        weight_dims = self.shape_of(weight)
        stride = self._pair(stride, "stride")
        padding = self._pair(padding, "padding")
        dilation = self._pair(dilation, "dilation")
        output_padding = self._pair(output_padding, "output_padding")
        n, cin, height, width = in_dims
        weight_cin, cout_per_group, kh, kw = weight_dims
        cout = self._nat_binop(core.nat.mul, cout_per_group, groups)

        def output_dim(size, kernel, step, dil, pad, out_pad):
            expanded = self._nat_binop(
                core.nat.mul, self._nat_binop(core.nat.sub, size, 1), step
            )
            kernel_span = self._nat_binop(
                core.nat.mul, dil, self._nat_binop(core.nat.sub, kernel, 1)
            )
            padded = self._nat_binop(
                core.nat.sub,
                self._nat_binop(core.nat.add, expanded, kernel_span),
                self._nat_binop(core.nat.mul, 2, pad),
            )
            return self._nat_binop(
                core.nat.add,
                padded,
                self._nat_binop(core.nat.add, out_pad, 1),
            )

        out_spatial = [
            output_dim(size, kernel, step, dil, pad, out_pad)
            for size, kernel, step, dil, pad, out_pad in zip(
                (height, width),
                (kh, kw),
                stride,
                dilation,
                padding,
                output_padding,
            )
        ]
        out_dims = [n, cout, *out_spatial]
        callee = self.world.annex(torch_dialect.conv.transpose2d.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee,
            [n, cin, cout, cout_per_group, height, width, kh, kw],
        )
        pairs = [
            self.world.tuple([self._to_nat(value) for value in values])
            for values in (stride, padding, dilation, output_padding)
        ]
        callee = self._apply_grouped(callee, [*pairs, self._to_nat(groups)])
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def convolution_transpose1d(
        self,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    ):
        n, cin, length = self.shape_of(x)
        weight_cin, cout_per_group, kernel = self.shape_of(weight)
        stride = self._single(stride, "stride")
        padding = self._single(padding, "padding")
        dilation = self._single(dilation, "dilation")
        output_padding = self._single(output_padding, "output_padding")
        cout = self._nat_binop(core.nat.mul, cout_per_group, groups)
        expanded = self._nat_binop(
            core.nat.mul, self._nat_binop(core.nat.sub, length, 1), stride
        )
        kernel_span = self._nat_binop(
            core.nat.mul,
            dilation,
            self._nat_binop(core.nat.sub, kernel, 1),
        )
        padded = self._nat_binop(
            core.nat.sub,
            self._nat_binop(core.nat.add, expanded, kernel_span),
            self._nat_binop(core.nat.mul, 2, padding),
        )
        out_length = self._nat_binop(
            core.nat.add,
            padded,
            self._nat_binop(core.nat.add, output_padding, 1),
        )
        out_dims = [n, cout, out_length]
        callee = self.world.annex(torch_dialect.conv.transpose1d.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee, [n, cin, cout, cout_per_group, length, kernel]
        )
        callee = self._apply_grouped(
            callee,
            [
                self._to_nat(stride),
                self._to_nat(padding),
                self._to_nat(dilation),
                self._to_nat(output_padding),
                self._to_nat(groups),
            ],
        )
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def convolution_transpose3d(
        self,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
    ):
        n, cin, depth, height, width = self.shape_of(x)
        weight_cin, cout_per_group, kd, kh, kw = self.shape_of(weight)
        stride = self._triple(stride, "stride")
        padding = self._triple(padding, "padding")
        dilation = self._triple(dilation, "dilation")
        output_padding = self._triple(output_padding, "output_padding")
        cout = self._nat_binop(core.nat.mul, cout_per_group, groups)

        def output_dim(size, kernel, step, dil, pad, out_pad):
            expanded = self._nat_binop(
                core.nat.mul, self._nat_binop(core.nat.sub, size, 1), step
            )
            kernel_span = self._nat_binop(
                core.nat.mul, dil, self._nat_binop(core.nat.sub, kernel, 1)
            )
            padded = self._nat_binop(
                core.nat.sub,
                self._nat_binop(core.nat.add, expanded, kernel_span),
                self._nat_binop(core.nat.mul, 2, pad),
            )
            return self._nat_binop(
                core.nat.add,
                padded,
                self._nat_binop(core.nat.add, out_pad, 1),
            )

        out_spatial = [
            output_dim(size, kernel, step, dil, pad, out_pad)
            for size, kernel, step, dil, pad, out_pad in zip(
                (depth, height, width),
                (kd, kh, kw),
                stride,
                dilation,
                padding,
                output_padding,
            )
        ]
        out_dims = [n, cout, *out_spatial]
        callee = self.world.annex(torch_dialect.conv.transpose3d.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee,
            [n, cin, cout, cout_per_group, depth, height, width, kd, kh, kw],
        )
        triples = [
            self.world.tuple([self._to_nat(value) for value in values])
            for values in (stride, padding, dilation, output_padding)
        ]
        callee = self._apply_grouped(callee, [*triples, self._to_nat(groups)])
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def convolution3d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        semantic_in_dims = self.shape_of(x)
        weight_dims = self.shape_of(weight)
        type_in_dims = self._shape_dims(x)
        if len(type_in_dims) == 4 and self._is_lit_nat_value(semantic_in_dims[0], 1):
            in_dims = [semantic_in_dims[0], *type_in_dims]
        else:
            in_dims = semantic_in_dims

        stride = self._triple(stride, "stride")
        padding = self._triple(padding, "padding")
        dilation = self._triple(dilation, "dilation")
        n, cin, depth, height, width = in_dims
        cout, cin_per_group, kd, kh, kw = weight_dims
        out_spatial = [
            self._conv2d_dim(size, kernel, step, dil, pad)
            for size, kernel, step, dil, pad in zip(
                (depth, height, width),
                (kd, kh, kw),
                stride,
                dilation,
                padding,
            )
        ]
        out_dims = [n, cout, *out_spatial]

        callee = self.world.annex(torch_dialect.conv.conv3d.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee,
            [n, cin, cout, cin_per_group, depth, height, width, kd, kh, kw],
        )
        triples = [
            self.world.tuple([self._to_nat(value) for value in values])
            for values in (stride, padding, dilation)
        ]
        callee = self._apply_grouped(
            callee,
            [
                *triples,
                self.world.lit_bool(False),
                self.world.tuple([self._lit_nat(0)] * 3),
                self._to_nat(groups),
            ],
        )
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def convolution1d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        semantic_in_dims = self.shape_of(x)
        weight_dims = self.shape_of(weight)
        type_in_dims = self._shape_dims(x)
        if len(type_in_dims) == 2 and self._is_lit_nat_value(semantic_in_dims[0], 1):
            in_dims = [semantic_in_dims[0], type_in_dims[0], type_in_dims[1]]
        else:
            in_dims = semantic_in_dims

        stride = self._single(stride, "stride")
        padding = self._single(padding, "padding")
        dilation = self._single(dilation, "dilation")
        n, cin, length = in_dims
        cout, cin_per_group, kernel = weight_dims
        out_length = self._conv2d_dim(length, kernel, stride, dilation, padding)
        out_dims = [n, cout, out_length]

        callee = self.world.annex(torch_dialect.conv.conv1d.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee, [n, cin, cout, cin_per_group, length, kernel]
        )
        callee = self._apply_grouped(
            callee,
            [
                self._to_nat(stride),
                self._to_nat(padding),
                self._to_nat(dilation),
                self.world.lit_bool(False),
                self._lit_nat(0),
                self._to_nat(groups),
            ],
        )
        if bias is None:
            bias_type = self.world.arr(cout, self._tensor_element_type(x))
            optional_bias = self.world.app(
                self.world.annex(option.none.value), bias_type
            )
        else:
            optional_bias = self.world.implicit_app(
                self.world.annex(option.some.value), bias
            )
        result = self.world.app(callee, self.world.tuple([x, weight, optional_bias]))
        return self._remember_shape(result, out_dims)

    def _to_nat(self, value):
        return self._lit_nat(value) if isinstance(value, int) else value

    def _pair(self, value, name):
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(value)
        raise NotImplementedError(f"{name} must be an int or length-2 sequence")

    def _triple(self, value, name):
        if isinstance(value, int):
            return (value, value, value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return tuple(value)
        raise NotImplementedError(f"{name} must be an int or length-3 sequence")

    def _single(self, value, name):
        if isinstance(value, int):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        raise NotImplementedError(f"{name} must be an int or length-1 sequence")

    def _nat_binop(self, op, lhs, rhs):
        return self.world.call(op, [self._to_nat(lhs), self._to_nat(rhs)])

    def _nat_product(self, dims):
        result = self._lit_nat(1)
        for dim in dims:
            result = self._nat_binop(core.nat.mul, result, dim)
        return result

    def nat_floordiv(self, lhs, rhs):
        return self._nat_binop(core.nat.div, lhs, rhs)

    def _conv2d_dim(self, size, kernel, stride, dilation, padding):
        two_pad = self._nat_binop(core.nat.mul, 2, padding)
        padded = self._nat_binop(core.nat.add, size, two_pad)
        kernel_minus_one = self._nat_binop(core.nat.sub, kernel, 1)
        effective_kernel_minus_one = self._nat_binop(core.nat.mul, dilation, kernel_minus_one)
        numerator = self._nat_binop(
            core.nat.sub,
            self._nat_binop(core.nat.sub, padded, effective_kernel_minus_one),
            1,
        )
        return self._nat_binop(core.nat.add, self._nat_binop(core.nat.div, numerator, stride), 1)

    def _conv2d_spatial_shape(self, spatial, kernel, stride, dilation, padding):
        return [
            self._conv2d_dim(spatial[0], kernel[0], stride[0], dilation[0], padding[0]),
            self._conv2d_dim(spatial[1], kernel[1], stride[1], dilation[1], padding[1]),
        ]

    def pool2d(self, x, kernel_size, stride=None, padding=0, dilation=1, mode="max"):
        semantic_in_dims = self.shape_of(x)
        if len(semantic_in_dims) != 4:
            raise NotImplementedError("pool2d currently supports 4D NCHW inputs only")

        type_in_dims = self._shape_dims(x)
        if len(type_in_dims) == 3 and self._is_lit_nat_value(semantic_in_dims[0], 1):
            in_dims = [semantic_in_dims[0], type_in_dims[0], type_in_dims[1], type_in_dims[2]]
        else:
            in_dims = semantic_in_dims

        kernel = self._pair(kernel_size, "kernel_size")
        if stride is None:
            stride = kernel
        stride = self._pair(stride, "stride")
        padding = self._pair(padding, "padding")
        dilation = self._pair(dilation, "dilation")

        n, c, h, w = in_dims
        out_spatial = self._conv2d_spatial_shape((h, w), kernel, stride, dilation, padding)
        out_dims = [n, c, out_spatial[0], out_spatial[1]]

        if mode == "max":
            reduce_fn = self._scalar_binary_lambda(self.f32_max_axm, self.F32, self.F32)
            init = self._f32_float_lit(-float("inf"))
        elif mode == "avg":
            reduce_fn = self._scalar_binary_lambda(self.f32_add_axm, self.F32, self.F32)
            init = self._f32_float_lit(0.0)
        else:
            raise NotImplementedError(f"pool2d mode {mode} is not implemented")

        callee = self.world.annex(tensor.pool.value)
        callee = self.world.app(callee, self.F32)
        callee = self._apply_grouped(callee, [reduce_fn, init])
        callee = self._apply_grouped(callee, [n, c, h, w])
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([self._to_nat(kernel[0]), self._to_nat(kernel[1])]),
                self.world.tuple([self._to_nat(stride[0]), self._to_nat(stride[1])]),
                self.world.tuple([self._to_nat(dilation[0]), self._to_nat(dilation[1])]),
                self.world.tuple([self._to_nat(padding[0]), self._to_nat(padding[1])]),
                self.world.tuple(out_spatial),
            ],
        )
        result = self.world.app(callee, [x])
        result = self._remember_shape(result, out_dims)

        if mode == "avg":
            scale = 1.0 / float(kernel[0] * kernel[1])
            result = self.mul(result, scale)
            self._remember_shape(result, out_dims)
        return result

    def max_pool2d(self, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
        in_dims = self.shape_of(x)
        if len(in_dims) != 4:
            raise NotImplementedError(
                f"max_pool2d currently supports 4D NCHW inputs only, got {in_dims}"
            )
        kernel = self._pair(kernel_size, "kernel_size")
        if stride is None:
            stride = kernel
        stride = self._pair(stride, "stride")
        padding = self._pair(padding, "padding")
        dilation = self._pair(dilation, "dilation")

        n, c, h, w = in_dims
        out_spatial = [
            self._pool2d_dim(h, kernel[0], stride[0], dilation[0], padding[0], ceil_mode),
            self._pool2d_dim(w, kernel[1], stride[1], dilation[1], padding[1], ceil_mode),
        ]
        operator = (
            torch_dialect.pool.max_pool2d_with_indices
            if return_indices
            else torch_dialect.pool.max_pool2d
        )
        callee = self.world.annex(operator.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, h, w])
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([self._to_nat(kernel[0]), self._to_nat(kernel[1])]),
                self.world.tuple([self._to_nat(stride[0]), self._to_nat(stride[1])]),
                self.world.tuple([self._to_nat(padding[0]), self._to_nat(padding[1])]),
                self.world.tuple([self._to_nat(dilation[0]), self._to_nat(dilation[1])]),
                self.world.lit_bool(ceil_mode),
            ],
        )
        result = self.world.app(callee, x)
        out_dims = [n, c, *out_spatial]
        if return_indices:
            self._remember_shape(result.proj(2, 0), out_dims)
            self._remember_shape(result.proj(2, 1), out_dims)
            return result
        return self._remember_shape(result, out_dims)

    def max_pool1d(
        self, x, kernel_size, stride=None, padding=0, dilation=1,
        ceil_mode=False
    ):
        in_dims = self.shape_of(x)
        if len(in_dims) != 3:
            raise NotImplementedError("max_pool1d requires a rank-3 NCL input")
        kernel = self._single(kernel_size, "kernel_size")
        if stride is None or stride == []:
            stride = kernel
        stride = self._single(stride, "stride")
        padding = self._single(padding, "padding")
        dilation = self._single(dilation, "dilation")
        n, c, length = in_dims
        output_length = self._pool2d_dim(
            length, kernel, stride, dilation, padding, ceil_mode
        )
        callee = self.world.annex(torch_dialect.pool.max_pool1d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, length])
        callee = self.world.app(
            callee,
            self.world.tuple([
                self._to_nat(kernel), self._to_nat(stride),
                self._to_nat(padding), self._to_nat(dilation),
                self.world.lit_bool(ceil_mode),
            ]),
        )
        return self._remember_shape(self.world.app(callee, x), [n, c, output_length])

    def max_pool3d(
        self, x, kernel_size, stride=None, padding=0, dilation=1,
        ceil_mode=False
    ):
        in_dims = self.shape_of(x)
        if len(in_dims) != 5:
            raise NotImplementedError("max_pool3d requires a rank-5 NCDHW input")
        kernel = self._triple(kernel_size, "kernel_size")
        if stride is None or stride == []:
            stride = kernel
        stride = self._triple(stride, "stride")
        padding = self._triple(padding, "padding")
        dilation = self._triple(dilation, "dilation")
        n, c, depth, height, width = in_dims
        spatial = (depth, height, width)
        out_spatial = [
            self._pool2d_dim(spatial[i], kernel[i], stride[i], dilation[i],
                             padding[i], ceil_mode)
            for i in range(3)
        ]
        callee = self.world.annex(torch_dialect.pool.max_pool3d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, depth, height, width])
        params = [
            self.world.tuple([self._to_nat(v) for v in values])
            for values in (kernel, stride, padding, dilation)
        ]
        params.append(self.world.lit_bool(ceil_mode))
        result = self.world.app(self._apply_grouped(callee, params), x)
        return self._remember_shape(result, [n, c, *out_spatial])

    def _pool2d_dim(self, size, kernel, stride, dilation, padding, ceil_mode):
        if not all(isinstance(value, int) for value in (size, kernel, stride, dilation, padding)):
            return self._conv2d_dim(size, kernel, stride, dilation, padding)
        effective_kernel = dilation * (kernel - 1) + 1
        numerator = size + 2 * padding - effective_kernel
        if ceil_mode:
            numerator += stride - 1
        output = numerator // stride + 1
        if ceil_mode and (output - 1) * stride >= size + padding:
            output -= 1
        return self._lit_nat(output)

    def hardtanh(self, x, min_val=-1.0, max_val=1.0):
        dims = self.shape_of(x)
        physical_dims = self._physical_dims(dims)
        callee = self.world.annex(torch_dialect.activation.hardtanh.value)
        callee = self.world.app(callee, self._torch_semantics(x))
        callee = self._apply_grouped(
            callee, [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)]
        )
        result = self.world.app(
            callee,
            self.world.tuple([
                x,
                self._f32_float_lit(float(min_val)),
                self._f32_float_lit(float(max_val)),
            ]),
        )
        return self._remember_shape(result, dims)

    def avg_pool2d(
        self,
        x,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        in_dims = self.shape_of(x)
        if len(in_dims) != 4:
            raise NotImplementedError("avg_pool2d currently supports 4D NCHW inputs only")
        kernel = self._pair(kernel_size, "kernel_size")
        if stride is None or stride == []:
            stride = kernel
        stride = self._pair(stride, "stride")
        padding = self._pair(padding, "padding")
        n, c, h, w = in_dims
        out_spatial = [
            self._pool2d_dim(h, kernel[0], stride[0], 1, padding[0], ceil_mode),
            self._pool2d_dim(w, kernel[1], stride[1], 1, padding[1], ceil_mode),
        ]

        callee = self.world.annex(torch_dialect.pool.avg_pool2d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, h, w])
        if divisor_override is None:
            optional_divisor = self.world.app(
                self.world.annex(option.none.value), self.world.type_nat()
            )
        else:
            divisor = self._to_nat(divisor_override)
            optional_divisor = self.world.implicit_app(
                self.world.annex(option.some.value), divisor
            )
        callee = self._apply_grouped(
            callee,
            [
                self.world.tuple([self._to_nat(kernel[0]), self._to_nat(kernel[1])]),
                self.world.tuple([self._to_nat(stride[0]), self._to_nat(stride[1])]),
                self.world.tuple([self._to_nat(padding[0]), self._to_nat(padding[1])]),
                self.world.lit_bool(ceil_mode),
                self.world.lit_bool(count_include_pad),
                optional_divisor,
            ],
        )
        result = self.world.app(callee, x)
        return self._remember_shape(result, [n, c, *out_spatial])

    def avg_pool1d(
        self, x, kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True
    ):
        in_dims = self.shape_of(x)
        if len(in_dims) != 3:
            raise NotImplementedError("avg_pool1d requires a rank-3 NCL input")
        kernel = self._single(kernel_size, "kernel_size")
        if stride is None or stride == []:
            stride = kernel
        stride = self._single(stride, "stride")
        padding = self._single(padding, "padding")
        n, c, length = in_dims
        output_length = self._pool2d_dim(
            length, kernel, stride, 1, padding, ceil_mode
        )
        callee = self.world.annex(torch_dialect.pool.avg_pool1d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, length])
        divisor_override = self.world.app(
            self.world.annex(option.none.value), self.world.type_nat()
        )
        callee = self.world.app(
            callee,
            self.world.tuple([
                self._to_nat(kernel), self._to_nat(stride),
                self._to_nat(padding), self.world.lit_bool(ceil_mode),
                self.world.lit_bool(count_include_pad), divisor_override,
            ]),
        )
        return self._remember_shape(self.world.app(callee, x), [n, c, output_length])

    def avg_pool3d(
        self, x, kernel_size, stride=None, padding=0, ceil_mode=False,
        count_include_pad=True, divisor_override=None
    ):
        in_dims = self.shape_of(x)
        if len(in_dims) != 5:
            raise NotImplementedError("avg_pool3d requires a rank-5 NCDHW input")
        kernel = self._triple(kernel_size, "kernel_size")
        if stride is None or stride == []:
            stride = kernel
        stride = self._triple(stride, "stride")
        padding = self._triple(padding, "padding")
        n, c, depth, height, width = in_dims
        spatial = (depth, height, width)
        out_spatial = [
            self._pool2d_dim(spatial[i], kernel[i], stride[i], 1,
                             padding[i], ceil_mode)
            for i in range(3)
        ]
        callee = self.world.annex(torch_dialect.pool.avg_pool3d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, depth, height, width])
        if divisor_override is None:
            optional_divisor = self.world.app(
                self.world.annex(option.none.value), self.world.type_nat()
            )
        else:
            optional_divisor = self.world.implicit_app(
                self.world.annex(option.some.value), self._to_nat(divisor_override)
            )
        params = [
            self.world.tuple([self._to_nat(v) for v in values])
            for values in (kernel, stride, padding)
        ]
        params.extend([
            self.world.lit_bool(ceil_mode),
            self.world.lit_bool(count_include_pad), optional_divisor,
        ])
        result = self.world.app(self._apply_grouped(callee, params), x)
        return self._remember_shape(result, [n, c, *out_spatial])

    def adaptive_avg_pool1d(self, x, output_size):
        in_dims = self.shape_of(x)
        if len(in_dims) != 3:
            raise NotImplementedError(
                "adaptive_avg_pool1d requires a rank-3 NCL input"
            )
        output_size = self._single(output_size, "output_size")
        if output_size != 1:
            raise NotImplementedError(
                "adaptive_avg_pool1d currently supports output_size=1"
            )
        n, c, length = in_dims
        callee = self.world.annex(torch_dialect.pool.adaptive_avg_pool1d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, length])
        result = self.world.app(callee, self._lit_nat(output_size))
        result = self.world.app(result, x)
        return self._remember_shape(result, [n, c, self._lit_nat(1)])

    def adaptive_avg_pool3d(self, x, output_size):
        in_dims = self.shape_of(x)
        if len(in_dims) != 5:
            raise NotImplementedError(
                "adaptive_avg_pool3d requires a rank-5 NCDHW input"
            )
        if isinstance(output_size, int):
            output_size = (output_size,) * 3
        else:
            output_size = tuple(output_size)
        if output_size != (1, 1, 1):
            raise NotImplementedError(
                "adaptive_avg_pool3d currently supports output_size=(1,1,1)"
            )
        n, c, depth, height, width = in_dims
        callee = self.world.annex(torch_dialect.pool.adaptive_avg_pool3d.value)
        callee = self.world.app(callee, self._torch_semantics(x, floating=True))
        callee = self._apply_grouped(callee, [n, c, depth, height, width])
        result = self.world.app(
            callee, self.world.tuple([self._lit_nat(1)] * 3)
        )
        result = self.world.app(result, x)
        return self._remember_shape(
            result, [n, c, self._lit_nat(1), self._lit_nat(1), self._lit_nat(1)]
        )

    def repeat(self, x, repeats):
        in_dims = self.shape_of(x)
        repeats = list(repeats)
        if len(repeats) < len(in_dims):
            raise ValueError(
                f"repeat dimensions ({len(repeats)}) cannot be fewer than "
                f"input dimensions ({len(in_dims)})"
            )
        aligned_dims = [self._lit_nat(1)] * (len(repeats) - len(in_dims)) + in_dims
        repeat_defs = [
            value if isinstance(value, mim.Def) else self._lit_nat(value)
            for value in repeats
        ]
        out_dims = [
            self._nat_binop(core.nat.mul, extent, count)
            for extent, count in zip(aligned_dims, repeat_defs)
        ]

        rank = self._lit_nat(len(repeats))
        in_shape_tuple = self.world.tuple(aligned_dims)
        callee = self.world.annex(torch_dialect.shape.repeat.value)
        elem_t = self._tensor_element_type(x)
        callee = self._apply_grouped(callee, [elem_t, rank, in_shape_tuple])
        callee = self.world.app(callee, self.world.tuple(repeat_defs))
        result = self.world.app(callee, x)
        return self._remember_shape(result, out_dims)

    def pad(self, x, padding, *, mode="constant", value=None):
        if mode != "constant":
            raise NotImplementedError("pad currently supports constant mode only")
        padding = list(padding)
        if not padding or len(padding) % 2 != 0:
            raise ValueError("pad expects a non-empty even-length padding list")
        if not all(isinstance(width, int) and width >= 0 for width in padding):
            raise NotImplementedError(
                "pad currently requires static nonnegative widths"
            )
        in_dims = self.shape_of(x)
        rank = len(in_dims)
        padded_rank = len(padding) // 2
        if padded_rank > rank:
            raise ValueError("pad length must not exceed input rank")

        out_dims = list(in_dims)
        for pad_axis in range(padded_rank):
            axis = rank - 1 - pad_axis
            out_dims[axis] = self._nat_binop(
                core.nat.add,
                self._nat_binop(
                    core.nat.add, out_dims[axis], padding[2 * pad_axis]
                ),
                padding[2 * pad_axis + 1],
            )

        elem_type = self._tensor_element_type(x)
        if value is None:
            value = 0
        if elem_type in (self.F32, self.F64):
            scalar = self._float_lit(elem_type, value)
        elif elem_type == self.I64:
            scalar = self.world.lit_i64(int(value))
        elif elem_type == self.Bool:
            scalar = self.world.lit_bool(bool(value))
        else:
            raise NotImplementedError(f"pad value for element type {elem_type}")

        callee = self.world.annex(torch_dialect.creation.constant_pad.value)
        callee = self._apply_grouped(
            callee,
            [elem_type, self._lit_nat(rank), self._lit_nat(padded_rank),
             self.world.tuple(in_dims)],
        )
        pad_tuple = self.world.tuple([self._lit_nat(width) for width in padding])
        result = self.world.app(callee, self.world.tuple([pad_tuple, scalar]))
        result = self.world.app(result, x)
        return self._remember_shape(result, out_dims)

    def diff(self, input, *, n=1, dim=-1, prepend=None, append=None):
        if append is not None:
            raise NotImplementedError("torch.diff append is not implemented")
        input_dims = self.shape_of(input)
        rank_val = len(input_dims)
        canonical_dim = dim + rank_val if dim < 0 else dim
        if canonical_dim < 0 or canonical_dim >= rank_val:
            raise ValueError(f"torch.diff dim {dim} is out of range for rank {rank_val}")
        if prepend is None:
            prepend = self.slice(input, canonical_dim, 0, 0)
        prepend_dims = self.shape_of(prepend)
        if len(prepend_dims) != rank_val:
            raise ValueError("torch.diff prepend rank must match input rank")

        physical_axes = [
            axis
            for axis, extent in enumerate(input_dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and input_dims:
            physical_axes = [len(input_dims) - 1]
        if canonical_dim not in physical_axes:
            raise NotImplementedError("torch.diff over a folded singleton axis")
        physical_input_dims = [input_dims[axis] for axis in physical_axes]
        physical_prepend_dims = [prepend_dims[axis] for axis in physical_axes]
        physical_dim = physical_axes.index(canonical_dim)

        output_dims = list(input_dims)
        combined = self._nat_binop(
            core.nat.add, input_dims[canonical_dim], prepend_dims[canonical_dim]
        )
        output_dims[canonical_dim] = self._nat_binop(core.nat.sub, combined, 1)

        elem_type = self._tensor_element_type(input)
        name = "scan.diff_i64" if elem_type == self.I64 else "scan.diff"
        callee = self.world.annex(self._torch_annex_id(name))
        if elem_type != self.I64:
            callee = self.world.app(callee, self._torch_semantics(input))
        callee = self._apply_grouped(
            callee,
            [
                self._lit_nat(len(physical_input_dims)),
                self.world.tuple(physical_input_dims),
                self.world.tuple(physical_prepend_dims),
            ],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([self._lit_nat(n), self.world.lit_i64(physical_dim)]),
        )
        result = self.world.app(callee, self.world.tuple([input, prepend]))
        return self._remember_shape(result, output_dims)

    def cumsum(self, input, dim, *, dtype=None, reverse=False):
        elem_type = self._tensor_element_type(input)
        boolean_input = elem_type == self.Bool
        if boolean_input:
            if dtype not in (None, torch.int64, torch.long):
                raise NotImplementedError(
                    f"boolean cumsum only supports the default int64 result, got {dtype}"
                )
        elif dtype not in (None, torch.float32, torch.float):
            raise NotImplementedError(
                f"floating cumsum dtype conversion to {dtype} is not implemented"
            )
        dims = self.shape_of(input)
        rank_val = len(dims)
        canonical_dim = dim + rank_val if dim < 0 else dim
        if canonical_dim < 0 or canonical_dim >= rank_val:
            raise ValueError(f"cumsum dim {dim} is out of range for rank {rank_val}")
        physical_axes = [
            axis
            for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and dims:
            physical_axes = [len(dims) - 1]
        if canonical_dim not in physical_axes:
            if boolean_input:
                raise NotImplementedError("boolean cumsum over a folded singleton axis")
            return input
        physical_dims = [dims[axis] for axis in physical_axes]
        physical_dim = physical_axes.index(canonical_dim)
        if reverse and (boolean_input or len(physical_dims) != 2):
            raise NotImplementedError(
                "reverse cumsum currently requires a rank-2 floating tensor"
            )
        if len(physical_dims) not in (1, 2):
            raise NotImplementedError("cumsum currently supports physical rank 1 or 2")
        scan_dims = (
            physical_dims
            if len(physical_dims) == 2
            else [self._lit_nat(1), physical_dims[0]]
        )
        scan_dim = physical_dim if len(physical_dims) == 2 else 1
        if boolean_input:
            callee = self.world.annex(torch_dialect.scan.cumsum_bool_i64.value)
            callee = self._apply_grouped(callee, scan_dims)
            callee = self.world.app(callee, self.world.lit_i64(scan_dim))
        else:
            op = (
                torch_dialect.scan.cumsum_2d_direction
                if reverse
                else torch_dialect.scan.cumsum_2d
            )
            callee = self.world.annex(op.value)
            callee = self.world.app(
                callee, self._torch_semantics(input, floating=True)
            )
            callee = self._apply_grouped(callee, scan_dims)
            if reverse:
                callee = self.world.app(
                    callee,
                    self.world.tuple(
                        [self.world.lit_i64(scan_dim), self.world.lit_tt()]
                    ),
                )
            else:
                callee = self.world.app(callee, self.world.lit_i64(scan_dim))
        result = self.world.app(callee, input)
        result = self._remember_shape(result, dims)
        if not reverse:
            self._cumsum_provenance[result] = (input, canonical_dim, dtype)
        return result

    def _exclusive_cumsum_2d(self, input, logical_dim, dtype=None):
        if dtype not in (None, torch.float32, torch.float):
            raise NotImplementedError(
                "exclusive cumsum dtype conversion is not implemented"
            )
        dims = self.shape_of(input)
        physical_axes = [
            axis
            for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [dims[axis] for axis in physical_axes]
        if len(physical_dims) != 2 or logical_dim not in physical_axes:
            raise NotImplementedError(
                "exclusive cumsum currently requires physical rank 2"
            )
        callee = self.world.annex(torch_dialect.scan.cumsum_exclusive_2d.value)
        callee = self.world.app(
            callee, self._torch_semantics(input, floating=True)
        )
        callee = self._apply_grouped(callee, physical_dims)
        callee = self.world.app(
            callee, self.world.lit_i64(physical_axes.index(logical_dim))
        )
        return self._remember_shape(self.world.app(callee, input), dims)

    def cumprod(self, input, dim, *, dtype=None):
        if dtype is not None:
            raise NotImplementedError("cumprod dtype conversion is not implemented")
        dims = self.shape_of(input)
        rank = len(dims)
        canonical_dim = dim + rank if dim < 0 else dim
        if canonical_dim < 0 or canonical_dim >= rank:
            raise ValueError(f"cumprod dim {dim} is out of range for rank {rank}")
        physical_axes = [
            axis
            for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if not physical_axes and dims:
            physical_axes = [rank - 1]
        if canonical_dim not in physical_axes:
            return input
        physical_dims = [dims[axis] for axis in physical_axes]
        physical_dim = physical_axes.index(canonical_dim)
        if len(physical_dims) not in (1, 2):
            raise NotImplementedError("cumprod currently supports physical rank 1 or 2")
        scan_dims = (
            physical_dims
            if len(physical_dims) == 2
            else [self._lit_nat(1), physical_dims[0]]
        )
        scan_dim = physical_dim if len(physical_dims) == 2 else 1
        callee = self.world.annex(torch_dialect.scan.cumprod_2d.value)
        callee = self.world.app(callee, self._torch_semantics(input, floating=True))
        callee = self._apply_grouped(callee, scan_dims)
        result = self.world.app(callee, self.world.lit_i64(scan_dim))
        result = self.world.app(result, input)
        return self._remember_shape(result, dims)

    def roll(self, input, shifts, dims=None):
        # PyTorch flattens the tensor when dims is omitted. Keeping that case
        # explicit avoids hiding a reshape policy in the Python bridge.
        if dims is None:
            raise NotImplementedError("roll without dims is not implemented")
        shift_values = [shifts] if isinstance(shifts, int) else list(shifts)
        dim_values = [dims] if isinstance(dims, int) else list(dims)
        if len(shift_values) != len(dim_values):
            raise ValueError("roll expects shifts and dims to have equal length")
        if not all(isinstance(value, int) for value in shift_values + dim_values):
            raise NotImplementedError("roll currently requires static shifts and dims")

        shape = self.shape_of(input)
        rank = len(shape)
        normalized_shifts = []
        for shift, dim in zip(shift_values, dim_values):
            canonical = dim + rank if dim < 0 else dim
            if canonical < 0 or canonical >= rank:
                raise IndexError(f"roll dim {dim} is out of range for rank {rank}")
            extent = shape[canonical]
            if not isinstance(extent, mim.Lit):
                raise NotImplementedError(
                    "roll currently requires static extents on rolled dimensions"
                )
            extent_value = extent.get_nat()
            normalized_shifts.append(0 if extent_value == 0 else shift % extent_value)

        callee = self.world.annex(torch_dialect.indexing.roll.value)
        callee = self._apply_grouped(
            callee,
            [self._tensor_element_type(input), self._lit_nat(rank),
             self._lit_nat(len(dim_values)), self.world.tuple(shape)],
        )
        params = self.world.tuple([
            self.world.tuple([self._lit_nat(value) for value in normalized_shifts]),
            self.world.tuple([self.world.lit_i64(value) for value in dim_values]),
        ])
        result = self.world.app(self.world.app(callee, params), input)
        return self._remember_shape(result, shape)

    def unfold(self, input, dimension, size, step):
        if not all(isinstance(value, int) for value in (dimension, size, step)):
            raise NotImplementedError(
                "unfold currently requires static dimension, size, and step"
            )
        shape = self.shape_of(input)
        rank = len(shape)
        if rank == 0:
            raise NotImplementedError("unfold on a scalar tensor is not implemented")
        canonical = dimension + rank if dimension < 0 else dimension
        if canonical < 0 or canonical >= rank:
            raise IndexError(
                f"unfold dimension {dimension} is out of range for rank {rank}"
            )
        if step <= 0:
            raise ValueError("unfold step must be positive")
        extent = shape[canonical]
        if isinstance(extent, mim.Lit) and size > extent.get_nat():
            raise ValueError("unfold size must not exceed the selected extent")

        windows = self._nat_binop(
            core.nat.add,
            self._nat_binop(
                core.nat.div,
                self._nat_binop(core.nat.sub, extent, size),
                step,
            ),
            1,
        )
        output_shape = list(shape)
        output_shape[canonical] = windows
        output_shape.append(self._lit_nat(size))

        callee = self.world.annex(torch_dialect.indexing.unfold.value)
        callee = self._apply_grouped(
            callee,
            [self._tensor_element_type(input), self._lit_nat(rank),
             self.world.tuple(shape)],
        )
        params = self.world.tuple([
            self.world.lit_i64(dimension), self._lit_nat(size), self._lit_nat(step)
        ])
        result = self.world.app(self.world.app(callee, params), input)
        return self._remember_shape(result, output_shape)

    def gather(self, input, index, dim=0):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        rank_val = len(input_dims)
        elem_t = self._tensor_element_type(input)
        rank = self._lit_nat(rank_val)
        dimension = dim if isinstance(dim, mim.Def) else self.world.lit_i64(dim)
        callee = self.world.annex(torch_dialect.indexing.gather.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_t,
                rank,
                self.world.tuple(input_dims),
                self.world.tuple(index_dims),
            ],
        )
        callee = self.world.app(callee, dimension)
        result = self.world.app(callee, [input, index])
        return self._remember_shape(result, index_dims)

    def index_tensor(self, input, index):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        if not index_dims:
            return self.select(input, 0, index)
        if len(input_dims) < 1:
            raise NotImplementedError("aten.index.Tensor requires tensor input")
        if len(input_dims) == 2 and self._tensor_element_type(index) == self.I64:
            # The common weight[position_ids] form is exactly inference
            # embedding semantics, including checked I64-to-index conversion.
            return self.embedding(input, index)

        # PyTorch `x[idx]` replaces dim 0 by every index dimension and preserves
        # trailing input dimensions. Align both operands to the output rank so
        # the tensor dialect's same-rank gather can express that rule.
        output_dims = index_dims + input_dims[1:]
        prefix_dims = index_dims[:-1]
        input_aligned_dims = [self._lit_nat(1)] * len(prefix_dims) + input_dims
        input = self.reshape(input, input_aligned_dims)
        input = self.expand(input, prefix_dims + input_dims)
        gather_index = index
        for _ in input_dims[1:]:
            gather_index = self.unsqueeze(gather_index, -1)
        gather_index = self.expand(gather_index, output_dims)
        return self.gather(input, gather_index, dim=len(prefix_dims))

    def index_2d(self, input, rows, columns):
        input_dims = self.shape_of(input)
        if len(input_dims) != 2:
            raise ValueError("index_2d expects a rank-2 input")
        if self._tensor_element_type(rows) != self.I64:
            raise TypeError("index_2d row indices must be int64")
        if self._tensor_element_type(columns) != self.I64:
            raise TypeError("index_2d column indices must be int64")
        output_dims = self.rules.broadcast_shape(
            self.shape_of(rows), self.shape_of(columns)
        )
        if self.shape_of(rows) != output_dims:
            rows = self.expand(rows, output_dims)
        if self.shape_of(columns) != output_dims:
            columns = self.expand(columns, output_dims)

        callee = self.world.annex(torch_dialect.indexing.index_2d.value)
        callee = self._apply_grouped(
            callee,
            [
                self._tensor_element_type(input),
                input_dims[0],
                input_dims[1],
                self._lit_nat(len(output_dims)),
                self.world.tuple(output_dims),
            ],
        )
        result = self.world.app(callee, self.world.tuple([input, rows, columns]))
        return self._remember_shape(result, output_dims)

    def scatter_src(self, input, dim, index, src):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        src_dims = self.shape_of(src)
        rank_val = len(input_dims)
        elem_t = self._tensor_element_type(input)
        rank = self._lit_nat(rank_val)
        dimension = dim if isinstance(dim, mim.Def) else self.world.lit_i64(dim)
        callee = self.world.annex(torch_dialect.indexing.scatter_src.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_t,
                rank,
                self.world.tuple(input_dims),
                self.world.tuple(index_dims),
                self.world.tuple(src_dims),
            ],
        )
        callee = self.world.app(callee, dimension)
        result = self.world.app(callee, [input, index, src])
        return self._remember_shape(result, input_dims)

    def scatter_value(self, input, dim, index, value):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        elem_t = self._tensor_element_type(input)
        rank = self._lit_nat(len(input_dims))
        dimension = dim if isinstance(dim, mim.Def) else self.world.lit_i64(dim)
        if isinstance(value, mim.Def):
            scalar = value
        elif elem_t == self.I64:
            scalar = self.world.lit_i64(int(value))
        else:
            scalar = self._float_lit(elem_t, value)
        callee = self.world.annex(torch_dialect.indexing.scatter_value.value)
        callee = self._apply_grouped(
            callee,
            [
                elem_t,
                rank,
                self.world.tuple(input_dims),
                self.world.tuple(index_dims),
            ],
        )
        callee = self.world.app(callee, self.world.tuple([dimension, scalar]))
        result = self.world.app(callee, self.world.tuple([input, index]))
        return self._remember_shape(result, input_dims)

    def embedding(self, weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
        weight_dims = self.shape_of(weight)
        index_dims = self.shape_of(indices)
        if len(weight_dims) != 2:
            raise ValueError("aten.embedding expects a rank-2 weight tensor")
        index_type = self._tensor_element_type(indices)
        if index_type != self.I64:
            raise TypeError(
                f"aten.embedding expects int64 indices, got {index_type}"
            )

        physical_index_dims = self._physical_dims(index_dims)
        callee = self.world.annex(torch_dialect.indexing.embedding.value)
        callee = self._apply_grouped(
            callee,
            [
                self._tensor_element_type(weight),
                weight_dims[0],
                weight_dims[1],
                self._lit_nat(len(physical_index_dims)),
            ],
        )
        callee = self._apply_grouped(callee, physical_index_dims)
        callee = self.world.app(
            callee,
            self.world.tuple(
                [
                    self.world.lit_i64(padding_idx),
                    self.world.lit_bool(scale_grad_by_freq),
                    self.world.lit_bool(sparse),
                ]
            ),
        )
        result = self.world.app(callee, self.world.tuple([weight, indices]))
        return self._remember_shape(result, index_dims + [weight_dims[1]])

    def assert_tensor_metadata(self, tensor_def, size=None, stride=None, dtype=None, device=None, layout=None):
        actual_dims = self.shape_of(tensor_def)

        actual_type = self._tensor_element_type(tensor_def)
        expected_type = {
            torch.float32: self.F32,
            torch.int64: self.I64,
        }.get(dtype, actual_type if dtype is None else None)
        if expected_type is None or actual_type != expected_type:
            raise NotImplementedError(
                f"_assert_tensor_metadata dtype {dtype} does not match the tensor element type"
            )
        if device is not None and torch.device(device).type != "cpu":
            raise NotImplementedError("_assert_tensor_metadata currently supports CPU tensors only")
        if layout is not None and layout is not torch.strided:
            raise NotImplementedError("_assert_tensor_metadata currently supports strided layout only")

        if stride is not None:
            expected_stride = []
            running = 1
            for dim in reversed(actual_dims):
                expected_stride.append(running)
                if not isinstance(dim, mim.Lit):
                    raise NotImplementedError(
                        "dynamic _assert_tensor_metadata stride checks are not represented yet"
                    )
                running *= dim.get_nat()
            expected_stride.reverse()
            if list(stride) != expected_stride:
                raise ValueError(
                    f"_assert_tensor_metadata stride mismatch: expected {expected_stride}, got {list(stride)}"
                )

        if size is None:
            return self.world.lit_tt()
        if len(size) != len(actual_dims):
            raise ValueError(
                f"_assert_tensor_metadata rank mismatch: expected {len(size)}, got {len(actual_dims)}"
            )

        expected_dims = [self._lit_nat(dim) if isinstance(dim, int) else dim for dim in size]
        rank = self._lit_nat(len(actual_dims))
        callee = self.world.annex(torch_dialect.metadata.assert_tensor_metadata.value)
        callee = self.world.app(callee, rank)
        return self.world.app(
            callee,
            self.world.tuple([self.world.tuple(actual_dims), self.world.tuple(expected_dims)]),
        )

    # Injective
    def _normalize_reshape_shape(self, x, shape):
        in_dims = self.shape_of(x)
        out_dims = list(shape)
        infer_positions = [i for i, dim in enumerate(out_dims) if isinstance(dim, int) and dim == -1]
        if not infer_positions:
            return out_dims
        if len(infer_positions) > 1:
            raise ValueError("reshape can only infer one dimension")

        known_dims = [dim for dim in out_dims if not (isinstance(dim, int) and dim == -1)]
        inferred = self._nat_binop(core.nat.div, self._nat_product(in_dims), self._nat_product(known_dims))
        out_dims[infer_positions[0]] = inferred
        return out_dims

    def reshape(self, x, shape):
        """
        Translates to `%tensor.reshape`.
        """
        shape = self._normalize_reshape_shape(x, shape)
        in_dims = self.shape_of(x)
        in_rank = self._lit_nat(len(in_dims))
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self._lit_nat(out_rank_val)
        callee = self.world.annex(torch_dialect.shape.reshape.value)
        callee = self._apply_grouped(callee, [self._tensor_element_type(x), in_rank, out_rank])
        callee = self.world.app(callee, [self.world.tuple(in_dims), out_shape_tuple])
        result = self.world.app(callee, x)
        return self._remember_shape(result, list(shape))

        # Legacy tensor implementation retained below for comparison.
        in_rank, in_shape_tuple = self._rank_and_shape(x)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self._lit_nat(out_rank_val)
        elem_t = self._tensor_element_type(x)

        callee = self.world.annex(tensor.reshape.value)
        callee = self._apply_grouped(callee, [elem_t, in_rank, out_rank])
        callee = self.world.app(callee, in_shape_tuple)
        callee = self.world.app(callee, out_shape_tuple)
        result = self.world.app(callee, x)
        return self._remember_shape(result, list(shape))

    def view(self, x, shape):
        return self.reshape(x, shape)

    def flatten(self, x, start_dim=0, end_dim=-1):
        in_dims = self.shape_of(x)
        rank = len(in_dims)
        if start_dim < 0:
            start_dim += rank
        if end_dim < 0:
            end_dim += rank
        if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
            raise ValueError(f"invalid flatten dims start_dim={start_dim}, end_dim={end_dim}, rank={rank}")

        out_dims = in_dims[:start_dim] + [self._nat_product(in_dims[start_dim : end_dim + 1])] + in_dims[end_dim + 1 :]
        return self.reshape(x, out_dims)

    def slice(self, x, dim, start, end, step=1):
        """
        Translates to `%tensor.slice`.
        """
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)

        if dim < 0: dim += rank_val
        
        # 1. Canonical shape transformation
        out_dims = self.rules.slice_shape(in_dims, dim, start, end, step)
        
        # 2. Prep start/step tuples for MimIR
        starts = [self._lit_nat(0)] * rank_val
        steps = [self._lit_nat(1)] * rank_val
        
        starts[dim] = self._lit_nat(start) if isinstance(start, int) else start
        steps[dim] = self._lit_nat(step) if isinstance(step, int) else (self._lit_nat(1) if step is None else step)

        callee = self.world.annex(torch_dialect.indexing.slice.value)
        callee = self._apply_grouped(callee, [self._tensor_element_type(x), self._lit_nat(rank_val)])
        callee = self.world.app(callee, [self.world.tuple(in_dims), self.world.tuple(starts),
                                         self.world.tuple(steps), self.world.tuple(out_dims)])
        result = self.world.app(callee, x)
        return self._remember_shape(result, out_dims)

        # Legacy tensor implementation retained below for comparison.

        callee = self.world.annex(tensor.slice.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self.world.app(callee, in_shape_tuple)
        
        callee = self.world.app(callee, self.world.tuple([
            self.world.tuple(starts),
            self.world.tuple(steps),
            self.world.tuple(out_dims)
        ]))
        result = self.world.app(callee, x)
        return self._remember_shape(result, out_dims)

    def _match_exclusive_cumsum_cat(self, tensors, dims_list, dim):
        if len(tensors) != 2 or tensors[0] not in self._known_zero_tensors:
            return None
        cumsum = self._cumsum_provenance.get(tensors[1])
        if cumsum is None:
            return None
        narrowed, cumsum_dim, dtype = cumsum
        narrow = self._narrow_provenance.get(narrowed)
        if narrow is None:
            return None
        original, narrow_dim, start, length = narrow
        if cumsum_dim != dim or narrow_dim != dim or start != 0:
            return None

        original_dims = self.shape_of(original)
        extent = self.rules._dim_literal_value(original_dims[dim])
        length_value = (
            length
            if isinstance(length, int)
            else self.rules._dim_literal_value(length)
        )
        if extent is None or length_value != extent - 1:
            return None
        if not self._is_lit_nat_value(dims_list[0][dim], 1):
            return None
        return self._exclusive_cumsum_2d(original, dim, dtype=dtype)

    def cat(self, tensors, dim=0):
        """
        Translates to `%torch.shape.cat`.

        A one-element concatenation is the identity and is eliminated eagerly.
        PyTorch also treats a one-dimensional empty tensor as a concatenation
        identity, even when the other inputs have a different rank. Eliminate
        that statically decidable special case before forming the uniformly
        ranked `%torch.shape.cat` input tuple.
        """
        if isinstance(tensors, mim.Tuple):
            num_inputs = tensors.num_projs()
            tensors = [tensors.proj(num_inputs, i) for i in range(num_inputs)]

        input_dims_list = [self.shape_of(t) for t in tensors]
        ordinary = [
            (tensor, dims)
            for tensor, dims in zip(tensors, input_dims_list)
            if not (len(dims) == 1 and self._is_lit_nat_value(dims[0], 0))
        ]
        if ordinary:
            tensors = [tensor for tensor, _ in ordinary]
            input_dims_list = [dims for _, dims in ordinary]

        num_inputs = len(tensors)
        if num_inputs == 1:
            return tensors[0]
        first_tensor = tensors[0]
        elem_t = self._tensor_element_type(first_tensor)

        logical_rank = len(input_dims_list[0])
        if dim < 0:
            dim += logical_rank
        if dim < 0 or dim >= logical_rank:
            raise ValueError("cat dimension is out of range")
        exclusive = self._match_exclusive_cumsum_cat(
            tensors, input_dims_list, dim
        )
        if exclusive is not None:
            return exclusive
        out_dims = self.rules.concat_shape(input_dims_list, dim)

        rank = self._lit_nat(logical_rank)
        input_shapes = [
            self.world.tuple(input_dims) for input_dims in input_dims_list
        ]
        callee = self.world.annex(torch_dialect.shape.cat.value)
        callee = self._apply_grouped(
            callee, [elem_t, self._lit_nat(num_inputs), rank]
        )
        callee = self.world.app(
            callee, self.world.lit(self.world.type_idx(rank), dim)
        )
        callee = self.world.app(callee, self.world.tuple(input_shapes))
        result = self.world.app(callee, self.world.tuple(tensors))
        return self._remember_shape(result, out_dims)

    def stack(self, tensors, dim=0):
        """Translate PyTorch ``stack`` directly to ``%torch.shape.stack``."""
        if isinstance(tensors, mim.Tuple):
            count = tensors.num_projs()
            tensors = [tensors.proj(count, i) for i in range(count)]
        else:
            tensors = list(tensors)
        if not tensors:
            raise ValueError("stack expects a non-empty tensor list")

        input_dims = self.shape_of(tensors[0])
        elem_t = self._tensor_element_type(tensors[0])
        for tensor in tensors[1:]:
            if self.shape_of(tensor) != input_dims:
                raise ValueError("stack expects each tensor to have equal shape")
            if self._tensor_element_type(tensor) != elem_t:
                raise ValueError("stack expects each tensor to have equal dtype")

        output_rank = len(input_dims) + 1
        actual_dim = dim + output_rank if dim < 0 else dim
        if actual_dim < 0 or actual_dim >= output_rank:
            raise IndexError("stack dimension is out of range")

        singleton_dims = list(input_dims)
        singleton_dims.insert(actual_dim, self._lit_nat(1))
        output_dims = list(singleton_dims)
        output_dims[actual_dim] = self._lit_nat(len(tensors))

        rank_in = self._lit_nat(len(input_dims))
        rank_out = self._lit_nat(output_rank)
        callee = self.world.annex(torch_dialect.shape.stack.value)
        callee = self._apply_grouped(
            callee,
            [elem_t, self._lit_nat(len(tensors)), rank_in, rank_out],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([
                self.world.tuple(input_dims),
                self.world.tuple(singleton_dims),
                self.world.lit(self.world.type_idx(rank_out), actual_dim),
            ]),
        )
        result = self.world.app(callee, self.world.tuple(tensors))
        return self._remember_shape(result, output_dims)

    def transpose(self, x, permutation):
        """
        Translates to `%tensor.transpose`.
        """
        rank_val, in_shape_tuple = self._rank_and_shape(x)
        in_dims = self.shape_of(x)
        elem_t = self._tensor_element_type(x)
        
        # Canonical shape transformation
        out_dims = self.rules.transpose_shape(in_dims, permutation)
        
        idx_t = self.world.type_idx(rank_val)
        perm_mim = self.world.tuple([self.world.lit(idx_t, p) for p in permutation])

        callee = self.world.annex(torch_dialect.shape.permute.value)
        callee = self._apply_grouped(callee, [elem_t, rank_val, in_shape_tuple])
        result = self.world.app(callee, [x, perm_mim])
        return self._remember_shape(result, out_dims)
        
        callee = self.world.annex(tensor.transpose.value)
        callee = self._apply_grouped(callee, [elem_t, rank_val, in_shape_tuple])
        result = self.world.app(callee, [x, perm_mim])
        return self._remember_shape(result, out_dims)

    def transpose_int(self, x, dim0, dim1):
        dims = self.shape_of(x)
        rank = len(dims)
        axis0 = dim0 + rank if dim0 < 0 else dim0
        axis1 = dim1 + rank if dim1 < 0 else dim1
        if axis0 < 0 or axis0 >= rank or axis1 < 0 or axis1 >= rank:
            raise ValueError("transpose dimensions are out of range")
        out_dims = list(dims)
        out_dims[axis0], out_dims[axis1] = out_dims[axis1], out_dims[axis0]

        logical_permutation = list(range(rank))
        logical_permutation[axis0], logical_permutation[axis1] = (
            logical_permutation[axis1],
            logical_permutation[axis0],
        )
        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [dims[axis] for axis in physical_axes]
        physical_rank = self._lit_nat(len(physical_dims))
        if axis0 in physical_axes and axis1 in physical_axes:
            callee = self.world.annex(torch_dialect.shape.transpose_int.value)
            callee = self._apply_grouped(
                callee,
                [
                    self._tensor_element_type(x),
                    physical_rank,
                    self.world.tuple(physical_dims),
                ],
            )
            result = self.world.app(
                callee,
                self.world.tuple(
                    [
                        x,
                        self.world.lit_i64(physical_axes.index(axis0)),
                        self.world.lit_i64(physical_axes.index(axis1)),
                    ]
                ),
            )
            return self._remember_shape(result, out_dims)

        physical_output_axes = [
            axis for axis in logical_permutation if axis in physical_axes
        ]
        physical_permutation = [
            physical_axes.index(axis) for axis in physical_output_axes
        ]
        if physical_permutation == list(range(len(physical_axes))):
            return self._remember_shape(x, out_dims)

        callee = self.world.annex(torch_dialect.shape.permute.value)
        callee = self._apply_grouped(
            callee,
            [
                self._tensor_element_type(x),
                physical_rank,
                self.world.tuple(physical_dims),
            ],
        )
        idx_t = self.world.type_idx(physical_rank)
        result = self.world.app(
            callee,
            self.world.tuple(
                [
                    x,
                    self.world.tuple(
                        [self.world.lit(idx_t, axis) for axis in physical_permutation]
                    ),
                ]
            ),
        )
        return self._remember_shape(result, out_dims)

    def _is_one(self, d):
        if isinstance(d, int): return d == 1
        return d == self._lit_nat(1)

    def squeeze(self, x, dim=None):
        """
        Translates to `reshape` with the canonical `squeeze_shape`.
        """
        in_dims = self.shape_of(x)
        out_dims = self.rules.squeeze_shape(in_dims, dim)
        return self.reshape(x, out_dims)

    def unsqueeze(self, x, dim):
        """
        Translates to `reshape` with the canonical `unsqueeze_shape`.
        """
        in_dims = self.shape_of(x)
        out_dims = self.rules.unsqueeze_shape(in_dims, dim)
        return self.reshape(x, out_dims)

    def split(self, x, split_size_or_sections, dim=0):
        """
        Translates to multiple `slice` operations.
        """
        in_dims = self.shape_of(x)
        rank_val = len(in_dims)
        if dim < 0: dim += rank_val
        
        extent = in_dims[dim]
        extent_val = self.rules._dim_literal_value(extent)
        
        output_shapes = self.rules.split_shapes(in_dims, split_size_or_sections, dim)
        slices = []
        curr = 0
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if extent_val is None:
                raise NotImplementedError("Dynamic split by size not supported")
            while curr < extent_val:
                end = min(curr + split_size, extent_val)
                slices.append(self.slice(x, dim, curr, end))
                curr = end
        else:
            for size, out_shape in zip(split_size_or_sections, output_shapes):
                end = curr + size
                part = self.slice(x, dim, curr, end)
                self._remember_shape(part, out_shape)
                slices.append(part)
                curr = end
        
        return tuple(slices)
        
    def select(self, x, dim, index):
        """
        Translates to `slice` followed by `squeeze`.
        """
        # slice(index, index + 1) then squeeze(dim)
        sliced = self.slice(x, dim, index, index + 1, 1)
        result = self.squeeze(sliced, dim)
        return self._remember_shape(result, self.rules.select_shape(self.shape_of(x), dim))

    def clone(self, x):
        """Translate `aten.clone` to its materializing Torch semantics."""
        dims = self.shape_of(x)
        callee = self.world.annex(torch_dialect.creation.clone.value)
        callee = self._apply_grouped(
            callee,
            [self._tensor_element_type(x), self._lit_nat(len(dims)), self.world.tuple(dims)],
        )
        return self._remember_shape(self.world.app(callee, x), dims)

    def copy(self, x): return x
