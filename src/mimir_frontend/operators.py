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
    def _lit_nat(self, value):
        return self.world.lit_nat(value)

    def __init__(self, world: mim.World):
        self.world = world
        self.rules = ShapeRules(world)
        self.f32_config = world.annex(math.f32.value)
        self.F32 = world.annex(math.F32.value)
        self.I64 = world.type_i64()
        self.Bool = world.type_bool()
        self.mode0 = world.lit_nat_0()
        self.sym_map = {} # Mapping from symbolic name to MimIR Nat variable
        self._shape_cache: dict[mim.Def, list[mim.Def]] = {}

        # The Torch dialect receives scalar behavior through explicit
        # dictionaries.  Keeping these values in the frontend makes the
        # generated graph independent of a hidden global dtype registry.

        
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

        def scalar_unary(op, arg_type, result_type):
            lam = world.mut_lam(arg_type, result_type)
            value = lam.var()
            lam.app(True, op, [value])
            return lam

        def scalar_compare(op, rhs):
            lam = world.mut_lam(self.F32, self.Bool)
            value = lam.var()
            lam.app(True, op, [world.tuple([value, rhs(value)])])
            return lam

        is_nan = scalar_compare(bind_math_axm(_math_cmp.u), lambda value: value)
        is_zero = scalar_compare(
            self.f32_eq_axm, lambda _value: self._f32_float_lit(0.0)
        )
        # `%math.conv.u2f` infers its source index width from the argument.
        from_nat = world.mut_lam(self.world.type_nat(), self.F32)
        nat_value = from_nat.var()
        nat_i64 = world.call(core.bitcast, world.type_i64(), nat_value)
        u2f = world.annex(_math_conv.u2f.value)
        # The source width is inferred from the `I64` argument; the
        # destination floating-point configuration is an implicit argument.
        u2f = world.implicit_app(u2f, self.f32_config)
        from_nat.app(True, u2f, [nat_i64])

        self.torch_arithmetic = world.tuple([
            world.tuple([
                self.F32, world.lit(self.F32, 0),
                self.f32_add_axm, self.f32_mul_axm,
            ]),
            self.f32_sub_axm, self.f32_div_axm,
            self.f32_max_axm, self.f32_min_axm,
            self.f32_eq_axm, self.f32_ne_axm,
            self.f32_lt_axm, self.f32_le_axm,
            self.f32_gt_axm, self.f32_ge_axm,
            is_nan, is_zero,
        ])
        self.torch_floating = world.tuple([
            self.torch_arithmetic,
            self.world.lit(self.F32, 0xFF800000),
            self.f32_exp_axm, self.f32_log_axm, self.f32_tanh_axm,
            self.f32_sqrt_axm, self.f32_rsqrt_axm,
            self.f32_sin_axm, self.f32_cos_axm, from_nat,
            self.f32_pow_axm,
        ])

    def _rank_and_shape(self, tensor_def):
        dims = self.shape_of(tensor_def)
        return self._lit_nat(len(dims)), self.world.tuple(dims)

    def _physical_dims(self, dims):
        """Drop literal singleton axes from a logical tensor shape."""
        return [dim for dim in dims if not self._is_lit_nat_value(dim, 1)]

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
            dims.append(tensor_type.arity())
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

    def _torch_binary(self, name, lhs, rhs, out_type=None):
        """Emit a Torch dialect binary op after frontend broadcasting."""
        if isinstance(rhs, (int, float)) or isinstance(lhs, (int, float)):
            scalar_ops = {
                "add_op": self.f32_add_axm, "sub_op": self.f32_sub_axm,
                "mul_op": self.f32_mul_axm, "div_op": self.f32_div_axm,
                "maximum_op": self.f32_max_axm, "minimum_op": self.f32_min_axm,
                "eq_op": self.f32_eq_axm, "ne_op": self.f32_ne_axm,
                "lt_op": self.f32_lt_axm, "le_op": self.f32_le_axm,
                "gt_op": self.f32_gt_axm, "ge_op": self.f32_ge_axm,
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
        callee = self.world.annex(getattr(torch_dialect, name).value)
        callee = self.world.app(callee, self.torch_arithmetic)
        callee = self._apply_grouped(callee, [rank, shape])
        result = self.world.app(callee, self.world.tuple([lhs, rhs]))
        return self._remember_shape(result, output_dims)

    def _torch_unary(self, name, input, *, floating=False, out_type=None):
        """Emit a Torch dialect unary op with the appropriate dictionary."""
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        rank = self._lit_nat(len(physical_dims))
        shape = self.world.tuple(physical_dims)
        callee = self.world.annex(getattr(torch_dialect, name).value)
        callee = self.world.app(callee, self.torch_floating if floating else self.torch_arithmetic)
        callee = self._apply_grouped(callee, [rank, shape])
        result = self.world.app(callee, input)
        return self._remember_shape(result, dims)

    def _torch_scalar(self, name, input, scalar, *, floating=False):
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        elem_type = self._tensor_element_type(input)
        if elem_type == self.I64:
            if name != "add_scalar_op":
                raise NotImplementedError(
                    f"{name} is not implemented for int64 tensors"
                )
            name = "add_i64_scalar_op"
        callee = self.world.annex(getattr(torch_dialect, name).value)
        if elem_type != self.I64:
            callee = self.world.app(
                callee, self.torch_floating if floating else self.torch_arithmetic
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
            scalar_def = self._f32_float_lit(scalar)
        result = self.world.app(callee, self.world.tuple([input, scalar_def]))
        return self._remember_shape(result, dims)

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
    def add(self, lhs, rhs):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar("add_scalar_op", lhs, rhs)
        if isinstance(lhs, (int, float)):
            return self._torch_scalar("add_scalar_op", rhs, lhs)
        return self._torch_binary("add_op", lhs, rhs)
    def sub(self, lhs, rhs): return self._torch_binary("sub_op", lhs, rhs)
    def mul(self, lhs, rhs):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar("mul_scalar_op", lhs, rhs)
        if isinstance(lhs, (int, float)):
            return self._torch_scalar("mul_scalar_op", rhs, lhs)
        return self._torch_binary("mul_op", lhs, rhs)
    def div(self, lhs, rhs): return self._torch_binary("div_op", lhs, rhs)
    def pow(self, lhs, rhs):
        if isinstance(rhs, (int, float)):
            return self._torch_scalar(
                "pow_tensor_scalar_op", lhs, rhs, floating=True
            )
        return self.binary(self.f32_pow_axm, lhs, rhs)
    
    # Comparison
    def eq(self, lhs, rhs): return self._torch_binary("eq_op", lhs, rhs, out_type=self.Bool)
    def ne(self, lhs, rhs): return self._torch_binary("ne_op", lhs, rhs, out_type=self.Bool)
    def lt(self, lhs, rhs): return self._torch_binary("lt_op", lhs, rhs, out_type=self.Bool)
    def le(self, lhs, rhs): return self._torch_binary("le_op", lhs, rhs, out_type=self.Bool)
    def gt(self, lhs, rhs): return self._torch_binary("gt_op", lhs, rhs, out_type=self.Bool)
    def ge(self, lhs, rhs): return self._torch_binary("ge_op", lhs, rhs, out_type=self.Bool)

    # Extrema
    def maximum(self, lhs, rhs): return self._torch_binary("maximum_op", lhs, rhs)
    def minimum(self, lhs, rhs): return self._torch_binary("minimum_op", lhs, rhs)
    
    def clamp_max(self, x, max_val):
        if isinstance(max_val, (int, float)):
            lam = self._f32_unary_lambda(
                self.f32_min_axm,
                lambda v: [v, self._f32_float_lit(float(max_val))]
            )
            return self.unary(lam, x)
        return self.minimum(x, max_val)

    def clamp_min(self, x, min_val):
        if isinstance(min_val, (int, float)):
            lam = self._f32_unary_lambda(
                self.f32_max_axm,
                lambda v: [v, self._f32_float_lit(float(min_val))]
            )
            return self.unary(lam, x)
        return self.maximum(x, min_val)

    def clamp(self, x, min_val=None, max_val=None):
        res = x
        if min_val is not None:
            res = self.clamp_min(res, min_val)
        if max_val is not None:
            res = self.clamp_max(res, max_val)
        return res

    # Unary
    def exp(self, x): return self._torch_unary("exp_op", x, floating=True)
    def log(self, x): return self._torch_unary("log_op", x, floating=True)
    def tanh(self, x): return self._torch_unary("tanh_op", x, floating=True)
    def sqrt(self, x): return self._torch_unary("sqrt_op", x, floating=True)
    def sin(self, x): return self._torch_unary("sin_op", x, floating=True)
    def cos(self, x): return self._torch_unary("cos_op", x, floating=True)
    def abs(self, x): return self._torch_unary("abs_op", x)
    def neg(self, x): return self._torch_unary("neg_op", x)
    def sigmoid(self, x): return self._torch_unary("sigmoid_op", x, floating=True)
    def silu(self, x): return self._torch_unary("silu_op", x, floating=True)
    def rsqrt(self, x): return self._torch_unary("rsqrt_op", x, floating=True)
    
    def relu(self, x):
        lam = self._f32_unary_lambda(
            self.f32_max_axm,
            lambda v: [self._f32_float_lit(0.0), v],
        )
        return self._torch_unary("relu_op", x)

    def reciprocal(self, x):
        lam = self._f32_unary_lambda(
            self.f32_div_axm,
            lambda v: [self._f32_float_lit(1.0), v],
        )
        return self._torch_unary("reciprocal_op", x, floating=True)

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
        x_dims = self.shape_of(x)
        y_dims = self.shape_of(y)
        output_dims = self.rules.broadcast_shape(self.rules.broadcast_shape(cond_dims, x_dims), y_dims)

        if not self.rules.same_shape(cond_dims, output_dims):
            cond = self.expand(cond, output_dims)
        if not self.rules.same_shape(x_dims, output_dims):
            x = self.expand(x, output_dims)
        if not self.rules.same_shape(y_dims, output_dims):
            y = self.expand(y, output_dims)

        rank = self._lit_nat(len(output_dims))
        shape = self.world.tuple(output_dims)
        elem_type = self._tensor_element_type(x)
        callee = self.world.annex(torch_dialect.where_op.value)
        callee = self._apply_grouped(callee, [elem_type, rank, shape])
        result = self.world.app(callee, self.world.tuple([cond, x, y]))
        return self._remember_shape(result, output_dims)

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
        # map path for this case; `%torch.expand_op` requires an array input.
        if in_rank_val == 0:
            callee = self.world.annex(tensor.map.value)
            callee = self._apply_grouped(callee, [elem_type, self._lit_nat(0), self.world.tuple([])])
            lam = self.world.mut_lam(self.world.sigma([]), elem_type)
            lam.set_body(True, input)
            callee = self.world.app(callee, lam)
            callee = self.world.app(callee, self.world.tuple([out_rank, out_shape_tuple]))
            result = self.world.app(callee, self.world.tuple([]))
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
        callee = self.world.annex(torch_dialect.expand_op.value)
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
        else:
            raise NotImplementedError(f"full with dtype {dtype} is not implemented")

        callee = self.world.annex(torch_dialect.full_op.value)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        callee = self._apply_grouped(callee, [elem_type, self._lit_nat(out_rank_val)])
        callee = self.world.app(callee, [out_shape_tuple, scalar_def])
        return self._remember_shape(callee, shape)
            
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
        callee = self.world.annex(torch_dialect.empty_strided_op.value)
        callee = self._apply_grouped(callee, [self.F32, self._lit_nat(rank_value)])
        result = self.world.app(callee, self.world.tuple([shape_tuple, stride_tuple]))
        return self._remember_shape(result, list(shape))

    def fill_scalar(self, input, value):
        dims = self.shape_of(input)
        elem_type = self._tensor_element_type(input)
        if elem_type == self.F32:
            scalar = self._f32_float_lit(float(value))
        elif elem_type == self.Bool:
            scalar = self.world.lit_tt() if value else self.world.lit_ff()
        else:
            raise NotImplementedError("fill.Scalar currently supports float32 and bool tensors")
        callee = self.world.annex(torch_dialect.fill_scalar_op.value)
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
        callee = self.world.annex(torch_dialect.arange_i64_op.value)
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

    def softmax(self, input, dim=-1):
        """Emit API-level softmax; stabilization and reduction live in MimIR."""
        dims = self.shape_of(input)
        physical_dims = self._physical_dims(dims)
        callee = self.world.annex(torch_dialect.softmax_op.value)
        callee = self.world.app(callee, self.torch_floating)
        callee = self._apply_grouped(
            callee,
            [self._lit_nat(len(physical_dims)), self.world.tuple(physical_dims)],
        )
        callee = self.world.app(
            callee,
            self.world.tuple([self.world.lit_i64(dim), self.world.lit_ff()]),
        )
        result = self.world.app(callee, input)
        return self._remember_shape(result, dims)

    def triu(self, input, diagonal=0):
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
            raise NotImplementedError(f"triu with element type {elem_type} is not implemented")
        if isinstance(diagonal, int):
            diagonal = self.world.lit(self.I64, diagonal)

        callee = self.world.annex(torch_dialect.triu_op.value)
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

    def native_layer_norm(self, input, normalized_shape, weight=None, bias=None, eps=1e-5):
        """Lower the common LayerNorm case through ordinary reduction operators."""
        input_dims = self.shape_of(input)
        normalized_shape = tuple(int(d) for d in normalized_shape)
        rn = len(normalized_shape)
        if rn == 0 or rn > len(input_dims):
            raise ValueError("normalized_shape must be a non-empty suffix of input shape")
        if rn != 1:
            raise NotImplementedError("LayerNorm currently requires a one-dimensional normalized_shape")
        dim = len(input_dims) - 1
        count = normalized_shape[0]
        mean = self.div(self.sum(input, dim=dim, keepdim=True), count)
        mean_square = self.div(self.sum(self.mul(input, input), dim=dim, keepdim=True), count)
        variance = self.sub(mean_square, self.mul(mean, mean))
        centered = self.sub(input, mean)
        rstd = self.rsqrt(self.add(variance, eps))
        result = self.mul(centered, rstd)
        if weight is not None:
            result = self.mul(result, self._tensor_reshape(weight, [1, normalized_shape[0]]))
        if bias is not None:
            result = self.add(result, self._tensor_reshape(bias, [1, normalized_shape[0]]))
        return self._remember_shape(result, input_dims)


    def _torch_reduce(self, kind, input, dim, keepdim):
        dims = self.shape_of(input)
        logical_rank = len(dims)
        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        physical_dims = [dims[axis] for axis in physical_axes]
        rank = len(physical_dims)
        if dim is None and kind == "sum" and not keepdim:
            callee = self.world.annex(torch_dialect.sum_all_op.value)
            callee = self.world.app(callee, self.torch_arithmetic)
            callee = self._apply_grouped(
                callee,
                [self._lit_nat(rank), self.world.tuple(physical_dims)],
            )
            result = self.world.app(callee, input)
            return self._remember_shape(result, [])

        if dim is None:
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
        if len(dim_values) == 1 and not keepdim:
            callee = self.world.annex(getattr(torch_dialect, f"{kind}_dim_op").value)
            dictionary = (
                self.torch_floating
                if kind in ("amax", "mean")
                else self.torch_arithmetic
            )
            callee = self.world.app(callee, dictionary)
            callee = self._apply_grouped(callee, [self._lit_nat(rank), shape])
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        if kind == "sum" and len(dim_values) == 1 and keepdim:
            callee = self.world.annex(torch_dialect.sum_dim_keepdim_op.value)
            callee = self.world.app(callee, self.torch_arithmetic)
            callee = self._apply_grouped(callee, [self._lit_nat(rank), shape])
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        if kind == "mean" and len(dim_values) == 1 and keepdim:
            callee = self.world.annex(torch_dialect.mean_dim_keepdim_op.value)
            callee = self.world.app(callee, self.torch_floating)
            callee = self._apply_grouped(
                callee, [self._lit_nat(rank), shape]
            )
            callee = self.world.app(callee, dim_values[0])
            result = self.world.app(callee, input)
            return self._remember_shape(result, output_dims)
        name = (
            f"{kind}_dims_keepdim_op"
            if keepdim
            else f"{kind}_dims_op"
        )
        if keepdim and kind == "amax":
            name = "amax_dims_op"
        callee = self.world.annex(getattr(torch_dialect, name).value)
        dictionary = (
            self.torch_floating
            if kind in ("amax", "mean")
            else self.torch_arithmetic
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

    def var_mean(self, input, dim=None, keepdim=False, correction=0):
        """Emit the Torch var_mean decomposition and restore optional keepdim."""
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
        callee = self.world.annex(torch_dialect.var_mean_dims_op.value)
        callee = self.world.app(callee, self.torch_floating)
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
                self._lit_nat(correction),
            ]),
        )
        result = self.world.app(callee, input)
        reduced_dims = self.rules.reduce_shape_spec(
            dims, dim=canonical, keepdim=False
        ).output_dims
        outputs = []
        for index in range(2):
            output = result.proj(2, index)
            self._remember_shape(output, reduced_dims)
            if keepdim:
                keep_dims = self.rules.reduce_shape_spec(
                    dims, dim=canonical, keepdim=True
                ).output_dims
                output = self.reshape(output, keep_dims)
            outputs.append(output)
        return self.world.tuple(outputs)

    # Linear Algebra
    def mm(self, lhs, rhs):
        """Translate `aten.mm`, whose operands are required to be matrices."""
        lhs_dims = self.shape_of(lhs)
        rhs_dims = self.shape_of(rhs)
        if len(lhs_dims) != 2 or len(rhs_dims) != 2:
            raise NotImplementedError("aten.mm requires two rank-2 operands")
        if not self.rules._same_dim(lhs_dims[-1], rhs_dims[-2]):
            raise ValueError("aten.mm contracting dimensions must match")

        callee = self.world.annex(torch_dialect.mm_op.value)
        callee = self.world.app(callee, self.torch_arithmetic)
        callee = self._apply_grouped(callee, [lhs_dims[0], lhs_dims[1], rhs_dims[1]])
        result = self.world.app(callee, self.world.tuple([lhs, rhs]))
        return self._remember_shape(result, [lhs_dims[0], rhs_dims[1]])

    def matmul(self, lhs, rhs):
        """Translate matrix and batch-matrix cases of `aten.matmul`."""
        lhs_dims = self.shape_of(lhs)
        rhs_dims = self.shape_of(rhs)
        if len(lhs_dims) < 2 or len(rhs_dims) < 2:
            raise NotImplementedError("aten.matmul vector and scalar cases are not implemented yet")
        if not self.rules._same_dim(lhs_dims[-1], rhs_dims[-2]):
            raise ValueError("aten.matmul contracting dimensions must match")
        if len(lhs_dims) == 2 and len(rhs_dims) == 2:
            return self.mm(lhs, rhs)

        batch_dims = self.rules.broadcast_shape(lhs_dims[:-2], rhs_dims[:-2])
        lhs_target = batch_dims + lhs_dims[-2:]
        rhs_target = batch_dims + rhs_dims[-2:]
        if not self.rules.same_shape(lhs_dims, lhs_target):
            lhs = self.expand(lhs, lhs_target)
        if not self.rules.same_shape(rhs_dims, rhs_target):
            rhs = self.expand(rhs, rhs_target)

        batch_rank = self._lit_nat(len(batch_dims))
        m, k, n = lhs_dims[-2], lhs_dims[-1], rhs_dims[-1]
        callee = self.world.annex(torch_dialect.matmul_op.value)
        callee = self.world.app(callee, self.torch_arithmetic)
        callee = self._apply_grouped(callee, [batch_rank, m, k, n])
        callee = self.world.app(callee, self.world.tuple(batch_dims))
        result = self.world.app(callee, self.world.tuple([lhs, rhs]))
        return self._remember_shape(result, batch_dims + [m, n])

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
        callee = self.world.annex(torch_dialect.linear_op.value)
        callee = self.world.app(callee, self.torch_arithmetic)
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
            "rnn": torch_dialect.rnn_direction_op,
            "gru": torch_dialect.gru_direction_op,
            "lstm": torch_dialect.lstm_direction_op,
        }[kind]
        callee = self.world.annex(axiom.value)
        callee = self.world.app(callee, self.torch_floating)
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

        callee = self.world.annex(torch_dialect.batch_norm_inference_op.value)
        callee = self.world.app(callee, self.torch_floating)
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

    def convolution(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        semantic_in_dims = self.shape_of(x)
        weight_dims = self.shape_of(weight)
        if len(semantic_in_dims) != 4 or len(weight_dims) != 4:
            raise NotImplementedError("aten.convolution currently supports 4D NCHW inputs only")

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

        callee = self.world.annex(torch_dialect.convolution_op.value)
        callee = self.world.app(callee, self.torch_arithmetic)
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

    def _to_nat(self, value):
        return self._lit_nat(value) if isinstance(value, int) else value

    def _pair(self, value, name):
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return tuple(value)
        raise NotImplementedError(f"{name} must be an int or length-2 sequence")

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
        if return_indices:
            raise NotImplementedError("max_pool2d return_indices=True is not implemented")

        in_dims = self.shape_of(x)
        if len(in_dims) != 4:
            raise NotImplementedError("max_pool2d currently supports 4D NCHW inputs only")
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
        callee = self.world.annex(torch_dialect.max_pool2d_op.value)
        callee = self.world.app(callee, self.torch_floating)
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
        callee = self.world.annex(torch_dialect.hardtanh_op.value)
        callee = self.world.app(callee, self.torch_arithmetic)
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
        if ceil_mode:
            raise NotImplementedError("avg_pool2d ceil_mode=True is not implemented")
        if not count_include_pad:
            raise NotImplementedError("avg_pool2d count_include_pad=False is not implemented")
        if divisor_override is not None:
            raise NotImplementedError("avg_pool2d divisor_override is not implemented")
        return self.pool2d(x, kernel_size, stride=stride, padding=padding, dilation=1, mode="avg")

    def repeat(self, x, shape):
        in_dims = self.shape_of(x)
        out_dims = list(shape)
        if len(in_dims) != len(out_dims):
            raise NotImplementedError("repeat currently requires input and output ranks to match")

        in_rank, in_shape_tuple = self._rank_and_shape(x)
        out_shape_tuple, _ = self._extract_shape(out_dims)
        callee = self.world.annex(tensor.repeat.value)
        elem_t = self._tensor_element_type(x)
        callee = self._apply_grouped(callee, [elem_t, in_rank])
        callee = self.world.app(callee, in_shape_tuple)
        callee = self.world.app(callee, out_shape_tuple)
        result = self.world.app(callee, x)
        return self._remember_shape(result, out_dims)

    def gather(self, input, index, dim=0):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        rank_val = len(input_dims)
        if dim < 0:
            dim += rank_val
        if dim < 0 or dim >= rank_val:
            raise ValueError(f"aten.gather dim {dim} is out of range for rank {rank_val}")
        if rank_val != len(index_dims):
            raise NotImplementedError("aten.gather requires index rank to match input rank")
        for axis, (source_extent, index_extent) in enumerate(zip(input_dims, index_dims)):
            if axis == dim:
                continue
            if isinstance(source_extent, mim.Lit) and isinstance(index_extent, mim.Lit):
                if index_extent.get_nat() > source_extent.get_nat():
                    raise ValueError(
                        "aten.gather index extent exceeds input extent on "
                        f"non-gather axis {axis}"
                    )

        elem_t = self._tensor_element_type(input)
        rank = self._lit_nat(rank_val)
        dim_idx = self.world.lit(self.world.type_idx(rank), dim)

        callee = self.world.annex(tensor.gather.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self._apply_grouped(callee, [self.world.tuple(input_dims), self.world.tuple(index_dims)])
        callee = self.world.app(callee, dim_idx)
        result = self.world.app(callee, [input, index])
        return self._remember_shape(result, index_dims)

    def index_tensor(self, input, index):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        if not index_dims:
            return self.select(input, 0, index)
        if len(input_dims) < 1:
            raise NotImplementedError("aten.index.Tensor requires tensor input")

        # PyTorch `x[idx]` indexes dim 0 and preserves trailing input dimensions.
        output_dims = index_dims + input_dims[1:]
        gather_index = index
        for _ in input_dims[1:]:
            gather_index = self.unsqueeze(gather_index, -1)
        if len(output_dims) != len(self.shape_of(gather_index)):
            raise NotImplementedError("aten.index.Tensor rank normalization failed")
        gather_index = self.expand(gather_index, output_dims)
        return self.gather(input, gather_index, dim=0)

    def scatter(self, input, dim, index, src):
        input_dims = self.shape_of(input)
        index_dims = self.shape_of(index)
        rank_val = len(input_dims)
        if dim < 0:
            dim += rank_val
        if dim < 0 or dim >= rank_val:
            raise ValueError(f"aten.scatter dim {dim} is out of range for rank {rank_val}")
        if rank_val != len(index_dims):
            raise NotImplementedError("aten.scatter requires index rank to match input rank")
        src_dims = self.shape_of(src)
        if rank_val != len(src_dims):
            raise ValueError("aten.scatter src and index must have the same rank")
        for axis, (source_extent, index_extent) in enumerate(zip(input_dims, index_dims)):
            if axis == dim:
                continue
            if isinstance(source_extent, mim.Lit) and isinstance(index_extent, mim.Lit):
                if index_extent.get_nat() > source_extent.get_nat():
                    raise ValueError(
                        "aten.scatter index extent exceeds input extent on "
                        f"non-scatter axis {axis}"
                    )
        for axis, (src_extent, index_extent) in enumerate(zip(src_dims, index_dims)):
            if isinstance(src_extent, mim.Lit) and isinstance(index_extent, mim.Lit):
                if index_extent.get_nat() > src_extent.get_nat():
                    raise ValueError(
                        f"aten.scatter index extent exceeds src extent on axis {axis}"
                    )

        elem_t = self._tensor_element_type(input)
        rank = self._lit_nat(rank_val)
        dim_idx = self.world.lit(self.world.type_idx(rank), dim)

        callee = self.world.annex(tensor.scatter.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self._apply_grouped(
            callee,
            [self.world.tuple(input_dims), self.world.tuple(index_dims), self.world.tuple(src_dims)],
        )
        callee = self.world.app(callee, dim_idx)
        result = self.world.app(callee, [input, index, src])
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
        callee = self.world.annex(torch_dialect.embedding_op.value)
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
        callee = self.world.annex(torch_dialect.assert_tensor_metadata_op.value)
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
        callee = self.world.annex(torch_dialect.reshape_op.value)
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

        callee = self.world.annex(torch_dialect.slice_op.value)
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

    def cat(self, tensors, dim=0):
        """
        Translates to `%torch.cat_op`.

        A one-element concatenation is the identity and is eliminated eagerly.
        """
        if isinstance(tensors, mim.Tuple):
            num_inputs = tensors.num_projs()
            tensors = [tensors.proj(num_inputs, i) for i in range(num_inputs)]

        num_inputs = len(tensors)
        if num_inputs == 1:
            return tensors[0]
        first_tensor = tensors[0]
        elem_t = self._tensor_element_type(first_tensor)

        input_dims_list = []
        for t in tensors:
            input_dims = self.shape_of(t)
            input_dims_list.append(input_dims)
        logical_rank = len(input_dims_list[0])
        if dim < 0:
            dim += logical_rank
        if dim < 0 or dim >= logical_rank:
            raise ValueError("cat dimension is out of range")
        out_dims = self.rules.concat_shape(input_dims_list, dim)

        physical_axes = [
            axis for axis, extent in enumerate(input_dims_list[0])
            if not self._is_lit_nat_value(extent, 1)
        ]
        if dim not in physical_axes:
            raise NotImplementedError(
                "cat along a folded singleton axis is not implemented"
            )
        physical_dim = physical_axes.index(dim)
        physical_shapes = [
            self._physical_dims(input_dims) for input_dims in input_dims_list
        ]
        physical_rank = self._lit_nat(len(physical_shapes[0]))
        input_shapes = [
            self.world.tuple(input_dims) for input_dims in physical_shapes
        ]
        callee = self.world.annex(torch_dialect.cat_op.value)
        callee = self._apply_grouped(
            callee, [elem_t, self._lit_nat(num_inputs), physical_rank]
        )
        callee = self.world.app(
            callee, self.world.lit(self.world.type_idx(physical_rank), physical_dim)
        )
        callee = self.world.app(callee, self.world.tuple(input_shapes))
        result = self.world.app(callee, self.world.tuple(tensors))
        return self._remember_shape(result, out_dims)

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

        callee = self.world.annex(torch_dialect.permute_op.value)
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

        physical_axes = [
            axis for axis, extent in enumerate(dims)
            if not self._is_lit_nat_value(extent, 1)
        ]
        if axis0 not in physical_axes or axis1 not in physical_axes:
            return self._remember_shape(x, out_dims)

        physical_dims = [dims[axis] for axis in physical_axes]
        physical_dim0 = physical_axes.index(axis0)
        physical_dim1 = physical_axes.index(axis1)
        callee = self.world.annex(torch_dialect.transpose_int_op.value)
        callee = self._apply_grouped(
            callee,
            [self._tensor_element_type(x), self._lit_nat(len(physical_dims)),
             self.world.tuple(physical_dims)],
        )
        result = self.world.app(
            callee,
            self.world.tuple(
                [x, self.world.lit_i64(physical_dim0), self.world.lit_i64(physical_dim1)]
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
        
        return self.world.tuple(slices)
        
    def select(self, x, dim, index):
        """
        Translates to `slice` followed by `squeeze`.
        """
        # slice(index, index + 1) then squeeze(dim)
        sliced = self.slice(x, dim, index, index + 1, 1)
        result = self.squeeze(sliced, dim)
        return self._remember_shape(result, self.rules.select_shape(self.shape_of(x), dim))

    def clone(self, x): return x
    def copy(self, x): return x
