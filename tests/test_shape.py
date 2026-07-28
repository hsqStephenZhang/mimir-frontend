import mim
import pytest

from mimir_frontend.translator import FXGraphTranslator
from mimir_frontend import expr
from mim._plugins.tensor import tensor

def make_world() -> mim.World:
    driver = mim.Driver()
    driver.load_plugins(["math", "tensor", "torch", "affine"])
    return driver.world()


def make_symbolic_tensor_input(world: mim.World, dims, elem_type=None):
    ops = FXGraphTranslator(world).ops
    if elem_type is None:
        elem_type = ops.F32
    if len(dims) == 1:
        tensor_ty = world.arr(dims[0], elem_type)
    else:
        tensor_ty = world.arr(world.tuple(dims), elem_type)
    return world.mut_con(tensor_ty).var()


def lit_nat_value(dim: mim.Def):
    if isinstance(dim, mim.Lit) and hasattr(dim, "get_nat"):
        return dim.get_nat()
    return None


def assert_dim_is_literal(dim: mim.Def, value):
    assert lit_nat_value(dim) == value


def assert_dims_same(ops, lhs, rhs):
    assert ops.rules._same_dim(lhs, rhs), f"expected dims to be equal: {lhs} vs {rhs}"

def assert_dims_not_same(ops, lhs, rhs):
    assert not ops.rules._same_dim(lhs, rhs), f"expected dims to differ: {lhs} vs {rhs}"

# expand (n, 4) -> (-1, 4)
def test_expand_keeps_existing_symbolic_dim_with_negative_one():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(4)])

    result = ops.expand(x, [-1, 4])
    dims = ops.shape_of(result)

    assert_dims_same(ops, dims[0], n)
    assert_dim_is_literal(dims[1], 4)

# expand (n, m) -> (3, n, m)
def test_expand_introduces_new_leading_literal_dim_and_preserves_tail_symbols():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m])

    result = ops.expand(x, [3, n, m])
    dims = ops.shape_of(result)

    assert_dim_is_literal(dims[0], 3)
    assert_dims_same(ops, dims[1], n)
    assert_dims_same(ops, dims[2], m)

# reduce (n, m, 8) -> (n, 1, 8)
def test_sum_keepdim_preserves_unreduced_symbol_and_inserted_one_dim():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m, world.lit_nat(8)])

    result = ops.sum(x, dim=1, keepdim=True)
    dims = ops.shape_of(result)

    assert len(dims) == 3
    assert_dims_same(ops, dims[0], n)
    assert_dim_is_literal(dims[1], 1)
    assert_dim_is_literal(dims[2], 8)

# sum (n, m, k) -> (n, k)
def test_sum_without_keepdim_preserves_remaining_symbol_order():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    k = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m, k])

    result = ops.sum(x, dim=1, keepdim=False)
    dims = ops.shape_of(result)

    assert len(dims) == 2
    assert_dims_same(ops, dims[0], n)
    assert_dims_same(ops, dims[1], k)


def test_reduce_shape_spec_without_keepdim_captures_map_reduce_aff_shapes():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    k = world.mut_con(world.type_nat()).var()

    spec = ops.rules.reduce_shape_spec([n, m, k], dim=1, keepdim=False)

    assert spec.reduce_dims == [1]
    assert spec.kept_dims == [0, 2]
    assert spec.input_projections == [0, 2, 1]
    assert_dims_same(ops, spec.output_dims[0], n)
    assert_dims_same(ops, spec.output_dims[1], k)
    assert_dims_same(ops, spec.loop_dims[0], n)
    assert_dims_same(ops, spec.loop_dims[1], k)
    assert_dims_same(ops, spec.loop_dims[2], m)


def test_reduce_shape_spec_keepdim_captures_map_reduce_aff_shapes():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()

    spec = ops.rules.reduce_shape_spec([n, m, world.lit_nat(8)], dim=1, keepdim=True)

    assert spec.reduce_dims == [1]
    assert spec.kept_dims == [0, 2]
    assert spec.input_projections == [0, 3, 2]
    assert_dims_same(ops, spec.output_dims[0], n)
    assert_dim_is_literal(spec.output_dims[1], 1)
    assert_dim_is_literal(spec.output_dims[2], 8)
    assert_dims_same(ops, spec.loop_dims[0], n)
    assert_dim_is_literal(spec.loop_dims[1], 1)
    assert_dim_is_literal(spec.loop_dims[2], 8)
    assert_dims_same(ops, spec.loop_dims[3], m)

# transpose (n, m, k) -> (k, m, n)
def test_transpose_permutates_symbolic_dims_without_losing_identity():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(4), m])

    result = ops.transpose(x, [2, 1, 0])
    dims = ops.shape_of(result)

    assert_dims_same(ops, dims[0], m)
    assert_dim_is_literal(dims[1], 4)
    assert_dims_same(ops, dims[2], n)

# slice (n, m, 8) -> (n, Top, 8)
def test_slice_preserves_unsliced_symbolic_dims_but_loses_sliced_dynamic_extent():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m, world.lit_nat(8)])

    result = ops.slice(x, dim=1, start=1, end=None, step=1)
    dims = ops.shape_of(result)

    assert_dims_same(ops, dims[0], n)
    assert_dims_not_same(ops, dims[1], m)
    assert dims[1] == world.top_nat()
    assert_dim_is_literal(dims[2], 8)

# select (n, m, 8) -> (n, 8)
def test_select_removes_selected_dim_and_preserves_remaining_symbols():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m, world.lit_nat(8)])

    result = ops.select(x, dim=1, index=0)
    dims = ops.shape_of(result)

    assert len(dims) == 2
    assert_dims_same(ops, dims[0], n)
    assert_dim_is_literal(dims[1], 8)

# squeeze (1, n, 1, 8) -> (n, 8)
def test_squeeze_removes_only_literal_one_dims_and_preserves_symbols():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [world.lit_nat(1), n, world.lit_nat(1), world.lit_nat(8)])

    result = ops.squeeze(x)
    dims = ops.shape_of(result)

    assert len(dims) == 2
    assert_dims_same(ops, dims[0], n)
    assert_dim_is_literal(dims[1], 8)

# unsqueeze (n, m) -> (n, 1, m)
def test_unsqueeze_inserts_literal_one_and_preserves_existing_symbols():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m])

    result = ops.unsqueeze(x, dim=1)
    dims = ops.shape_of(result)

    assert len(dims) == 3
    assert_dims_same(ops, dims[0], n)
    assert_dim_is_literal(dims[1], 1)
    assert_dims_same(ops, dims[2], m)

# split (n, 8) -> (n, 3) + (n, 5)
def test_split_with_static_sections_preserves_outer_symbols_and_creates_literal_chunk_sizes():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(8)])

    result = ops.split(x, [3, 5], dim=1)

    first = result.proj(2, 0)
    second = result.proj(2, 1)
    first_dims = ops.shape_of(first)
    second_dims = ops.shape_of(second)

    assert_dims_same(ops, first_dims[0], n)
    assert_dim_is_literal(first_dims[1], 3)
    assert_dims_same(ops, second_dims[0], n)
    assert_dim_is_literal(second_dims[1], 5)


def test_cat_leaves_plugin_to_compute_concat_dim_so_shared_symbol_info_is_not_explicitly_proven_here():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(3)])
    y = make_symbolic_tensor_input(world, [n, world.lit_nat(5)])
    pair = world.tuple([x, y])

    result = ops.cat(pair, dim=1)
    dims = ops.shape_of(result)

    assert_dims_same(ops, dims[0], n)
    assert_dims_not_same(ops, dims[1], world.lit_nat(3))
    assert_dims_not_same(ops, dims[1], world.lit_nat(5))


def test_transpose_shape_rule_permutates_symbolic_dims():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()

    out = ops.rules.transpose_shape([n, world.lit_nat(4), m], [2, 1, 0])

    assert_dims_same(ops, out[0], m)
    assert_dim_is_literal(out[1], 4)
    assert_dims_same(ops, out[2], n)


def test_select_shape_rule_removes_selected_dim():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()

    out = ops.rules.select_shape([n, m, world.lit_nat(8)], dim=1)

    assert len(out) == 2
    assert_dims_same(ops, out[0], n)
    assert_dim_is_literal(out[1], 8)


def test_split_shapes_rule_preserves_outer_dims_and_sections():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()

    outputs = ops.rules.split_shapes([n, world.lit_nat(8)], [3, 5], dim=1)

    assert len(outputs) == 2
    assert_dims_same(ops, outputs[0][0], n)
    assert_dim_is_literal(outputs[0][1], 3)
    assert_dims_same(ops, outputs[1][0], n)
    assert_dim_is_literal(outputs[1][1], 5)


def test_concat_result_shape_can_be_read_from_result_type_without_frontend_cache():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(3)])
    y = make_symbolic_tensor_input(world, [n, world.lit_nat(5)])

    rank = world.lit_nat(2)
    callee = world.annex(tensor.concat.value)
    callee = world.app(callee, world.tuple([ops.F32, world.lit_nat(2), rank]))
    callee = world.app(callee, world.lit(world.type_idx(rank), 1))
    callee = world.app(callee, world.tuple([
        world.tuple(ops.shape_of(x)),
        world.tuple(ops.shape_of(y)),
    ]))
    result = world.app(callee, world.tuple([x, y]))

    uncached_ops = FXGraphTranslator(world).ops
    dims = uncached_ops.shape_of(result)

    assert_dims_same(uncached_ops, dims[0], n)
    assert_dim_is_literal(dims[1], 8)


def test_pad_result_shape_can_be_read_from_result_type_without_frontend_cache():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, world.lit_nat(4)])
    lo = world.tuple([world.lit_nat(1), world.lit_nat(2)])
    hi = world.tuple([world.lit_nat(3), world.lit_nat(4)])

    rank = world.lit_nat(2)
    callee = world.annex(tensor.pad.value)
    callee = world.app(callee, world.tuple([ops.F32, rank]))
    callee = world.app(callee, world.tuple(ops.shape_of(x)))
    callee = world.app(callee, world.tuple([world.lit_nat(0), lo, hi]))
    result = world.app(callee, world.tuple([x, ops._f32_float_lit(0.0)]))

    uncached_ops = FXGraphTranslator(world).ops
    dims = uncached_ops.shape_of(result)

    assert len(dims) == 2
    assert_dims_not_same(uncached_ops, dims[0], n)
    assert_dim_is_literal(dims[1], 10)




# reshape (n, m) -> (n * m)
def test_reshape_does_not_infer_product_equality():
    world = make_world()
    ops = FXGraphTranslator(world).ops
    n = world.mut_con(world.type_nat()).var()
    m = world.mut_con(world.type_nat()).var()
    x = make_symbolic_tensor_input(world, [n, m])

    prod = expr.mul(n, m)
    result = ops.reshape(x, [prod])
    dims = ops.shape_of(result)

    assert_dims_same(ops, dims[0], prod)
