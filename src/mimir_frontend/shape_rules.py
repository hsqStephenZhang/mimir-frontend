from dataclasses import dataclass

import mim


@dataclass(frozen=True)
class ReduceShapeSpec:
    reduce_dims: list[int]
    kept_dims: list[int]
    output_dims: list[mim.Def]
    loop_dims: list[mim.Def]
    input_projections: list[int]


class ShapeRules:
    """
    Centralized Shape Invariants and Transformation Rules.
    This class implements the shape logic described in shape_invariants.md.
    """
    def __init__(self, world: mim.World):
        self.world = world

    def _dim_literal_value(self, dim: mim.Def):
        if isinstance(dim, mim.Lit) and hasattr(dim, "get_nat"):
            return dim.get_nat()
        return None

    def _is_dim_one(self, dim: mim.Def):
        return self._dim_literal_value(dim) == 1

    def _same_dim(self, lhs: mim.Def, rhs: mim.Def):
        lhs_value = self._dim_literal_value(lhs)
        rhs_value = self._dim_literal_value(rhs)
        if lhs_value is not None and rhs_value is not None:
            return lhs_value == rhs_value
        return lhs == rhs

    def same_shape(self, lhs: list[mim.Def], rhs: list[mim.Def]) -> bool:
        return len(lhs) == len(rhs) and all(self._same_dim(d1, d2) for d1, d2 in zip(lhs, rhs))

    def broadcast_dim(self, lhs_dim: mim.Def, rhs_dim: mim.Def):
        """
        Binary dimension broadcasting rule.
        If both are same, return it. If one is 1, return the other.
        """
        if self._same_dim(lhs_dim, rhs_dim):
            return lhs_dim
        if self._is_dim_one(lhs_dim):
            return rhs_dim
        if self._is_dim_one(rhs_dim):
            return lhs_dim
        
        lhs_value = self._dim_literal_value(lhs_dim)
        rhs_value = self._dim_literal_value(rhs_dim)
        if lhs_value is not None and rhs_value is not None:
            raise NotImplementedError(
                f"broadcast incompatible dimensions: {lhs_value} vs {rhs_value}"
            )
        
        # Conservatively pick lhs for symbolic dims that might be equal at runtime
        return lhs_dim

    def broadcast_shape(self, s1: list[mim.Def], s2: list[mim.Def]) -> list[mim.Def]:
        """
        Standard PyTorch broadcasting rule for two shapes.
        """
        out_reversed = []
        for offset in range(1, max(len(s1), len(s2)) + 1):
            d1 = s1[-offset] if offset <= len(s1) else self.world.lit(self.world.lit_nat_0().type(), 1)
            d2 = s2[-offset] if offset <= len(s2) else self.world.lit(self.world.lit_nat_0().type(), 1)
            out_reversed.append(self.broadcast_dim(d1, d2))
        return list(reversed(out_reversed))

    def normalize_reduce_dims(self, dim, rank: int) -> list[int]:
        if dim is None:
            dims = list(range(rank))
        elif isinstance(dim, int):
            dims = [dim]
        elif isinstance(dim, (list, tuple)):
            dims = list(dim)
        else:
            raise NotImplementedError(f"reduce dim {dim!r} is not supported")

        normalized = []
        for d in dims:
            if not isinstance(d, int):
                raise NotImplementedError(f"reduce dim {d!r} is not supported")
            if d < 0:
                d += rank
            if d < 0 or d >= rank:
                raise ValueError(f"reduce dim {d} out of range for rank {rank}")
            if d not in normalized:
                normalized.append(d)
        return normalized

    def reduce_shape_spec(self, in_shape: list[mim.Def], dim=None, keepdim: bool = False) -> ReduceShapeSpec:
        input_rank = len(in_shape)
        reduce_dims = self.normalize_reduce_dims(dim, input_rank)
        kept_dims = [axis for axis in range(input_rank) if axis not in reduce_dims]
        output_dims = self.reduce_shape(in_shape, reduce_dims, keepdim)

        if keepdim:
            input_projections = [
                input_rank + reduce_dims.index(axis) if axis in reduce_dims else axis
                for axis in range(input_rank)
            ]
        else:
            kept_positions = {axis: pos for pos, axis in enumerate(kept_dims)}
            input_projections = [
                len(kept_dims) + reduce_dims.index(axis)
                if axis in reduce_dims
                else kept_positions[axis]
                for axis in range(input_rank)
            ]

        loop_dims = output_dims + [in_shape[axis] for axis in reduce_dims]
        return ReduceShapeSpec(
            reduce_dims=reduce_dims,
            kept_dims=kept_dims,
            output_dims=output_dims,
            loop_dims=loop_dims,
            input_projections=input_projections,
        )

    def reduce_shape(self, in_shape: list[mim.Def], dims: list[int], keepdim: bool) -> list[mim.Def]:
        """
        Canonical reduction shape transformation.
        """
        out_shape = []
        for i, d in enumerate(in_shape):
            if i in dims:
                if keepdim:
                    out_shape.append(self.world.lit(self.world.lit_nat_0().type(), 1))
            else:
                out_shape.append(d)
        return out_shape

    def squeeze_shape(self, in_shape: list[mim.Def], dim: int = None) -> list[mim.Def]:
        """
        Canonical squeeze transformation.
        """
        if dim is None:
            return [d for d in in_shape if not self._is_dim_one(d)]
        
        rank = len(in_shape)
        actual_dim = dim + rank if dim < 0 else dim
        
        out_shape = []
        for i, d in enumerate(in_shape):
            if i == actual_dim:
                if not self._is_dim_one(d):
                    out_shape.append(d)
            else:
                out_shape.append(d)
        return out_shape

    def unsqueeze_shape(self, in_shape: list[mim.Def], dim: int) -> list[mim.Def]:
        """
        Canonical unsqueeze transformation.
        """
        rank = len(in_shape)
        actual_dim = dim + rank + 1 if dim < 0 else dim
        out_shape = list(in_shape)
        out_shape.insert(actual_dim, self.world.lit(self.world.lit_nat_0().type(), 1))
        return out_shape

    def slice_shape(self, in_shape: list[mim.Def], dim: int, start: int, end: int, step: int) -> list[mim.Def]:
        """
        Canonical slice transformation.
        """
        out_shape = list(in_shape)
        s_in = in_shape[dim]
        s_in_val = self._dim_literal_value(s_in)
        
        if isinstance(start, int) and isinstance(step, int):
            if isinstance(end, int):
                # Even if s_in is dynamic, if start and end are close enough, we might know the size statically
                # e.g., select: start=i, end=i+1 -> size 1
                out_size = (end - start + step - 1) // step
                # if s_in_val is known, we must clamp end
                if s_in_val is not None:
                    actual_end = min(end, s_in_val)
                    out_size = (actual_end - start + step - 1) // step
                
                out_shape[dim] = self.world.lit(self.world.lit_nat_0().type(), max(0, int(out_size)))
            elif s_in_val is not None:
                # end is None, but we know s_in
                out_size = (s_in_val - start + step - 1) // step
                out_shape[dim] = self.world.lit(self.world.lit_nat_0().type(), max(0, int(out_size)))
            else:
                out_shape[dim] = self.world.top_nat()
        else:
            out_shape[dim] = self.world.top_nat()
        return out_shape

    def concat_shape(self, shapes: list[list[mim.Def]], dim: int) -> list[mim.Def]:
        """
        Canonical concat transformation.
        Sum extents along the concat dimension.
        """
        if not shapes: return []
        out_shape = list(shapes[0])
        concat_dim_extent_val = 0
        is_fully_static = True
        
        for s in shapes:
            d = s[dim]
            val = self._dim_literal_value(dim=d)
            if val is not None:
                concat_dim_extent_val += val
            else:
                is_fully_static = False
                break
        
        if is_fully_static:
            out_shape[dim] = self.world.lit(self.world.lit_nat_0().type(), concat_dim_extent_val)
        else:
            out_shape[dim] = self.world.top_nat()
        return out_shape

    def transpose_shape(self, in_shape: list[mim.Def], permutation: list[int]) -> list[mim.Def]:
        return [in_shape[p] for p in permutation]

    def select_shape(self, in_shape: list[mim.Def], dim: int) -> list[mim.Def]:
        rank = len(in_shape)
        actual_dim = dim + rank if dim < 0 else dim
        return [d for i, d in enumerate(in_shape) if i != actual_dim]

    def split_shapes(self, in_shape: list[mim.Def], split_size_or_sections, dim: int) -> list[list[mim.Def]]:
        rank = len(in_shape)
        actual_dim = dim + rank if dim < 0 else dim
        extent = in_shape[actual_dim]
        extent_val = self._dim_literal_value(extent)

        outputs: list[list[mim.Def]] = []
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if extent_val is None:
                raise NotImplementedError("Dynamic split by size not supported")
            curr = 0
            while curr < extent_val:
                end = min(curr + split_size, extent_val)
                part = list(in_shape)
                part[actual_dim] = self.world.lit(self.world.lit_nat_0().type(), end - curr)
                outputs.append(part)
                curr = end
            return outputs

        for size in split_size_or_sections:
            part = list(in_shape)
            part[actual_dim] = self.world.lit(self.world.lit_nat_0().type(), size)
            outputs.append(part)
        return outputs
