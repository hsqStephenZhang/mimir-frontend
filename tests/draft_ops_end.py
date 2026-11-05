    # Linear Algebra
    def mm(self, lhs, rhs):
        # Ring: [T: *, _0: T, add: [T, T] -> T, mul: [T, T] -> T]
        ring = self.world.tuple([self.F32, self._f32_float_lit(0.0), self.f32_add_axm, self.f32_mul_axm])
        callee = self.world.annex(tensor.product_2d.value)
        callee = self.world.app(callee, ring)
        return self.world.implicit_app(callee, [lhs, rhs])

    def convolution(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        raise NotImplementedError("aten.convolution is not implemented")

    # Injective
    def reshape(self, x, shape):
        in_rank, in_shape = self._rank_and_shape(x)
        out_shape_tuple, out_rank_val = self._extract_shape(shape)
        out_rank = self.world.lit_nat(out_rank_val)
        elem_t = self._tensor_element_type(x)

        callee = self.world.annex(tensor.reshape.value)
        callee = self._apply_grouped(callee, [elem_t, in_rank, out_rank])
        callee = self.world.app(callee, in_shape)
        callee = self.world.app(callee, out_shape_tuple)
        return self.world.app(callee, x)

    def view(self, x, shape):
        return self.reshape(x, shape)

    def slice(self, x, dim, start, end, step=1):
        rank, in_shape = self._rank_and_shape(x)
        in_dims = self._shape_dims(x)
        rank_val = len(in_dims)
        elem_t = self._tensor_element_type(x)

        if dim < 0: dim += rank_val
        
        starts = []
        steps = []
        out_dims = []
        
        for i in range(rank_val):
            if i == dim:
                s_in = in_dims[i]
                is_static = isinstance(start, int) and (end is None or isinstance(end, int)) and isinstance(step, int)
                
                actual_start = self.world.lit_nat(start) if isinstance(start, int) else start
                actual_step = self.world.lit_nat(step) if isinstance(step, int) else (self.world.lit_nat(1) if step is None else step)
                
                if end is None or (isinstance(end, int) and end > 1000000000):
                    actual_end = s_in
                else:
                    actual_end = self.world.lit_nat(end) if isinstance(end, int) else end
                
                starts.append(actual_start)
                steps.append(actual_step)
                
                if is_static:
                    v_step = step if step is not None else 1
                    if end is None:
                        out_dims.append(self.world.top(world.type_nat()))
                    else:
                        out_dims.append(self.world.lit_nat((end - start + v_step - 1) // v_step))
                else:
                    out_dims.append(self.world.top(world.type_nat()))
            else:
                starts.append(self.world.lit_nat(0))
                steps.append(self.world.lit_nat(1))
                out_dims.append(in_dims[i])
        
        callee = self.world.annex(tensor.slice.value)
        callee = self._apply_grouped(callee, [elem_t, rank])
        callee = self.world.app(callee, in_shape)
        
        callee = self.world.app(callee, self.world.tuple([
            self.world.tuple(starts),
            self.world.tuple(steps),
            self.world.tuple(out_dims)
        ]))
        return self.world.app(callee, x)

    def cat(self, tensors, dim=0):
        num_inputs = tensors.num_projs()
        first_tensor = tensors.proj(num_inputs, 0)
        rank, _ = self._rank_and_shape(first_tensor)
        rank_val = len(self._shape_dims(first_tensor))
        elem_t = self._tensor_element_type(first_tensor)
        
        if dim < 0: dim += rank_val
        
        callee = self.world.annex(tensor.concat.value)
        callee = self._apply_grouped(callee, [elem_t, self.world.lit_nat(num_inputs), rank])
        
        idx_t = self.world.type_idx(rank)
        ax = self.world.lit(idx_t, dim)
        callee = self.world.app(callee, ax)
        
        input_shapes = []
        for i in range(num_inputs):
            _, s = self._rank_and_shape(tensors.proj(num_inputs, i))
            input_shapes.append(s)
        
        callee = self.world.app(callee, self.world.tuple(input_shapes))
        return self.world.app(callee, tensors)

    def transpose(self, x, permutation):
        rank, shape = self._rank_and_shape(x)
        elem_t = self._tensor_element_type(x)
        idx_t = self.world.type_idx(rank)
        perm_mim = self.world.tuple([self.world.lit(idx_t, p) for p in permutation])
        
        callee = self.world.annex(tensor.transpose.value)
        callee = self._apply_grouped(callee, [elem_t, rank, shape])
        return self.world.app(callee, [x, perm_mim])

    def _is_one(self, d):
        if isinstance(d, int): return d == 1
        return d == self.world.lit_nat(1)

    def squeeze(self, x, dim=None):
        in_dims = self._shape_dims(x)
        if dim is None:
            out_dims = [d for d in in_dims if not self._is_one(d)]
        else:
            if dim < 0: dim += len(in_dims)
            out_dims = []
            for i, d in enumerate(in_dims):
                if i == dim:
                    if not self._is_one(d):
                         out_dims.append(d)
                else:
                    out_dims.append(d)
        return self.reshape(x, out_dims)

    def unsqueeze(self, x, dim):
        in_dims = self._shape_dims(x)
        if dim < 0: dim += len(in_dims) + 1
        out_dims = list(in_dims)
        out_dims.insert(dim, 1)
        return self.reshape(x, out_dims)

    def split(self, x, split_size_or_sections, dim=0):
        in_dims = self._shape_dims(x)
        rank_val = len(in_dims)
        if dim < 0: dim += rank_val
        
        extent = in_dims[dim]
        slices = []
        if isinstance(split_size_or_sections, int):
            split_size = split_size_or_sections
            if isinstance(extent, int):
                curr = 0
                while curr < extent:
                    end = min(curr + split_size, extent)
                    slices.append(self.slice(x, dim, curr, end))
                    curr = end
            else:
                raise NotImplementedError("Dynamic split by size not supported")
        else:
            curr = 0
            for size in split_size_or_sections:
                end = curr + size
                slices.append(self.slice(x, dim, curr, end))
                curr = end
        
        return self.world.tuple(slices)
        
    def select(self, x, dim, index):
        sliced = self.slice(x, dim, index, index + 1, 1)
        return self.squeeze(sliced, dim)

    def clone(self, x): return x
    def copy(self, x): return x
