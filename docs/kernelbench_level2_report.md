# KernelBench Level 2 Coverage

## Validated Baseline

The full 100-case Lighthouse/KernelBench Level 2 corpus was revalidated on
2026-08-24 at frontend commit `9a2ac97` and MimIR commit `b14def34d0` with:

```text
--suite level2 --size-divisor 16 --timeout 240 --max-memory-gb 16 \
  --max-fp-iters 512
```

Each case runs in an isolated subprocess and the run used an isolated writable
`MIMIR_CACHE_DIR`. The validated result is:

| Status | Cases | Rate |
| --- | ---: | ---: |
| PASS | 96 | 96.0% of the corpus |
| FAIL | 1 | 1.0% |
| INVALID | 3 | 3.0% |
| TIMEOUT | 0 | 0.0% |

Excluding fixtures that fail in PyTorch eager before MimIR is invoked, the
compiler pass rate is **96/97 (99.0%)**.

At 128 fixed-point iterations, only 78 cases pass and 18 otherwise-correct
convolution compositions stop in `%mem.seo`. All 18 pass when rerun at 512.
The higher budget is therefore part of the reproducible baseline, not a
case-specific semantic workaround.

## Implemented In This Increment

- `%torch.activation.mish` implements `x * tanh(softplus(x))` in MimIR.
- `%torch.reduction.logsumexp_dims[_keepdim]` implements PyTorch's stable
  max-shift decomposition, including infinite-maximum masking, negative and
  multiple dimensions, and `keepdim`.
- `%torch.normalization.native_group_norm` captures Aten's complete
  `(output, mean, rstd)` result schema, validates the redundant `N`, `C`, and
  `HxW` arguments, and decomposes normalization and optional channel affine
  parameters into tensor operations.
- `%torch.pool.max_pool2d_with_indices` returns both pooled values and PyTorch's
  flattened spatial indices. Its index reduction preserves first-index ties,
  excludes implicit padding, and handles dilation and ceil-mode windows.
- The `torch.compile` ABI accepts lifted rank-zero tensor inputs and passes
  their scalar MimIR representation by value, including broadcast use sites.
- The frontend now maps `torch.multiply` and tensor `detach` aliases directly.
- `torch.min(tensor, tensor)` is dispatched to binary minimum instead of being
  misinterpreted as the value-and-index reduction overload.

Focused MimIR lit, frontend translation tests, numerical `torch.compile` E2E,
and the complete Level 2 corpus validate these changes.

## Remaining Failures

- `level2/28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py` leaves the Torch
  dialect, but `%tensor.map_reduce` reaches the LLVM backend without being
  bufferized. This is a tensor-to-memory lowering defect and must not be hidden
  by a Torch-specific frontend workaround.

The three INVALID fixtures have incompatible shapes after fixture scaling and
also fail under eager execution.
