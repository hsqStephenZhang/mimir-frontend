# KernelBench Level 2 Coverage

## Validated Baseline

The full 100-case Lighthouse/KernelBench Level 2 corpus was run with:

```text
--suite level2 --size-divisor 16 --timeout 240 --max-fp-iters 128
```

Each of four shards used an isolated writable `MIMIR_CACHE_DIR`. The validated
result is:

| Status | Cases | Rate |
| --- | ---: | ---: |
| PASS | 94 | 94.0% of the corpus |
| FAIL | 3 | 3.0% |
| INVALID | 3 | 3.0% |
| TIMEOUT | 0 | 0.0% |

Excluding fixtures that fail in PyTorch eager before MimIR is invoked, the
compiler pass rate is **94/97 (96.9%)**. This improves the preceding baseline
from 70/100 by 24 passing cases.

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
- The frontend now maps `torch.multiply` and tensor `detach` aliases directly.
- `torch.min(tensor, tensor)` is dispatched to binary minimum instead of being
  misinterpreted as the value-and-index reduction overload.

Focused MimIR lit, frontend translation tests, numerical `torch.compile` E2E,
and the complete Level 2 corpus validate these changes.

## Remaining Failures

- 2 cases: rank-zero tensor graph inputs are rejected by the frontend/backend
  boundary.
- 1 case: the fixture triggers a Dynamo graph break in `_warnings.warn` before
  the MimIR backend is called.

The three INVALID fixtures have incompatible shapes after fixture scaling and
also fail under eager execution.
