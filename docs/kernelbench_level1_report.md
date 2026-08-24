# KernelBench Level 1 Coverage

Validated on 2026-08-24 at frontend commit `f1404a6` and MimIR commit
`5e8537cdec`. The serial runner isolates every case in its own subprocess:

```text
--suite level1 --size-divisor 16 --timeout 240 --max-memory-gb 16 \
  --max-fp-iters 512
```

| Status | Cases | Rate |
| --- | ---: | ---: |
| PASS | 100 | 100.0% of the corpus |
| INVALID | 0 | 0.0% of the corpus |
| FAIL | 0 | 0.0% |

`level1/95_CrossEntropyLoss.py` now uses an I64 class-target fixture and maps
the default rank-2 API to `%torch.loss.cross_entropy_mean_2d`. Its MimIR
implementation composes stable log-softmax, checked gather, negation, and mean
reduction; the complete graph passes eager numerical comparison.

The fixed-point budget is 512 because three grouped/dilated 3D transposed
convolution cases require more than the previous 128 `%mem.seo` iterations;
their numerical comparisons pass with the higher budget.
