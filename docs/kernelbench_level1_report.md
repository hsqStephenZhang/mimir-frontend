# KernelBench Level 1 Coverage

Validated on 2026-08-24 at frontend commit `9a2ac97` and MimIR commit
`b14def34d0`. The serial runner isolates every case in its own subprocess:

```text
--suite level1 --size-divisor 16 --timeout 240 --max-memory-gb 16 \
  --max-fp-iters 512
```

| Status | Cases | Rate |
| --- | ---: | ---: |
| PASS | 99 | 99.0% of the corpus |
| INVALID | 1 | 1.0% of the corpus |
| FAIL | 0 | 0.0% |

The invalid case is `level1/95_CrossEntropyLoss.py`; its fixture constructs a
floating-point target, while PyTorch requires a `Long` or `Byte` target. It
fails in eager execution before the MimIR backend is invoked.

The fixed-point budget is 512 because three grouped/dilated 3D transposed
convolution cases require more than the previous 128 `%mem.seo` iterations;
their numerical comparisons pass with the higher budget.
