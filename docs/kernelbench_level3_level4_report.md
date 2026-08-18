# KernelBench Level 3 and Level 4 Baseline

## Level 3

The 50-case corpus was discovered and run with four isolated cache shards:

```text
--suite level3 --size-divisor 16 --timeout 240 --max-fp-iters 128
```

| Status | Cases | Meaning |
| --- | ---: | --- |
| PASS | 9 | All maintained, semantics-preserving scaled fixtures pass eager comparison. |
| INVALID | 39 | No scaled fixture exists; these were not compiled at reduced sizes. |
| FAIL | 2 | Mamba modules fail to import because `einops` is not installed. |

The meaningful maintained-fixture rate is therefore **9/9 (100%)**. The 41
remaining native fixtures require either full-size execution or additional
model-specific fixtures; counting them as compiler failures would be
misleading.

Structured results are stored in:

```text
/tmp/kernelbench-level3-20260818-shard-{0,1,2,3}.json
```

## Level 4

Level 4 contains 20 full pretrained HuggingFace model cases and has no scaled
fixtures. The runner now exposes it explicitly as `--suite level4`, while the
existing `full` suite remains Level 1-3 to avoid implicit multi-gigabyte model
downloads.

Two complete-weight architecture baselines were executed:

| Model | Result | First blocking stage |
| --- | --- | --- |
| GPT-2, batch 1, sequence 1023 | FAIL | Backend graph construction: `cat dimension is out of range`. |
| Electra-small, batch 1, sequence 511 | FAIL | Tensor dtype dispatch: an F32 unary predicate is applied to I64 input. |

The full 20-case serial run was also started. It blocked while acquiring the
first GPT-Neo 2.7B weights, with no local cache progress, and was terminated
without assigning results to unexecuted cases. GPT-2 and Electra used their
official downloaded weights; these are model-level failures rather than
synthetic or reduced reproductions.

Current Level 4 conclusions are architectural, not a coverage percentage:

- pretrained loading and eager execution work for the two downloaded models;
- transformer graphs reach the MimIR backend;
- integer tensor semantics and output/concatenation shape handling fail before
  the existing operator coverage can be evaluated end to end;
- the remaining five model families need their weights available before a
  valid 20-case pass rate can be reported.
