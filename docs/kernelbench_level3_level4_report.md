# KernelBench Level 3 and Level 4 Baseline

## Level 3

The 50-case corpus was discovered and run with four isolated cache shards:

```text
--suite level3 --size-divisor 16 --timeout 240 --max-fp-iters 512
```

| Status | Cases | Meaning |
| --- | ---: | --- |
| PASS | 12 | All maintained, semantics-preserving fixtures pass eager comparison. |
| INVALID | 36 | No scaled fixture exists; these were not compiled at reduced sizes. |
| FAIL | 2 | Mamba modules fail to import because `einops` is not installed. |

The meaningful maintained-fixture rate is therefore **12/12 (100%)**. The 38
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
| GPT-2, batch 1, sequence 1023 | FAIL | Memory optimization: `%mem.seo` does not converge within 128 iterations. |
| Electra-small, batch 1, sequence 511 | FAIL | Memory optimization: `%mem.seo` does not converge within 512 iterations. |

The earlier semantic blockers have been fixed. GPT-2's `(0,)` KV-cache
concatenation now observes PyTorch's cross-rank empty identity rule, and
Electra's I64 scalar comparisons resolve to integer Torch semantics. Both
graphs now fully leave the Torch dialect before reaching the common memory
optimizer failure. A GPT-2 profile recorded 736 Torch decompositions,
including 236 reshapes, 64 expands, 48 `addmm`s, 24 `bmm`s, and 25 native layer
normalizations.

The full 20-case serial run was also started. It blocked while acquiring the
first GPT-Neo 2.7B weights, with no local cache progress, and was terminated
without assigning results to unexecuted cases. GPT-2 and Electra used their
official downloaded weights; these are model-level failures rather than
synthetic or reduced reproductions.

Current Level 4 conclusions are architectural, not a coverage percentage:

- pretrained loading and eager execution work for the two downloaded models;
- transformer graphs reach the MimIR backend;
- both downloaded graphs pass frontend mapping and Torch-to-tensor
  decomposition; the next shared blocker is `%mem.seo` convergence on the
  generated large control/data-flow graph;
- the remaining five model families need their weights available before a
  valid 20-case pass rate can be reported.
