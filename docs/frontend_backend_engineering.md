# Frontend Backend Engineering Baseline

This document records the engineering changes made after comparing the Torch
frontend with the backend infrastructure on `mimir-frontend`'s `fodinabor/main`
line. The Torch frontend already had platform suffixes, cross-process JIT
caching, source fingerprints, decomposition policies, profiling, and debug
dumps. The missing pieces were artifact lifecycle guarantees, resource-bounded
benchmarking, and a reproducible debug manifest.

## Reviewable Commits

1. `e514f5b fix(frontend): harden JIT artifact lifecycle`

   Invalid cache libraries are removed and rebuilt. Non-debug temporary build
   directories are removed after both successful and failed compilation on
   POSIX. Successful Windows DLL directories remain alive because a loaded DLL
   cannot be unlinked safely. Platform-only cache tests no longer require
   clang, while native lifecycle tests remain in the integration suite.

2. `930c1b5 feat(frontend): add resource-bounded backend benchmark`

   `scripts/benchmark_backends.py` compares eager, Inductor, and MimIR in one
   subprocess per model/backend. It verifies outputs against eager, reports
   first-call and steady-state timings, supports JSON output, and defaults to
   one thread, 16 GiB address space, and a 300-second timeout.

3. `9a2ac97 feat(frontend): record debug compilation manifests`

   `debug_dir` now receives an atomically updated manifest per graph. It records
   shapes, dtypes, decomposition policy, plugin list, fixed-point/profile/cache
   options, fingerprints, artifacts, and success/failure diagnostics. Tensor
   values and model weights are never serialized.

## Level 1 Closure

MimIR commit `5e8537cdec` adds the API-level
`%torch.loss.cross_entropy_mean_2d` axiom and decomposes it through stable
log-softmax, checked gather, negation, and mean reduction. Frontend commit
`f1404a6` maps the default PyTorch API, adds typed KernelBench fixtures, and
provides direct-mapping and numerical E2E tests. This changes the Level 1 result
from 99 PASS plus one invalid fixture to 100/100 executable PASS.

## Validation

- Backend infrastructure and focused native lifecycle/manifest tests pass.
- The benchmark runner has 9 unit tests; an eager/MimIR MLP smoke run passed
  numerical comparison under a 16 GiB subprocess limit.
- KernelBench Level 1: 100 PASS, 0 invalid, 0 compiler failures.
- KernelBench Level 2 at 512 fixed-point iterations: 96 PASS, 3 eager-invalid,
  1 tensor-to-memory failure.
- KernelBench Level 3: all 27 maintained fixtures pass; 21 native models have no
  semantics-preserving scaled fixture and 2 Mamba files lack `einops`.

The complete frontend pytest suite is not yet a release-green gate. Its first
known operator-semantic failure is `avg_pool2d` with `ceil_mode`, where Dynamo's
output metadata and the imported implementation disagree on the output shape.
This predates the infrastructure changes and should be fixed as a separate,
test-driven operator commit.

## Reproduction

```bash
PYTHONPATH=src:/path/to/MimIR/build/mim_py_stage/main/src \
  .venv/bin/python scripts/run_kernelbench_mimir.py \
  --suite level2 --size-divisor 16 --max-fp-iters 512 \
  --max-memory-gb 16 --timeout 240 --results-json level2.json
```

Use `--resume` with the same JSON file to continue interrupted runs. Set an
isolated writable `MIMIR_CACHE_DIR` when comparing compiler revisions.
