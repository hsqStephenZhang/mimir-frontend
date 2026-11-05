# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` holds the project owner's standing instructions (answer in Chinese, always go through `uv`,
TDD, no cheating tests, report hard problems instead of working around them). Follow it; this file adds
the build/test mechanics and the architecture.

## Setup and build

- First build (clones submodules, fetches the two plugins into `MimIR/extra/`, builds MimIR's `mim_py`
  target, creates `.venv`, runs `uv sync`): `./scripts/bootstrap_mimir.sh`
- On macOS Apple clang's libc++ headers are incomplete; export
  `CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++"` before the first
  configure (`MimIR/build` caches the compiler; if it was configured with the wrong one, move the build
  dir away and rerun bootstrap). `MIMIR_BUILD_TYPE=Debug` and `MIMIR_BUILD_JOBS=N` are honoured.
- After editing MimIR or a plugin under `MimIR/extra/`, rerun `./scripts/bootstrap_mimir.sh` (it rebuilds
  `mim_py` and resyncs the `mim` wheel, which lives at `MimIR/build/mim_py_stage/main`). The JIT cache is
  keyed on a fingerprint of the MimIR install, so a rebuild invalidates it automatically.
- The binding generator needs `clang.cindex`, so `libclang` is a dev dependency (`uv sync` installs it).
- Plugin revisions are pinned in the bootstrap script (`MIMIR_TORCH_PLUGIN_REF`, default `nominal`;
  `MIMIR_RUNTIME_PLUGIN_REF`, default `feat/ll-emit`). `operators.py` looks annexes up by name
  (`mim._plugins.torch`), so the frontend and the pinned `torch.mim` must agree on the axiom set.

## Running things

Always `uv run --no-sync ...`; never call the venv Python or `pip` directly.

```bash
uv run --no-sync pytest -q                                   # unit + integration suite
uv run --no-sync pytest tests/test_shape.py -q               # one file
uv run --no-sync pytest tests/test_torch_compile_backend.py -k addmm_beta_zero   # one test
uv run --no-sync pytest -m "not slow"                        # skip whole-model E2E tests
uv run --no-sync python models/py/mlp.py                     # compile+check one example model
uv run --no-sync python scripts/export_models_to_mimir.py    # write models/mim/*.mim from export specs
uv run --no-sync python scripts/benchmark_backends.py models/py/mlp.py --repeat 5 --threads 1
uv run --no-sync python scripts/run_kernelbench_mimir.py --suite level1 --size-divisor 16 \
    --timeout 300 --max-memory-gb 12 --max-fp-iters 8192 --allow-invalid --results-json out.json
```

- Tests that JIT native code need `clang` on `PATH` (they skip otherwise). `tests/test_real_inductor_graphs.py`
  needs `MIMIR_INDUCTOR_LOG_ROOT` pointing at a local Inductor dump directory (`*/fx_graph_readable.py`).
  `pretrained`-marked tests download weights and only run with `MIMIR_RUN_PRETRAINED=1`.
- Set `MIMIR_CACHE_DIR` to keep an experiment's JIT cache away from `~/.cache/mimir-frontend/jit`;
  `MIMIR_DEBUG_DIR`, `MIMIR_PROFILE` (`summary|tree|trace`), `MIMIR_MAX_FP_ITERS` and
  `MIMIR_DECOMPOSITION_POLICY` (`mimir|none`) mirror the `torch.compile(..., options={...})` keys.
- KernelBench expectations at size divisor 16: Level 1 100/100, Level 2 99/100 (case 28 is a known
  lowering defect, see `docs/kernelbench_level2_report.md`). CI (`.github/workflows/kernelbench.yml`)
  enforces pass rates 1.00 / 0.95 / 1.00 for levels 1–3 on Linux and macOS.

## Architecture

The package is a `torch.compile` backend named `"mimir"` (`import mimir_frontend.backend` registers it).
One compiled graph flows through:

1. `backend.mimir_backend` validates inputs (only `float32`/`int64` tensors and scalars; `get_attr`
   nodes are rejected), then applies the `DecompositionPolicy` (`decomposition.py`) through `make_fx`.
   The policy is deliberately tiny: a few ATen ops are expanded by PyTorch's own tables (`fallback`),
   `native_layer_norm` is preserved as a single op. Operator semantics are never reimplemented here;
   they live either in PyTorch's decomposition table or in the MimIR torch plugin.
2. `translator.FXGraphTranslator` walks the FX graph with a convert map (`create_convert_map`, FX target →
   `_wrap_*` adapter) that normalises overloads, scalar vs tensor arguments, negative dims, `alpha`,
   `keepdim`, multi-output nodes, and then calls into `operators.OperatorLibrary`. `utils.ShapeEnv` and
   `shape_rules.ShapeRules` keep PyTorch's *logical* shape next to the MimIR type, because MimIR
   normalisation folds singleton axes while `transpose`/`keepdim`/broadcast still count logical ranks.
   Symbolic dims (`Sym("s")` from Inductor) become MimIR dependent-type parameters.
3. `OperatorLibrary` builds MimIR terms through the Python bindings using annexes from the `tensor`,
   `math`, `affine`, `option`, `core`, `runtime` plugins and the torch dialect (`MimIR/extra/torch/torch.mim`:
   public `torch.*` axioms plus `*_impl` lambdas; `MimIR/extra/runtime` supplies `runtime.require`
   checks and the LLVM emission used by the JIT).
4. `world.optimize()` runs MimIR's pipeline (torch `lower_torch` decomposes axioms into their `_impl`,
   tensor lowering, bufferisation), `mim.JIT` emits LLVM IR and compiles it with clang, the shared library
   is loaded via `ctypes`, outputs are unpacked from metadata, and the artefacts are stored in the JIT
   cache (`_cache_key` = FX graph + input shapes + MimIR fingerprint). `debug_dir` keeps pre/post
   optimisation dumps, LLVM IR, the `.so` and a `*_manifest.json` per graph.

Around that core: `audit.py` wraps the backend to record one deterministic record per Dynamo partition
(used by the pretrained-model E2E tests); `model_export.py` / `utils.model_to_mimir` turn
the declarative `export_to_mim` specs in `models/py/*.py` into `.mim` files; `inductor_readable.py`
imports Inductor's `fx_graph_readable.py` dumps and reports partial translations; `scripts/` holds the
resource-bounded runners (each case in its own subprocess with memory/timeout limits).

Adding an operator therefore touches three places in order: an axiom + `_impl` in the torch plugin
(`torch.mim`, rebuild through bootstrap), a convert-map entry / `OperatorLibrary` method in the frontend,
and tests covering static and dynamic shapes at more than one rank. If PyTorch should expand the op
instead, add it to `DecompositionPolicy.fallback` together with a regression test proving the
expansion only reaches ops MimIR accepts.

## Reference material in the repo

- `docs/kernelbench_level*_report.md`: per-level failure analysis; `docs/frontend_backend_engineering.md`:
  JIT cache, manifest and benchmark design.
- `shape_invariants.md` and `dynamic_shape.md`: the shape/symbolic-dimension design rules the translator
  follows; `operators_summary.md` and `operator_gemini_impl_summary.md`: operator coverage tables.
- `MimIR/` is the pinned MimIR submodule (fork branch `frontend-dev`, upstream master plus `tensor.scan` and fixes); `KernelBench/` is the
  benchmark corpus submodule, excluded from pytest collection.
