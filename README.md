# Mimir Frontend

A PyTorch FX graph importer for MimIR.

## Setup

This project vendors its tested MimIR revision as a submodule. Clone with submodules:

```bash
git clone --recursive git@github.com:hsqStephenZhang/mimir-frontend.git
cd mimir-frontend
```

Then build the local MimIR Python binding and sync the Python environment:

```bash
./scripts/bootstrap_mimir.sh
```

The script initializes recursive submodules, creates a Python 3.14 `uv` venv,
builds MimIR's `mim_py` target, and runs `uv sync`.

MimIR needs a C++23 toolchain (`<print>` must compile). On macOS use Homebrew
LLVM rather than Apple clang, whose Command Line Tools can ship incomplete
libc++ headers; CI does the same:

```bash
brew install llvm ninja
export CC="$(brew --prefix llvm)/bin/clang" CXX="$(brew --prefix llvm)/bin/clang++"
./scripts/bootstrap_mimir.sh
```

Set `CC`/`CXX` before the first configure; `MimIR/build` caches the compiler.

The `mim` dependency is resolved from the local submodule build output:

```toml
mim = { path = "MimIR/build/mim_py_stage/main" }
```

After editing MimIR locally, rebuild and resync with the same command.
The build type defaults to `Release`; set `MIMIR_BUILD_TYPE=Debug` for a debug build
(note that `world.optimize()` is drastically slower in debug builds).

## Running Examples

Each model in `models/py/` is runnable and JIT-compiles itself through the
`"mimir"` `torch.compile` backend, checking the result against eager PyTorch:

```bash
uv run --no-sync python models/py/mlp.py
```

The same files also serve as declarative export specs: `export_to_mim` is
consumed by `scripts/export_models_to_mimir.py` to write `.mim` files into
`models/mim/`.

To use the backend on your own model:

```python
import torch
import mimir_frontend.backend  # registers the "mimir" backend

compiled = torch.compile(model, backend="mimir")
# pass options={"debug_dir": "dbg/"} to keep the pre/post-optimize
# MimIR dumps, emitted LLVM IR, shared library, and compilation manifest
# per compiled graph
# pass options={"profile": "summary" | "tree" | "trace"} (or set MIMIR_PROFILE)
# to report MimIR phase runtimes; "trace" writes chrome://tracing JSON
# pass options={"max_fp_iters": N} (or set MIMIR_MAX_FP_ITERS) for large graphs
# pass options={"decomposition_policy": "mimir" | "none"} to select the
# frontend decomposition boundary
```

Compiled graphs are cached in `~/.cache/mimir-frontend/jit` keyed by the FX
graph, input shapes, and a fingerprint of the MimIR installation (rebuilding
MimIR invalidates the cache). Override the location with `MIMIR_CACHE_DIR` or
`options={"cache_dir": ...}`; disable with `options={"cache": False}`. Cache
updates are atomic, invalid shared libraries are discarded and rebuilt, and
non-debug temporary compilation directories are removed after each compile on
POSIX systems. Windows retains successful build directories while their DLL is
loaded.

A debug directory also contains one `mimir_graph_*_manifest.json` per compiled
graph. It records input/output shape and dtype metadata, the decomposition and
fixed-point options, loaded plugins, cache/fingerprint identifiers, generated
artifacts, and success or failure diagnostics. It deliberately excludes tensor
values and model weights.

## Running Tests

Use `uv run --no-sync pytest -q` after bootstrap:

```bash
uv run --no-sync pytest -q
```

## Benchmarking Backends

Compare eager PyTorch, Inductor, and MimIR with each backend isolated in a
resource-bounded subprocess:

```bash
uv run --no-sync python scripts/benchmark_backends.py models/py/mlp.py \
  --repeat 5 --threads 1 --max-memory-gb 16 --timeout 300
```

The runner checks every result against eager execution and reports first-call
time, steady-state mean/minimum time, eager-relative speed, and maximum absolute
error. Use `--no-cache` to include a fresh MimIR compile and `--results-json`
to retain machine-readable results. Memory and timeout limits apply per model
and backend, so a failed large compile does not terminate the parent run.

## Running KernelBench

`scripts/run_kernelbench_mimir.py` compiles every KernelBench model through the
MimIR backend and checks it against eager PyTorch. The model sources come from
the `KernelBench` submodule (initialized by the bootstrap). The per-kernel
fixtures (input shapes, dtypes, initializations) are vendored copies of
Lighthouse's `examples/KernelBench/*.yaml` in
`scripts/kernelbench_fixtures/lighthouse/`, with local overlays in
`scripts/kernelbench_fixtures/*.yaml`. No extra checkout is needed:

```bash
uv run --no-sync python scripts/run_kernelbench_mimir.py --suite level1 \
  --size-divisor 16 --timeout 300 --max-memory-gb 12 --max-fp-iters 512 \
  --allow-invalid --results-json kernelbench-level1.json
```

- `--suite` selects `ci`, `level1`, `level2`, `level3`, `full` (levels 1-3),
  or `level4` (full pretrained models). `--kernel TEXT` filters by path
  substring; `--list` prints the selection without running it.
- Every case runs in its own subprocess under `--timeout` seconds and a
  `--max-memory-gb` address-space limit. Set `MIMIR_CACHE_DIR` to keep a run's
  JIT cache separate from `~/.cache/mimir-frontend/jit`.
- `--lighthouse PATH` reads the fixtures and corpus from a
  [libxsmm/lighthouse](https://github.com/libxsmm/lighthouse) checkout instead
  (CI pins `d330e762`); `--resume` skips cases already passing in the results
  JSON.

Expected coverage at `--size-divisor 16` is 100/100 on Level 1 and 99/100 on
Level 2; the remaining Level 2 failure is documented in
`docs/kernelbench_level2_report.md`. Level 3 and 4 details are in the
other `docs/kernelbench_level*_report.md` files.

## Fresh Clone Smoke Test

The expected mentor workflow is:

```bash
git clone --recursive git@github.com:hsqStephenZhang/mimir-frontend.git
cd mimir-frontend
./scripts/bootstrap_mimir.sh
uv run --no-sync python -c 'import mim; print("mim import ok")'
uv run --no-sync pytest -q
```

This was verified from a clean clone with:

```text
166 passed, 6 skipped
```

If pytest reports a cache warning about `.pytest_cache` under a sandboxed runner,
it is an environment permission warning and does not affect the test result.

## Architecture

- `src/mimir_frontend/translator.py`: The core `FXGraphTranslator` class.
- `src/mimir_frontend/operators.py`: MimIR operator definitions using the Python bindings.
- `tests/test_basic.py`: Basic verification tests.

## Operator Support

Currently supports:
- `torch.add`, `torch.mul`, `torch.sub`, `torch.div`
- `torch.relu`
- Basic shape propagation

TODO:
- Comprehensive `reduce_sum` implementation.
- `pooling` (pad + reduce + elementwise).
- Dynamic shape handling refinements.
