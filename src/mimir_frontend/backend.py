"""torch.compile backend that executes FX graphs through MimIR.

Usage::

    import mimir_frontend.backend  # registers the "mimir" backend

    compiled = torch.compile(model, backend="mimir")
    # or: torch.compile(model, backend=mimir_frontend.backend.mimir_backend)

The backend translates the Dynamo FX graph with `build_model_function`, lowers it
through the `opt` plugin's default pipeline (which bufferizes tensor ops), lets
`%ll.emit` write LLVM IR, and JIT-compiles it to a shared library via `mim.JIT`.

Calling convention of the emitted function (verified against eager PyTorch):
- every input tensor is passed as a flat row-major `float*`,
- the result tensor is returned as a callee-malloc'd `float*`.

Dynamo lifts module parameters into graph placeholders, so all tensors — inputs
and parameters alike — arrive as runtime arguments and are marshaled uniformly.

Multiple outputs are supported by packing: the graph is rewritten to flatten and
concatenate all outputs into one flat tensor (keeping the single-`float*` return
ABI), which the runtime wrapper slices and reshapes back.

Debugging: pass ``options={"debug_dir": "some/dir"}`` to ``torch.compile`` (or
set the ``MIMIR_DEBUG_DIR`` environment variable) to keep all per-graph
compilation artifacts in that directory, including a dump of the MimIR world
before (`<name>_pre.mim`) and after (`<name>_post.mim`) `world.optimize()`,
alongside the emitted `<name>.ll` and `<name>.so`.

Profiling: pass ``options={"profile": "summary" | "tree" | "trace"}`` (or set
``MIMIR_PROFILE``) to record MimIR phase runtimes during `world.optimize()`.
`summary` and `tree` print to stderr (or to `<name>_profile.txt` under a
`debug_dir`); `trace` writes chrome://tracing-compatible JSON to
`<name>_profile.json` for chrome://tracing, Perfetto, or speedscope.
Profiling (like `debug_dir`) forces a fresh compile instead of a cache hit.

Caching: compiled `.so` files are reused across processes from
``~/.cache/mimir-frontend/jit`` (override with ``options={"cache_dir": ...}``
or ``MIMIR_CACHE_DIR``; disable with ``options={"cache": False}``). The cache
key hashes the canonicalized FX graph and input shapes together with a
fingerprint of the MimIR installation (bindings, libmim, the loaded plugins,
and clang), so rebuilding MimIR invalidates stale entries. Parameter *values*
are runtime arguments and deliberately not part of the key. A `debug_dir`
request always recompiles (to produce the dumps) but still refreshes the cache.

Current limitations: float32 tensors only, static shapes, inference only (the
compiled function does not participate in autograd).

Platform support: developed and tested on Linux. The platform-specific pieces
(loaded-library discovery for the cache fingerprint, the CRT `free` paired
with the JIT'd `malloc`, shared-library suffixes, and cache locations) have
macOS and Windows code paths, but those are untested; `mim.JIT` itself
supports all three platforms.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import platform
import shutil
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path

import mim
import torch
from torch import fx

from .utils import build_model_function

# `opt` supplies the default compile pipeline (incl. %tensor.lower_to_mem
# bufferization) and `ll` its final %ll.emit phase — mirrors the lit-test
# invocation `mim -p opt -p clos <file>.mim -p ll`.
EXEC_PLUGINS = ["opt", "clos", "math", "tensor", "ll"]

def _load_crt() -> ctypes.CDLL:
    """The C runtime providing the `free` that matches the JIT'd code's `malloc`."""
    if platform.system() == "Windows":
        # clang on Windows links the universal CRT; its allocator must be
        # paired with its own free. CDLL(None) is not supported there.
        for crt in ("ucrtbase", "msvcrt"):
            try:
                return ctypes.CDLL(crt)
            except OSError:
                continue
        raise OSError("no C runtime found (tried ucrtbase, msvcrt)")
    return ctypes.CDLL(None)


_LIBC = _load_crt()
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None

# %ll.emit and mim.JIT write `<name>.ll`/`<name>.so` to the current directory,
# so compilation chdirs into a scratch dir; the lock keeps that process-global
# state change safe.
_compile_lock = threading.Lock()


def _stat_sig(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path.name}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{path.name}:?"


def _loaded_module_paths() -> Iterator[str]:
    """Paths of the shared libraries loaded into this process, per platform."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/self/maps") as maps:
                for line in maps:
                    if "/" in line:
                        yield line.rsplit(maxsplit=1)[-1]
        except OSError:
            pass
    elif system == "Darwin":
        libc = ctypes.CDLL(None)
        libc._dyld_image_count.restype = ctypes.c_uint32
        libc._dyld_get_image_name.restype = ctypes.c_char_p
        libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        for i in range(libc._dyld_image_count()):
            if name := libc._dyld_get_image_name(i):
                yield os.fsdecode(name)
    elif system == "Windows":
        import ctypes.wintypes as wt

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        handles = (wt.HMODULE * 1024)()
        needed = wt.DWORD()
        if psapi.EnumProcessModules(
            kernel32.GetCurrentProcess(), handles, ctypes.sizeof(handles), ctypes.byref(needed)
        ):
            buffer = ctypes.create_unicode_buffer(32768)
            for i in range(min(needed.value // ctypes.sizeof(wt.HMODULE), len(handles))):
                if kernel32.GetModuleFileNameW(handles[i], buffer, len(buffer)):
                    yield buffer.value


def _libmim_path() -> Path | None:
    """Locate the loaded libmim shared library (importing mim loads it)."""
    for entry in _loaded_module_paths():
        name = Path(entry).name.lower()
        # libmim.so[.x.y] / libmim.dylib on POSIX, [lib]mim.dll on Windows.
        if name.startswith("libmim.") or name in ("mim.dll", "libmim.dll"):
            return Path(entry)
    return None


def _shared_lib_suffix() -> str:
    system = platform.system()
    if system == "Windows":
        return ".dll"
    return ".dylib" if system == "Darwin" else ".so"


@lru_cache(maxsize=1)
def _mim_fingerprint() -> str:
    """Fingerprint of everything the compiled artifact depends on besides the graph.

    Covers the Python bindings, libmim, every file in the plugin directory
    (annex `.mim` files and `libmim_*` modules), and the clang used by mim.JIT.
    Stat-based (size + mtime), ccache-style.
    """
    parts = [getattr(mim, "__version__", "unversioned")]
    parts.extend(_stat_sig(p) for p in sorted(Path(mim.__file__).parent.glob("_mim*")))
    if libmim := _libmim_path():
        parts.append(_stat_sig(libmim))
        plugin_dir = libmim.parent / "mim"
        if plugin_dir.is_dir():
            parts.extend(_stat_sig(p) for p in sorted(plugin_dir.iterdir()))
    if clang := shutil.which("clang"):
        parts.append(_stat_sig(Path(clang)))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _target_repr(target) -> str:
    """Canonical, process-independent representation of an FX node target.

    `repr()` of builtins like `torch.relu` embeds a memory address, so build
    the name from module/qualname instead.
    """
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if qualname:
        return f"{module}.{qualname}" if module else qualname
    return str(target)


def _cache_key(gm: fx.GraphModule, input_shapes) -> str:
    hasher = hashlib.sha256(_mim_fingerprint().encode())
    for node in gm.graph.nodes:
        args = fx.node.map_arg((node.args, node.kwargs), lambda n: f"%{n.name}")
        hasher.update(f"{node.op}|{_target_repr(node.target)}|{args!r}\n".encode())
    hasher.update(repr(input_shapes).encode())
    return hasher.hexdigest()[:16]


def _default_cache_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "mimir-frontend" / "jit"


def _cache_store(cache_dir: Path, sources: list[Path]) -> None:
    """Atomically publish the compiled artifacts into the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for src in sources:
        if not src.exists():
            continue
        tmp = cache_dir / f".{src.name}.{uuid.uuid4().hex}"
        shutil.copy(src, tmp)
        try:
            os.replace(tmp, cache_dir / src.name)
        except OSError:
            # Windows refuses to replace a library another process holds open;
            # drop this update, the existing entry stays valid.
            tmp.unlink(missing_ok=True)


def _check_tensors(tensors, what: str) -> None:
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            raise NotImplementedError(f"mimir backend supports tensor {what} only, {what} {i} is {type(t).__name__}")
        if t.dtype != torch.float32:
            raise NotImplementedError(f"mimir backend supports float32 only, {what} {i} has dtype {t.dtype}")
        if t.dim() == 0:
            raise NotImplementedError(f"mimir backend does not support 0-d tensor {what}s")


def _pack_outputs(gm: fx.GraphModule, out_shapes: list[tuple[int, ...]]) -> None:
    """Rewrite the graph so it returns a single flat tensor.

    Dynamo graphs return a container of tensors. To keep the verified
    single-`float*` return ABI, flatten every output and concatenate them:
    `output((a, b))` becomes `output(cat([reshape(a, (n_a,)), reshape(b, (n_b,))]))`.
    A single output is simply unwrapped from its container.
    """
    output = next(n for n in gm.graph.nodes if n.op == "output")
    nodes = output.args[0]
    if len(nodes) == 1:
        output.args = (nodes[0],)
    else:
        with gm.graph.inserting_before(output):
            flats = [
                gm.graph.call_function(torch.reshape, (node, (math.prod(shape),)))
                for node, shape in zip(nodes, out_shapes)
            ]
            packed = gm.graph.call_function(torch.cat, (flats, 0))
        output.args = (packed,)
    gm.graph.lint()
    gm.recompile()


def mimir_backend(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: dict | None = None,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Compile a Dynamo FX graph to native code through MimIR."""
    opts = dict(options or {})
    debug_dir = opts.pop("debug_dir", None) or os.environ.get("MIMIR_DEBUG_DIR")
    profile = opts.pop("profile", None) or os.environ.get("MIMIR_PROFILE")
    cache_enabled = opts.pop("cache", True)
    cache_dir = Path(opts.pop("cache_dir", None) or os.environ.get("MIMIR_CACHE_DIR") or _default_cache_dir())
    if opts:
        raise TypeError(f"unknown mimir backend options: {sorted(opts)}")
    if profile is not None and profile not in ("summary", "tree", "trace"):
        raise ValueError(f"profile must be 'summary', 'tree', or 'trace', got {profile!r}")

    _check_tensors(example_inputs, "input")
    if any(n.op == "get_attr" for n in gm.graph.nodes):
        # Dynamo lifts parameters, buffers, and tensor constants into
        # placeholders; get_attr would need extra trailing-argument marshaling.
        raise NotImplementedError("mimir backend does not support get_attr nodes")

    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    container = output_node.args[0]
    if not isinstance(container, (tuple, list)) or not all(isinstance(n, fx.Node) for n in container):
        raise NotImplementedError(f"unsupported output structure: {container!r}")
    restore = type(container)

    # Shapes are static: one specialization per Dynamo guard set.
    input_shapes = [tuple(t.shape) for t in example_inputs]
    with torch.no_grad():
        example_outs = gm(*example_inputs)
    _check_tensors(example_outs, "output")
    out_shapes = [tuple(t.shape) for t in example_outs]
    out_numels = [math.prod(shape) for shape in out_shapes]
    out_offsets = [sum(out_numels[:i]) for i in range(len(out_numels))]
    total_numel = sum(out_numels)

    _pack_outputs(gm, out_shapes)

    # The key covers the packed graph, input shapes, and the MimIR install;
    # it also names the exported symbol, so cached libraries are self-contained.
    name = f"mimir_graph_{_cache_key(gm, input_shapes)}"
    cached_so = cache_dir / f"{name}{_shared_lib_suffix()}"

    lib = keep_alive = None
    # debug_dir and profile both require an actual compile to observe.
    if cache_enabled and not debug_dir and not profile and cached_so.exists():
        lib = ctypes.cdll.LoadLibrary(str(cached_so))

    if lib is None:
        if debug_dir:
            build_dir = Path(debug_dir)
            build_dir.mkdir(parents=True, exist_ok=True)
            print(f"[mimir] {name}: keeping compilation artifacts in {build_dir}", file=sys.stderr)
        else:
            build_dir = Path(tempfile.mkdtemp(prefix=f"{name}-"))

        driver = mim.Driver()
        driver.load_plugins(EXEC_PLUGINS)
        world = driver.world()
        if profile:
            # Phases only record spans while this flag is set (see mim Phase::run).
            driver.flags().profile = {
                "summary": mim.Flags.Profile.Summary,
                "tree": mim.Flags.Profile.Tree,
                "trace": mim.Flags.Profile.Trace,
            }[profile]

        build_model_function(world, gm, input_shapes, name=name)

        with _compile_lock:
            old_cwd = os.getcwd()
            os.chdir(build_dir)
            try:
                if debug_dir:
                    # world.write() dumps the world to `<world name>.mim` in cwd.
                    world.set(f"{name}_pre")
                    world.write()
                # The opt plugin's default pipeline ends in %ll.emit -> `<name>.ll`.
                world.set(name)
                world.optimize()
                if debug_dir:
                    world.set(f"{name}_post")
                    world.write()
                jit = mim.JIT(name, [name])
                lib = jit.compile_and_load()
                # Resolve while still chdir'd: on POSIX the JIT builds
                # `./<name>.so` in cwd, on Windows a DLL in a temp dir.
                built_lib = Path(jit._get_so_path()).resolve()
            finally:
                os.chdir(old_cwd)

        if profile:
            report = {
                "summary": driver.profiler().summary,
                "tree": driver.profiler().tree,
                "trace": driver.profiler().chrome_trace,
            }[profile]()
            if profile == "trace" or debug_dir:
                # chrome://tracing JSON (and anything under debug_dir) goes to a file.
                dest = build_dir / f"{name}_profile.{'json' if profile == 'trace' else 'txt'}"
                dest.write_text(report)
                print(f"[mimir] {name}: phase profile written to {dest}", file=sys.stderr)
            else:
                print(f"[mimir] {name}: phase profile\n{report}", file=sys.stderr)

        if cache_enabled:
            _cache_store(cache_dir, [built_lib, build_dir / f"{name}.ll"])
        # Keep the driver/world alive as long as the callable.
        keep_alive = driver

    fn = lib[name]
    fn.argtypes = [ctypes.c_void_p] * len(input_shapes)
    fn.restype = ctypes.POINTER(ctypes.c_float)

    def compiled(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Keep the contiguous buffers referenced until the call returns.
        buffers = [a.detach().contiguous() for a in args]
        out_ptr = fn(*[b.data_ptr() for b in buffers])
        array = ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float * total_numel)).contents
        flat = torch.frombuffer(array, dtype=torch.float32).clone()
        _LIBC.free(ctypes.cast(out_ptr, ctypes.c_void_p))
        # The outputs are disjoint views of one flat buffer.
        return restore(
            flat[offset : offset + numel].reshape(shape)
            for offset, numel, shape in zip(out_offsets, out_numels, out_shapes)
        )

    # Keep the loaded library (and the driver on a cache miss) alive.
    compiled._mimir = (keep_alive, lib)
    return compiled


try:
    from torch._dynamo.backends.registry import register_backend

    register_backend(mimir_backend, name="mimir")
except Exception:  # pragma: no cover - registration is best-effort
    pass
