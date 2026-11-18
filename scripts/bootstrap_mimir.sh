#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMIR_DIR="${ROOT_DIR}/MimIR"
BUILD_DIR="${MIMIR_DIR}/build"

git -C "${ROOT_DIR}" submodule update --init --recursive

# MimIR discovers plugins under extra/ at configure time; the frontend's torch dialect and runtime checks live there.
fetch_plugin() {
  local dir="${MIMIR_DIR}/extra/$1"
  [[ -d "${dir}/.git" ]] || git clone "$2" "${dir}"
  # Fetch from the URL, not `origin`, so a checkout cloned from another fork still switches.
  git -C "${dir}" fetch -q "$2" "$3"
  git -C "${dir}" checkout -q --detach FETCH_HEAD
}
fetch_plugin torch "${MIMIR_TORCH_PLUGIN_URL:-https://github.com/HarithF/mimir-torch-plugin.git}" "${MIMIR_TORCH_PLUGIN_REF:-art-eval}"
fetch_plugin runtime https://github.com/hsqStephenZhang/mimir-runtime-plugin.git "${MIMIR_RUNTIME_PLUGIN_REF:-feat/ll-emit}"

# Optional MLIR emitter (`emit_only="mlir"` backend option); MIM_MLIR_SKIP=1 builds without it.
MLIR_PLUGIN_DIR="${MIMIR_DIR}/extra/mim_mlir"
if [[ -n "${MIM_MLIR_SKIP:-}" ]]; then
  echo "MIM_MLIR_SKIP set; building without the mlir plugin"
elif [[ ! -d "${MLIR_PLUGIN_DIR}/.git" ]]; then
  fetch_plugin mim_mlir "${MIM_MLIR_URL:-https://github.com/HarithF/mim_mlir.git}" "${MIM_MLIR_REF:-mimir-frontend-dev}"
fi

if [[ ! -d "${ROOT_DIR}/.venv" ]]; then
  uv venv --python 3.14
fi
# MimIR's binding generator needs clang.cindex before the project (and its mim path dependency) can be synced.
uv pip install -q libclang

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/Scripts/python.exe"
else
  echo "Could not find Python in ${ROOT_DIR}/.venv" >&2
  exit 1
fi

cmake -S "${MIMIR_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${MIMIR_BUILD_TYPE:-Release}" \
  -DMIM_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="${PYTHON_BIN}"
cmake --build "${BUILD_DIR}" --target mim_py -j "${MIMIR_BUILD_JOBS:-8}"

# MimIR stages the package under build/python without packaging metadata; wrap it so uv can install it.
STAGE_DIR="${BUILD_DIR}/mim_py_stage/main"
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/src"
cp -R "${BUILD_DIR}/python/mim" "${STAGE_DIR}/src/mim"
cat >"${STAGE_DIR}/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mim"
version = "0.5.0.dev0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
mim = ["*.so", "*.pyd", "*.dll", "*.pyi", "py.typed", "plug/**/*"]
EOF

cd "${ROOT_DIR}"
uv sync --reinstall-package mim
