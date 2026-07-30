#!/bin/bash
# ---------------------------------------------------------------------------
# Builds tfhe-rs' C API with only the features tc-fhe needs. Run this once
# (or whenever tfhe-rs itself changes) -- for rebuilding tc-fhe after editing
# our own code, use scripts/build_tc_fhe.sh instead, which is much faster.
#
# Default (CPU): tc-fhe's CPU backend only calls boolean_* functions (see
# include/btc/tfhe_backend.hpp), so we skip shortint-c-api, high-level-c-api,
# zk-pok, and extended-types -- these are what `make build_c_api` in tfhe-rs
# pulls in by default and they add a lot of compile time for code paths we
# never touch. The `shortint` feature (not shortint-c-api) is still required:
# the boolean C API is implemented on top of the shortint core module
# internally.
#
# GPU: the GPU backend (include/btc/tfhe_gpu_backend.hpp) uses FheUint2 from
# the high-level API plus CUDA-specific functions, a different feature set --
# build it in a separate clone (keeps the CPU-only build untouched):
#   git clone https://github.com/zama-ai/tfhe-rs.git ~/tfhe-rs-gpu
#   module load cudatoolkit-standalone   # needed for nvcc; load AFTER set_env.sh
#   TFHE_RS_DIR=~/tfhe-rs-gpu FEATURES=high-level-c-api,gpu scripts/build_tfhe_rs.sh
#
# GPU-specific gotchas already worked around below (CUDA_LIB_DIR -> LIBRARY_PATH):
# tfhe-cuda-backend's build.rs looks for CUDA via pkg-config, which Polaris
# doesn't have, then falls back to a hardcoded /usr/local/cuda/lib64 that
# doesn't exist here either -- linking then fails with "cannot find -lcudart".
# Two gotchas that are NOT automated (rare enough not to script around):
#   - `cc` must resolve to gcc, not NVIDIA's nvc -- `source set_env.sh` first
#     (loading gcc-native swaps PrgEnv-nvidia -> PrgEnv-gnu as a side effect).
#   - if a previous attempt failed *before* CUDA_ROOT/cudatoolkit was set up,
#     CMake caches CMAKE_CUDA_COMPILER=NOTFOUND and won't re-probe. Clear it:
#     rm -rf "$TFHE_RS_DIR"/target/release/build/tfhe-cuda-common-*
#
# Run this inside tmux/screen -- the tfhe crate takes a long time to compile
# and an SSH disconnect will SIGHUP-kill a foreground build:
#   tmux new -s tfhe-build
#   scripts/build_tfhe_rs.sh
#   (Ctrl+b d to detach, `tmux attach -t tfhe-build` to check back in)
#
# Override the toolchain, source dir, or CUDA lib dir via env vars, e.g.:
#   NIGHTLY=nightly-2026-08-01 TFHE_RS_DIR=~/tfhe-rs-gpu scripts/build_tfhe_rs.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TFHE_RS_DIR=${TFHE_RS_DIR:-"$HOME/tfhe-rs"}
NIGHTLY=${NIGHTLY:-nightly-2026-07-15}
FEATURES=${FEATURES:-boolean-c-api,shortint}
CUDA_LIB_DIR=${CUDA_LIB_DIR:-/soft/compilers/cudatoolkit/cuda-12.2.2/targets/x86_64-linux/lib}

cd "${TFHE_RS_DIR}"

RUSTFLAGS="-C target-cpu=native" LIBRARY_PATH="${CUDA_LIB_DIR}:${LIBRARY_PATH:-}" \
    cargo "+${NIGHTLY}" build --profile release \
    --features="${FEATURES}" \
    -p tfhe

echo "=== Built ${TFHE_RS_DIR}/target/release/libtfhe.a (features: ${FEATURES}) ==="
