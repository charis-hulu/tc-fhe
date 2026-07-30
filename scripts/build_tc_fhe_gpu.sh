#!/bin/bash
# ---------------------------------------------------------------------------
# Builds tc-fhe's GPU target (bench_tfhe_gpu) against an already-built
# tfhe-rs GPU C API. Fast -- use this for every rebuild after editing tc-fhe's
# own code. Run scripts/build_tfhe_rs.sh with FEATURES=high-level-c-api,gpu
# first (and only again if tfhe-rs itself changes).
#
# Override the tfhe-rs location, cmake build dir, or CUDA path via env vars, e.g.:
#   TFHE_RS_DIR=~/tfhe-rs-gpu BUILD_DIR=build_gpu \
#       CUDA_ROOT=/soft/compilers/cudatoolkit/cuda-12.2.2 scripts/build_tc_fhe_gpu.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TC_FHE_DIR=${TC_FHE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
TFHE_RS_DIR=${TFHE_RS_DIR:-"$HOME/tfhe-rs-gpu"}
BUILD_DIR=${BUILD_DIR:-build_gpu}
CUDA_ROOT=${CUDA_ROOT:-/soft/compilers/cudatoolkit/cuda-12.2.2}

cd "${TC_FHE_DIR}"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DTC_WITH_TFHE_GPU=ON \
      -DTFHE_ROOT="$(realpath "${TFHE_RS_DIR}/target/release")" -DCUDA_ROOT="${CUDA_ROOT}" \
      -DTC_BUILD_TESTS=OFF -DTC_BUILD_BENCHMARKS=ON
cmake --build "${BUILD_DIR}" --target bench_tfhe_gpu -j "$(nproc)"

echo "=== Built ${TC_FHE_DIR}/${BUILD_DIR}/benchmarks/bench_tfhe_gpu ==="
