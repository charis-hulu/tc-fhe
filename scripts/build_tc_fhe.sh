#!/bin/bash
# ---------------------------------------------------------------------------
# Builds tc-fhe (bench_tfhe) against an already-built tfhe-rs C API. Fast --
# use this for every rebuild after editing our own code. Run
# scripts/build_tfhe_rs.sh first (and only again if tfhe-rs itself changes).
#
# Override the tfhe-rs location or cmake build dir via env vars, e.g.:
#   TFHE_RS_DIR=~/tfhe-rs-gpu BUILD_DIR=build_gpu scripts/build_tc_fhe.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TC_FHE_DIR=${TC_FHE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
TFHE_RS_DIR=${TFHE_RS_DIR:-"$HOME/tfhe-rs"}
BUILD_DIR=${BUILD_DIR:-build}

cd "${TC_FHE_DIR}"
cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DTC_WITH_TFHE=ON \
      -DTFHE_ROOT="$(realpath "${TFHE_RS_DIR}/target/release")" \
      -DTC_BUILD_TESTS=OFF -DTC_BUILD_BENCHMARKS=ON
cmake --build "${BUILD_DIR}" --target bench_tfhe -j "$(nproc)"

echo "=== Built ${TC_FHE_DIR}/${BUILD_DIR}/benchmarks/bench_tfhe ==="
