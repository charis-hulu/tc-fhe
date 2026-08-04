#!/bin/bash
# ---------------------------------------------------------------------------
# Builds the bgv-batched backend (bgv_batched_closure, bench_bgv_batched,
# generate_bgv_batched_profiles) against an already-built OpenFHE install.
# Run this on the login node -- compute nodes don't have cmake on PATH.
#
# Environment quirks this script works around, specific to this OpenFHE
# 1.1.2 install / Polaris login-node toolchain (see docs/bgv_batched.md and
# docs/bgv_parameter_profiles.md for the backend itself):
#   - No `cmake` on PATH by default here; use the one under /soft/spack.
#   - The system default g++ (7.5.0) can't compile OpenFHE's headers
#     (SFINAE on __int128 needs a newer compiler) -- use gcc-native/13.
#   - CMakeLists.txt sets CMAKE_CXX_EXTENSIONS OFF (-std=c++17, not
#     gnu++17), but OpenFHE's NativeIntegerT constructor SFINAE only
#     resolves for __int128 under GNU extensions -- force -std=gnu++17.
#   - libOPENFHEpke.so was built with OpenMP (a `#pragma omp threadprivate`
#     PRNG member), so the final link needs -fopenmp too, or you get a
#     "TLS reference mismatches non-TLS reference" link error.
#   - nlohmann_json has no installed CMake config package here; a minimal
#     header-only shim config lives in ~/local/cmake/nlohmann_json
#     (pointing at ~/include/nlohmann/json.hpp) -- see nlohmann_json_DIR
#     below. Adjust/rebuild that shim if your account doesn't have it.
#
# Override the OpenFHE location or cmake build dir via env vars, e.g.:
#   OPENFHE_DIR=~/openfhe-other BUILD_DIR=build_bgv_batched_alt \
#     scripts/build_tc_fhe_bgv_batched.sh
# ---------------------------------------------------------------------------
set -euo pipefail

TC_FHE_DIR=${TC_FHE_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
OPENFHE_DIR=${OPENFHE_DIR:-"$HOME/openfhe"}
BUILD_DIR=${BUILD_DIR:-build_bgv_batched}
NLOHMANN_JSON_DIR=${NLOHMANN_JSON_DIR:-"$HOME/local/cmake/nlohmann_json"}
CMAKE_BIN=${CMAKE_BIN:-/soft/spack/pe/0.10.1/base/install/linux-sles15-x86_64_v3/gcc-13.3.1/cmake-3.30.5-bpqb3xljvz3qwekwpgjdcp6vk3qfhocj/bin/cmake}

source /opt/cray/pe/lmod/lmod/init/bash 2>/dev/null || true
module load gcc-native/13 >/dev/null 2>&1 || true
GXX_BIN=/opt/cray/pe/gcc-native/13/bin/g++

# Wrapper that force-appends -std=gnu++17 -fopenmp after CMake's own
# -std=c++17 flag (last -std wins with GCC), and after any other flags CMake
# passes, so the fixes above always take effect regardless of CMakeLists.txt
# settings.
WRAP_DIR="${BUILD_DIR}_ccwrap"
mkdir -p "${WRAP_DIR}"
cat > "${WRAP_DIR}/g++" <<EOF
#!/bin/bash
exec ${GXX_BIN} "\$@" -std=gnu++17 -fopenmp
EOF
chmod +x "${WRAP_DIR}/g++"

cd "${TC_FHE_DIR}"
export PATH="$(dirname "${CMAKE_BIN}"):${PATH}"

cmake -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DTC_WITH_BGV_BATCHED=ON \
      -DCMAKE_PREFIX_PATH="${OPENFHE_DIR}" \
      -DCMAKE_CXX_COMPILER="$(realpath "${WRAP_DIR}/g++")" \
      -Dnlohmann_json_DIR="${NLOHMANN_JSON_DIR}" \
      -DTC_BUILD_TESTS=OFF -DTC_BUILD_BENCHMARKS=OFF
cmake --build "${BUILD_DIR}" --target bgv_batched_closure generate_bgv_batched_profiles -j "$(nproc)"

echo "=== Built ${TC_FHE_DIR}/${BUILD_DIR}/examples/bgv_batched_closure ==="
echo "=== Built ${TC_FHE_DIR}/${BUILD_DIR}/tools/generate_bgv_batched_profiles ==="
