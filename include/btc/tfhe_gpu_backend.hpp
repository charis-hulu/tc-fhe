#pragma once

#ifdef TC_WITH_TFHE_GPU

#include "matrix.hpp"
#include <tfhe.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace btc {

namespace detail {

struct FheUint2Deleter {
    void operator()(FheUint2* p) const noexcept { if (p) fhe_uint2_destroy(p); }
};

struct CudaServerKeyDeleter {
    void operator()(CudaServerKey* p) const noexcept { if (p) cuda_server_key_destroy(p); }
};

struct CompressedServerKeyDeleter {
    void operator()(CompressedServerKey* p) const noexcept {
        if (p) compressed_server_key_destroy(p);
    }
};

} // namespace detail

using FheUint2Ptr = std::unique_ptr<FheUint2, detail::FheUint2Deleter>;
using CudaServerKeyPtr = std::unique_ptr<CudaServerKey, detail::CudaServerKeyDeleter>;
using CompressedServerKeyPtr = std::unique_ptr<CompressedServerKey, detail::CompressedServerKeyDeleter>;

// Matrix of FheUint2 ciphertexts (row-major, square).
// Each cell holds a 1-bit boolean value encrypted as a 2-bit FHE integer.
class TFHEGPUMatrix {
public:
    explicit TFHEGPUMatrix(std::size_t n) : n_(n), data_(n * n) {}

    std::size_t dim() const noexcept { return n_; }

    FheUint2* get(std::size_t i, std::size_t j) const noexcept {
        return data_[i * n_ + j].get();
    }

    void set(std::size_t i, std::size_t j, FheUint2Ptr ct) {
        data_[i * n_ + j] = std::move(ct);
    }

private:
    std::size_t n_;
    std::vector<FheUint2Ptr> data_;
};

// TFHEGPUBackend — satisfies the Backend concept using FheUint2 on GPU.
//
// The GPU server key is set globally via set_cuda_server_key; all FheUint2
// bitand/bitor calls then execute on the GPU automatically.
//
// Usage:
//   ClientKey* ckey = nullptr;
//   CompressedServerKey* csk = nullptr;
//   // generate keys (high-level API)
//   ...
//   CudaServerKey* gpu_key = nullptr;
//   compressed_server_key_decompress_to_gpu(csk, &gpu_key);
//   set_cuda_server_key(gpu_key);
//
//   TFHEGPUBackend backend;
//   auto enc_A = TFHEGPUBackend::encrypt(plain_A, ckey);
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = TFHEGPUBackend::decrypt(enc_S, ckey);
class TFHEGPUBackend {
public:
    using matrix_type = TFHEGPUMatrix;

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] )
    TFHEGPUMatrix matmul(const TFHEGPUMatrix& A, const TFHEGPUMatrix& B) const {
        const std::size_t n = A.dim();
        TFHEGPUMatrix C(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                FheUint2Ptr acc;
                for (std::size_t k = 0; k < n; ++k) {
                    FheUint2* term_raw = nullptr;
                    if (fhe_uint2_bitand(A.get(i, k), B.get(k, j), &term_raw) != 0)
                        throw std::runtime_error("TFHEGPUBackend::matmul: AND failed");
                    FheUint2Ptr term(term_raw);

                    if (!acc) {
                        acc = std::move(term);
                    } else {
                        FheUint2* or_raw = nullptr;
                        if (fhe_uint2_bitor(acc.get(), term.get(), &or_raw) != 0)
                            throw std::runtime_error("TFHEGPUBackend::matmul: OR failed");
                        acc = FheUint2Ptr(or_raw);
                    }
                }
                C.set(i, j, std::move(acc));
            }
        }
        return C;
    }

    TFHEGPUMatrix or_mat(const TFHEGPUMatrix& A, const TFHEGPUMatrix& B) const {
        const std::size_t n = A.dim();
        TFHEGPUMatrix C(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                FheUint2* r = nullptr;
                if (fhe_uint2_bitor(A.get(i, j), B.get(i, j), &r) != 0)
                    throw std::runtime_error("TFHEGPUBackend::or_mat: OR failed");
                C.set(i, j, FheUint2Ptr(r));
            }
        }
        return C;
    }

    TFHEGPUMatrix copy(const TFHEGPUMatrix& A) const {
        const std::size_t n = A.dim();
        TFHEGPUMatrix C(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                FheUint2* ct = nullptr;
                if (fhe_uint2_clone(A.get(i, j), &ct) != 0)
                    throw std::runtime_error("TFHEGPUBackend::copy: clone failed");
                C.set(i, j, FheUint2Ptr(ct));
            }
        }
        return C;
    }

    std::size_t dim(const TFHEGPUMatrix& A) const { return A.dim(); }

    // Encrypt a plaintext BoolMatrix into a TFHEGPUMatrix.
    static TFHEGPUMatrix encrypt(const BoolMatrix& plain, ClientKey* ckey) {
        const std::size_t n = plain.rows();
        TFHEGPUMatrix enc(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                FheUint2* ct = nullptr;
                uint8_t val = plain.get(i, j) ? 1u : 0u;
                if (fhe_uint2_try_encrypt_with_client_key_u8(val, ckey, &ct) != 0)
                    throw std::runtime_error("TFHEGPUBackend::encrypt failed");
                enc.set(i, j, FheUint2Ptr(ct));
            }
        }
        return enc;
    }

    // Decrypt a TFHEGPUMatrix back to a plaintext BoolMatrix.
    static BoolMatrix decrypt(const TFHEGPUMatrix& enc, ClientKey* ckey) {
        const std::size_t n = enc.dim();
        BoolMatrix plain(n, n, false);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                uint8_t val = 0;
                if (fhe_uint2_decrypt(enc.get(i, j), ckey, &val) != 0)
                    throw std::runtime_error("TFHEGPUBackend::decrypt failed");
                plain.set(i, j, val != 0);
            }
        }
        return plain;
    }
};

// Helper: build a CudaServerKey from a ClientKey.
// Returns owning pointer; call set_cuda_server_key(ptr.get()) before running.
inline CudaServerKeyPtr make_cuda_server_key(ClientKey* ckey) {
    CompressedServerKey* csk_raw = nullptr;
    if (compressed_server_key_new(ckey, &csk_raw) != 0)
        throw std::runtime_error("make_cuda_server_key: compressed key generation failed");
    CompressedServerKeyPtr csk(csk_raw);

    CudaServerKey* gpu_key = nullptr;
    if (compressed_server_key_decompress_to_gpu(csk.get(), &gpu_key) != 0)
        throw std::runtime_error("make_cuda_server_key: GPU decompression failed");
    return CudaServerKeyPtr(gpu_key);
}

} // namespace btc

#endif // TC_WITH_TFHE_GPU
