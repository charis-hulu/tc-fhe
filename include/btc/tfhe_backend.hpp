#pragma once

#ifdef TC_WITH_TFHE

#include "matrix.hpp"
#include <tfhe.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace btc {

namespace detail {

struct CtDeleter {
    void operator()(BooleanCiphertext* p) const noexcept {
        if (p) boolean_destroy_ciphertext(p);
    }
};

} // namespace detail

using CtPtr = std::unique_ptr<BooleanCiphertext, detail::CtDeleter>;

// Matrix of BooleanCiphertext pointers (row-major, square).
class TFHEMatrix {
public:
    explicit TFHEMatrix(std::size_t n) : n_(n), data_(n * n) {}

    std::size_t dim() const noexcept { return n_; }

    BooleanCiphertext* get(std::size_t i, std::size_t j) const noexcept {
        return data_[i * n_ + j].get();
    }

    void set(std::size_t i, std::size_t j, CtPtr ct) {
        data_[i * n_ + j] = std::move(ct);
    }

private:
    std::size_t n_;
    std::vector<CtPtr> data_;
};

// TFHEBackend — satisfies the Backend concept using TFHE-rs BooleanCiphertext.
//
// Usage:
//   BooleanClientKey* ckey = nullptr;
//   BooleanServerKey* skey = nullptr;
//   boolean_gen_keys_with_default_parameters(&ckey, &skey);
//
//   TFHEBackend backend(skey);
//   auto enc_A = TFHEBackend::encrypt(plain_A, ckey);
//   auto enc_S = btc::bounded_transitive_closure(enc_A, T, backend);
//   auto plain_S = TFHEBackend::decrypt(enc_S, ckey);
//
//   boolean_destroy_client_key(ckey);
//   boolean_destroy_server_key(skey);
class TFHEBackend {
public:
    using matrix_type = TFHEMatrix;

    explicit TFHEBackend(BooleanServerKey* skey) : skey_(skey) {}

    // C[i][j] = OR_k ( A[i][k] AND B[k][j] )
    TFHEMatrix matmul(const TFHEMatrix& A, const TFHEMatrix& B) const {
        const std::size_t n = A.dim();
        TFHEMatrix C(n);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                CtPtr acc;
                for (std::size_t k = 0; k < n; ++k) {
                    BooleanCiphertext* term_raw = nullptr;
                    if (boolean_server_key_and(skey_, A.get(i, k), B.get(k, j), &term_raw) != 0)
                        throw std::runtime_error("TFHEBackend::matmul: AND failed");
                    CtPtr term(term_raw);

                    if (!acc) {
                        acc = std::move(term);
                    } else {
                        BooleanCiphertext* or_raw = nullptr;
                        if (boolean_server_key_or(skey_, acc.get(), term.get(), &or_raw) != 0)
                            throw std::runtime_error("TFHEBackend::matmul: OR failed");
                        acc = CtPtr(or_raw);
                    }
                }
                C.set(i, j, std::move(acc));
            }
        }
        return C;
    }

    TFHEMatrix or_mat(const TFHEMatrix& A, const TFHEMatrix& B) const {
        const std::size_t n = A.dim();
        TFHEMatrix C(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                BooleanCiphertext* r = nullptr;
                if (boolean_server_key_or(skey_, A.get(i, j), B.get(i, j), &r) != 0)
                    throw std::runtime_error("TFHEBackend::or_mat: OR failed");
                C.set(i, j, CtPtr(r));
            }
        }
        return C;
    }

    // No direct clone in the C API — round-trip through serialization.
    TFHEMatrix copy(const TFHEMatrix& A) const {
        const std::size_t n = A.dim();
        TFHEMatrix C(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                DynamicBuffer buf{};
                if (boolean_serialize_ciphertext(A.get(i, j), &buf) != 0)
                    throw std::runtime_error("TFHEBackend::copy: serialize failed");
                DynamicBufferView view{buf.pointer, buf.length};
                BooleanCiphertext* ct = nullptr;
                int rc = boolean_deserialize_ciphertext(view, &ct);
                if (buf.destructor) buf.destructor(buf.pointer, buf.length);
                if (rc != 0)
                    throw std::runtime_error("TFHEBackend::copy: deserialize failed");
                C.set(i, j, CtPtr(ct));
            }
        }
        return C;
    }

    std::size_t dim(const TFHEMatrix& A) const { return A.dim(); }

    // Encrypt a plaintext BoolMatrix into a TFHEMatrix.
    static TFHEMatrix encrypt(const BoolMatrix& plain, BooleanClientKey* ckey) {
        const std::size_t n = plain.rows();
        TFHEMatrix enc(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                BooleanCiphertext* ct = nullptr;
                if (boolean_client_key_encrypt(ckey, plain.get(i, j), &ct) != 0)
                    throw std::runtime_error("TFHEBackend::encrypt failed");
                enc.set(i, j, CtPtr(ct));
            }
        }
        return enc;
    }

    // Decrypt a TFHEMatrix back to a plaintext BoolMatrix.
    static BoolMatrix decrypt(const TFHEMatrix& enc, BooleanClientKey* ckey) {
        const std::size_t n = enc.dim();
        BoolMatrix plain(n, n, false);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                bool val = false;
                if (boolean_client_key_decrypt(ckey, enc.get(i, j), &val) != 0)
                    throw std::runtime_error("TFHEBackend::decrypt failed");
                plain.set(i, j, val);
            }
        }
        return plain;
    }

private:
    BooleanServerKey* skey_; // non-owning
};

} // namespace btc

#endif // TC_WITH_TFHE
