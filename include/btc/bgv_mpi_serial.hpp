#pragma once

#ifdef TC_WITH_BGV_MPI

// ── Ciphertext <-> byte-buffer serialization for MPI transport ────────────
//
// OpenFHE's Serial::Serialize/Deserialize (openfhe/core/utils/serial.h)
// operate on std::ostream/std::istream, not directly on byte buffers or MPI
// calls -- this file is the small adapter layer between "an OpenFHE
// Ciphertext" and "a std::vector<char> an MPI_Send/Recv can move", used by
// bgv_distributed_rowcol_backend.hpp's row/column assembly phase.
//
// Binary (SerType::SERBINARY), not JSON: this is wire format between
// compute ranks, not a human-readable artifact, and binary is both smaller
// and faster to (de)serialize -- there is no reason to pay JSON's overhead
// here.
//
// ── Communication batching (Stage 3) ───────────────────────────────────────
//
// SendCiphertext/RecvCiphertext/BroadcastCiphertext below move ONE
// ciphertext per call (two messages each: an int64 size, then the payload).
// SerializeCiphertextBatch/DeserializeCiphertextBatch and their Send/Recv/
// Broadcast wrappers pack MANY (tag, ciphertext) pairs into a single byte
// blob so a whole row/column-assembly round trip between two ranks (or one
// broadcast root and its subcommunicator) costs one message pair instead of
// one per ciphertext -- see bgv_distributed_rowcol_backend.hpp's
// AssembleRows/AssembleCols/or_mat/GatherToRankWithinGrid/
// ScatterFromRankToGrid, all of which now batch by owner/root instead of
// looping per-row or per-column. This is still not MPI_Alltoallv/Gatherv (a
// true collective would let MPI's own implementation choose the message
// schedule); it is one explicit point-to-point or broadcast message per
// (source, destination-group) pair, which is the actual bottleneck this
// project's N=4/N=8 test sizes hit (message COUNT, not per-message size).

#include "bgv_batched.hpp"

// openfhe.h (included transitively via bgv_batched.hpp) does NOT pull in
// OpenFHE's Cereal type-registration headers -- serialization support is a
// separate opt-in, per OpenFHE's own comment in cryptocontext-ser.h ("the
// routines below are only instantiated if the user includes the
// appropriate serialize-*.h file"). Without these, Serial::Serialize on a
// CryptoContext/Ciphertext/key throws cereal::Exception at runtime
// ("Trying to save an unregistered polymorphic type") instead of failing
// to compile -- verified empirically the hard way while building this
// backend (see git history of this file).
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"

#include <mpi.h>

#include <cstdint>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace btc {
namespace bgv_mpi {

// ── Ciphertext byte serialization ──────────────────────────────────────────

inline std::string SerializeCiphertext(const bgv_batched::Ciphertext& ct) {
    std::ostringstream oss(std::ios::binary);
    lbcrypto::Serial::Serialize(ct, oss, lbcrypto::SerType::SERBINARY());
    return oss.str();
}

inline bgv_batched::Ciphertext DeserializeCiphertext(const std::string& bytes) {
    std::istringstream iss(bytes, std::ios::binary);
    bgv_batched::Ciphertext ct;
    lbcrypto::Serial::Deserialize(ct, iss, lbcrypto::SerType::SERBINARY());
    return ct;
}

// ── Batched ciphertext serialization (many tagged ciphertexts, one blob) ──
//
// Wire format: int64 count, then `count` repetitions of
// { int64 tag; int64 payload_size; payload_size bytes }. `tag` carries the
// caller's own identifier for each ciphertext (a global row/column index in
// every call site here) so the receiver can place each entry back where it
// belongs without a separate side-channel message.
inline std::string SerializeCiphertextBatch(const std::vector<std::size_t>& tags,
                                             const std::vector<bgv_batched::Ciphertext>& cts) {
    if (tags.size() != cts.size())
        throw std::invalid_argument("SerializeCiphertextBatch: tags/cts size mismatch");
    std::ostringstream oss(std::ios::binary);
    const auto count = static_cast<std::int64_t>(cts.size());
    oss.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (std::size_t k = 0; k < cts.size(); ++k) {
        const auto tag = static_cast<std::int64_t>(tags[k]);
        const std::string payload = SerializeCiphertext(cts[k]);
        const auto size = static_cast<std::int64_t>(payload.size());
        oss.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        oss.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }
    return oss.str();
}

inline void DeserializeCiphertextBatch(const std::string& bytes, std::vector<std::size_t>& tags,
                                        std::vector<bgv_batched::Ciphertext>& cts) {
    std::istringstream iss(bytes, std::ios::binary);
    std::int64_t count = 0;
    iss.read(reinterpret_cast<char*>(&count), sizeof(count));
    tags.assign(static_cast<std::size_t>(count), 0);
    cts.assign(static_cast<std::size_t>(count), bgv_batched::Ciphertext{});
    for (std::int64_t k = 0; k < count; ++k) {
        std::int64_t tag = 0, size = 0;
        iss.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        iss.read(reinterpret_cast<char*>(&size), sizeof(size));
        std::string payload(static_cast<std::size_t>(size), '\0');
        iss.read(payload.data(), static_cast<std::streamsize>(size));
        tags[static_cast<std::size_t>(k)] = static_cast<std::size_t>(tag);
        cts[static_cast<std::size_t>(k)] = DeserializeCiphertext(payload);
    }
}

// ── Crypto-context / key material distribution ────────────────────────────
//
// All compute ranks need an IDENTICAL CryptoContext (including EvalMult and
// rotation keys) to operate on ciphertexts produced by any other rank --
// OpenFHE ties keys to the specific CryptoContext object instance, not just
// its logical parameters. Rank 0 builds the context (see
// BGVDistributedRowColBackend's constructor); every other rank must load the
// exact same serialized bytes rather than calling bgv_batched::setup() a
// second time (which would produce a DIFFERENT context object and
// incompatible keys even with identical Params).
//
// The secret key is deliberately NOT part of this bundle -- see the task's
// "secret key must not be required by compute ranks" requirement.
// BGVDistributedRowColBackend keeps it only on rank 0 for final decryption.
struct SerializedContextBundle {
    std::string crypto_context;
    std::string public_key;
    std::string eval_mult_key;
    std::string eval_automorphism_key; // rotation keys
};

inline SerializedContextBundle SerializeContextBundle(const bgv_batched::Context& ctx) {
    SerializedContextBundle bundle;
    {
        std::ostringstream oss(std::ios::binary);
        lbcrypto::Serial::Serialize(ctx.crypto_context, oss, lbcrypto::SerType::SERBINARY());
        bundle.crypto_context = oss.str();
    }
    {
        std::ostringstream oss(std::ios::binary);
        lbcrypto::Serial::Serialize(ctx.public_key, oss, lbcrypto::SerType::SERBINARY());
        bundle.public_key = oss.str();
    }
    {
        std::ostringstream oss(std::ios::binary);
        ctx.crypto_context->SerializeEvalMultKey(oss, lbcrypto::SerType::SERBINARY());
        bundle.eval_mult_key = oss.str();
    }
    {
        std::ostringstream oss(std::ios::binary);
        ctx.crypto_context->SerializeEvalAutomorphismKey(oss, lbcrypto::SerType::SERBINARY());
        bundle.eval_automorphism_key = oss.str();
    }
    return bundle;
}

// Reconstructs a Context on a non-owning rank from a bundle built by
// SerializeContextBundle. `secret_key` is left default-constructed (null) --
// compute ranks must never decrypt; only the rank that originally called
// bgv_batched::setup() retains the secret key.
inline bgv_batched::Context DeserializeContextBundle(const SerializedContextBundle& bundle) {
    bgv_batched::Context ctx;

    // Serial::Deserialize's CryptoContext<T> overload (cryptocontext-ser.h)
    // is special-cased: it deserializes into a temporary, then resolves the
    // real object via CryptoContextFactory<T>::GetContext(...), which
    // interns/deduplicates by parameters -- OpenFHE deliberately does not
    // want multiple live CryptoContextImpl instances for the same logical
    // context. EvalMultKey/EvalAutomorphismKey are stored in a global map
    // keyed off the resolved context, so this MUST run before the two key
    // deserializations below (they look the key up by the context they're
    // called on).
    {
        std::istringstream iss(bundle.crypto_context, std::ios::binary);
        lbcrypto::Serial::Deserialize(ctx.crypto_context, iss, lbcrypto::SerType::SERBINARY());
    }
    {
        std::istringstream iss(bundle.public_key, std::ios::binary);
        lbcrypto::Serial::Deserialize(ctx.public_key, iss, lbcrypto::SerType::SERBINARY());
    }
    {
        std::istringstream iss(bundle.eval_mult_key, std::ios::binary);
        ctx.crypto_context->DeserializeEvalMultKey(iss, lbcrypto::SerType::SERBINARY());
    }
    {
        std::istringstream iss(bundle.eval_automorphism_key, std::ios::binary);
        ctx.crypto_context->DeserializeEvalAutomorphismKey(iss, lbcrypto::SerType::SERBINARY());
    }

    return ctx;
}

// ── MPI byte-buffer transport for a single serialized blob ────────────────
//
// Thin wrappers so call sites in bgv_distributed_rowcol_backend.hpp read as
// "send/recv a ciphertext" rather than manually managing size-then-payload
// two-message protocols at every call site.

inline void SendBytes(const std::string& bytes, int dest, int tag, MPI_Comm comm) {
    auto size = static_cast<std::int64_t>(bytes.size());
    MPI_Send(&size, 1, MPI_INT64_T, dest, tag, comm);
    if (size > 0)
        MPI_Send(bytes.data(), static_cast<int>(size), MPI_CHAR, dest, tag + 1, comm);
}

inline std::string RecvBytes(int source, int tag, MPI_Comm comm) {
    std::int64_t size = 0;
    MPI_Recv(&size, 1, MPI_INT64_T, source, tag, comm, MPI_STATUS_IGNORE);
    std::string bytes(static_cast<std::size_t>(size), '\0');
    if (size > 0)
        MPI_Recv(bytes.data(), static_cast<int>(size), MPI_CHAR, source, tag + 1, comm, MPI_STATUS_IGNORE);
    return bytes;
}

inline void BroadcastBytes(std::string& bytes, int root, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    auto size = static_cast<std::int64_t>(bytes.size());
    MPI_Bcast(&size, 1, MPI_INT64_T, root, comm);
    if (rank != root)
        bytes.resize(static_cast<std::size_t>(size));
    if (size > 0)
        MPI_Bcast(bytes.data(), static_cast<int>(size), MPI_CHAR, root, comm);
}

// ── Non-blocking byte transport (deadlock avoidance for peer-to-peer
//    exchanges) ─────────────────────────────────────────────────────────────
//
// SendBytes above is a BLOCKING send. That is fine for star-topology
// patterns (one root receiving from many senders, e.g. GatherToRankWithinGrid
// -- the root always eventually services every sender in its own receive
// loop, so no sender can be stuck waiting on another sender). It is NOT safe
// for a peer-to-peer exchange where multiple ranks may need to send to EACH
// OTHER and post all their sends before posting any receive (see
// bgv_distributed_rowcol_backend.hpp's AssembleOwnedBucket): if rank A sends
// to rank B while rank B simultaneously sends to rank A, and BOTH ranks
// finish their whole send loop before starting their receive loop, a message
// large enough to cross MPI's eager/rendezvous threshold makes BOTH
// MPI_Send calls block waiting for a matching MPI_Recv that neither rank has
// posted yet -- a real deadlock, not just added latency. This was reproduced
// empirically: 1-rank tests batching only trivial (self-only) exchanges
// passed, but the first real 2-rank exchange hung until PBS's walltime
// killed it.
//
// PendingSend owns the serialized byte buffer and size for one in-flight
// batched message (kept alive until Wait/WaitAll, since MPI_Isend does not
// copy the buffer) -- caller collects these into a std::deque (NOT a
// std::vector: push_back on a full vector can reallocate and invalidate the
// pointers already handed to MPI_Isend for in-flight requests; deque never
// invalidates references to existing elements on push_back).
struct PendingSend {
    std::int64_t size = 0;
    std::string bytes;
    MPI_Request size_request = MPI_REQUEST_NULL;
    MPI_Request payload_request = MPI_REQUEST_NULL;
};

inline void IsendBytes(std::string bytes, int dest, int tag, MPI_Comm comm, std::deque<PendingSend>& pending) {
    pending.emplace_back();
    PendingSend& p = pending.back();
    p.bytes = std::move(bytes);
    p.size = static_cast<std::int64_t>(p.bytes.size());
    MPI_Isend(&p.size, 1, MPI_INT64_T, dest, tag, comm, &p.size_request);
    if (p.size > 0)
        MPI_Isend(p.bytes.data(), static_cast<int>(p.size), MPI_CHAR, dest, tag + 1, comm, &p.payload_request);
}

// Waits for every send in `pending` to complete, then clears it. Must be
// called before `pending` goes out of scope (or before its PendingSend
// entries are otherwise destroyed) -- destroying a PendingSend with an
// outstanding MPI_Request is undefined behavior.
inline void WaitAllSends(std::deque<PendingSend>& pending) {
    for (auto& p : pending) {
        MPI_Wait(&p.size_request, MPI_STATUS_IGNORE);
        if (p.payload_request != MPI_REQUEST_NULL)
            MPI_Wait(&p.payload_request, MPI_STATUS_IGNORE);
    }
    pending.clear();
}

// Sends a ciphertext to `dest` within `comm`, tagged `tag`/`tag+1` (size then
// payload -- see SendBytes). Correctness-first point-to-point; batching
// multiple ciphertexts into one message is a Stage-3 optimization (see this
// file's header comment).
inline void SendCiphertext(const bgv_batched::Ciphertext& ct, int dest, int tag, MPI_Comm comm) {
    SendBytes(SerializeCiphertext(ct), dest, tag, comm);
}

inline bgv_batched::Ciphertext RecvCiphertext(int source, int tag, MPI_Comm comm) {
    return DeserializeCiphertext(RecvBytes(source, tag, comm));
}

inline void BroadcastCiphertext(bgv_batched::Ciphertext& ct, int root, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::string bytes = (rank == root) ? SerializeCiphertext(ct) : std::string();
    BroadcastBytes(bytes, root, comm);
    if (rank != root)
        ct = DeserializeCiphertext(bytes);
}

// Batched counterparts: one message (two MPI_Send/Recv/Bcast calls -- size
// then payload, see SendBytes/RecvBytes/BroadcastBytes) moves an entire
// tagged group of ciphertexts instead of one message pair per ciphertext.
inline void SendCiphertextBatch(const std::vector<std::size_t>& tags, const std::vector<bgv_batched::Ciphertext>& cts,
                                 int dest, int tag, MPI_Comm comm) {
    SendBytes(SerializeCiphertextBatch(tags, cts), dest, tag, comm);
}

inline void RecvCiphertextBatch(int source, int tag, MPI_Comm comm, std::vector<std::size_t>& tags,
                                 std::vector<bgv_batched::Ciphertext>& cts) {
    DeserializeCiphertextBatch(RecvBytes(source, tag, comm), tags, cts);
}

// Non-blocking counterpart of SendCiphertextBatch -- see IsendBytes/
// WaitAllSends's doc comment for why AssembleOwnedBucket needs this instead
// of the blocking version for its peer-to-peer exchange.
inline void IsendCiphertextBatch(const std::vector<std::size_t>& tags, const std::vector<bgv_batched::Ciphertext>& cts,
                                  int dest, int tag, MPI_Comm comm, std::deque<PendingSend>& pending) {
    IsendBytes(SerializeCiphertextBatch(tags, cts), dest, tag, comm, pending);
}

inline void BroadcastCiphertextBatch(std::vector<std::size_t>& tags, std::vector<bgv_batched::Ciphertext>& cts,
                                      int root, MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    std::string bytes = (rank == root) ? SerializeCiphertextBatch(tags, cts) : std::string();
    BroadcastBytes(bytes, root, comm);
    if (rank != root)
        DeserializeCiphertextBatch(bytes, tags, cts);
}

} // namespace bgv_mpi
} // namespace btc

#endif // TC_WITH_BGV_MPI
