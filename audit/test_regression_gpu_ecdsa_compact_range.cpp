/* ============================================================================
 * UltrafastSecp256k1 -- GPU ECDSA compact-signature strict-range regression
 * ============================================================================
 * Consensus-relevant finding: for a compact ECDSA signature whose r or s is
 * OUT of the scalar range [1, n-1] (i.e. r >= n or s >= n), the verification
 * verdict was NOT uniform across the library's own paths:
 *
 *   CUDA  ecdsa_verify_batch_kernel     strict  (ecdsa_sig_parse_compact_strict)
 *   CUDA  ecdsa_verify_collect_kernel   permissive (host bytes_to_ecdsa_sig +
 *           ecdsa_verify, which only rejected zero scalars and reduced the
 *           limbs mod n)
 *   Metal ecdsa_verify_batch_compressed permissive (same mod-n reduction)
 *   Metal lbtc_ecdsa_verify_collect     permissive (same mod-n reduction)
 *   OpenCL ecdsa_verify_compressed/rows strict  (lbtc_parse_compact_signature)
 *   CPU   ufsecp_ecdsa_verify           strict  (parse_compact_strict + low-S)
 *
 * For an encoding (r, s+n) with s+n < 2^256 the scalar s+n is congruent to s
 * mod n, so the permissive paths returned the SAME verdict as for the valid
 * (r, s) -- i.e. they accepted a non-canonical compact encoding that the
 * strict paths reject. Ubiquitous txs use low-S, and ufsecp_ecdsa_sign only
 * emits canonical r/s < n, so a valid base with s < 2^256 - n is infeasible to
 * synthesize; the divergence is therefore mostly theoretical in the wild BUT
 * it still breaks the documented invariant "collect EC verdict is bit-identical
 * to verify_batch" (gpu_backend_cuda.cu, ecdsa_verify_collect_kernel) and the
 * analogous invariant in the Metal shaders.
 *
 * FIX: each backend's device-side ecdsa_verify() entry point now rejects
 * r >= n or s >= n up front (single choke point shared by single/batch/collect
 * and the sign-and-verify countermeasure path), restoring one uniform strict
 * compact contract across CPU, CUDA, OpenCL and Metal.
 *
 * This module pins the fix two ways:
 *   [A] CPU-only SOURCE GATE (always runs, every runner): scans the in-tree
 *       kernel sources and asserts the strict range guard is present in each
 *       backend's ecdsa_verify() and that the CUDA batch kernel keeps its
 *       strict parse. This FAILED against the pre-fix sources, so it is the
 *       real always-bit-rot regression gate.
 *   [B] On-device BOUNDARY-SCALAR differential (advisory; self-skips when no
 *       GPU backend is available): feeds r/s on the {0, n-1, n, 2^256-1,
 *       s+n-congruent class} boundaries through BOTH the batch and collect
 *       entrypoints and asserts per-row uniformity with the CPU strict oracle.
 *
 * PUBLIC-DATA inputs only; verify is variable-time by design. No GPU needed to
 * build/run the source gate; the on-device part self-skips cleanly (operational
 * errors surface as SKIP, never a false PASS), mirroring
 * test_gpu_collect_verify_parity.
 * ============================================================================ */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"
#include "audit_check.hpp"

static int g_pass = 0;
static int g_fail = 0;

namespace {

// Group order n, big-endian 32 bytes (0xFFFFFFFF...D0364141).
static const uint8_t kOrderBe[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x41
};

// n-1, big-endian.
static const uint8_t kOrderMinus1Be[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE,
    0xBA, 0xAE, 0xDC, 0xE6, 0xAF, 0x48, 0xA0, 0x3B,
    0xBF, 0xD2, 0x5E, 0x8C, 0xD0, 0x36, 0x41, 0x40
};

// 2^256 - 1, big-endian (all ones; > n).
static const uint8_t kMax32Be[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

static const uint8_t kZero32[32] = {0};

// s := s + n mod 2^256 (big-endian 32-byte add). Returns false if it would
// overflow 32 bytes (encoding not representable). This is the s-congruent
// malleation class; only representable when s < 2^256 - n.
bool be_add_order(const uint8_t* s, uint8_t out[32]) {
    unsigned carry = 0;
    for (int i = 31; i >= 0; --i) {
        unsigned v = (unsigned)s[i] + kOrderBe[i] + carry;
        out[i] = (uint8_t)v;
        carry = v >> 8;
    }
    return carry == 0;
}

// A backend runtime error (or a backend without the native entrypoints) means
// "not assertable on this runner" -> treat as skip, never a false FAIL.
bool is_skip_err(ufsecp_error_t e) {
    return e == UFSECP_ERR_GPU_LAUNCH || e == UFSECP_ERR_GPU_MEMORY ||
           e == UFSECP_ERR_GPU_BACKEND || e == UFSECP_ERR_GPU_QUEUE ||
           e == UFSECP_ERR_GPU_DEVICE || e == UFSECP_ERR_GPU_UNSUPPORTED;
}

void copy32(const uint8_t src[32], uint8_t dst[32]) { std::memcpy(dst, src, 32); }

// ---------------------------------------------------------------------------
// [A] CPU-only source gate -- runs on EVERY runner, no GPU required. Each
//     backend's device ecdsa_verify() must carry the strict >= n guard; the
//     CUDA batch kernel must keep its strict compact parse. These CHECKs fail
//     against the pre-fix sources, so this section trees the regression even
//     on CPU-only CI (no GPU ever exercised it).
// ---------------------------------------------------------------------------
void test_source_gate() {
    AUDIT_LOG("[gpu_ecdsa_compact_range] source gate (CPU-only, always runs)\n");

    // -- CUDA: device ecdsa_verify() reject r/s >= n (ecdsa.cuh), and the
    //    batch kernel keeps the strict compact parse (secp256k1.cu). ------
    {
        std::string cuh = audit_read_source_file("src/cuda/include/ecdsa.cuh");
        CHECK(!cuh.empty(), "src/cuda/include/ecdsa.cuh must be readable (in-tree source)");
        if (!cuh.empty()) {
            CHECK(cuh.find("scalar_ge(&sig->r, ORDER)") != std::string::npos,
                  "[A] CUDA ecdsa_verify rejects r >= n (scalar_ge guard)");
            CHECK(cuh.find("scalar_ge(&sig->s, ORDER)") != std::string::npos,
                  "[A] CUDA ecdsa_verify rejects s >= n (scalar_ge guard)");
        }
        std::string cu = audit_read_source_file("src/cuda/src/secp256k1.cu");
        CHECK(!cu.empty(), "src/cuda/src/secp256k1.cu must be readable (in-tree source)");
        if (!cu.empty()) {
            CHECK(cu.find("ecdsa_sig_parse_compact_strict") != std::string::npos,
                  "[A] CUDA batch kernel keeps strict compact parse (parse_compact_strict)");
        }
    }

    // -- Metal: ecdsa_verify() reject r/s >= n (secp256k1_extended.h) via the
    //    scalar256_ge_n() helper that compares limbs against SECP256K1_N. --
    {
        std::string mh = audit_read_source_file("src/metal/shaders/secp256k1_extended.h");
        CHECK(!mh.empty(), "src/metal/shaders/secp256k1_extended.h must be readable (in-tree source)");
        if (!mh.empty()) {
            CHECK(mh.find("inline bool scalar256_ge_n") != std::string::npos,
                  "[A] Metal defines scalar256_ge_n() range helper");
            CHECK(mh.find("scalar256_ge_n(sig.r)") != std::string::npos,
                  "[A] Metal ecdsa_verify rejects r >= n (scalar256_ge_n guard)");
            CHECK(mh.find("scalar256_ge_n(sig.s)") != std::string::npos,
                  "[A] Metal ecdsa_verify rejects s >= n (scalar256_ge_n guard)");
        }
    }

    // -- OpenCL: ecdsa_verify_impl() reject r/s >= n (secp256k1_extended.cl)
    //    using the same lbtc_scalar_ge_order() primitive the strict compact
    //    parser already uses. ----------------------------------------------
    {
        std::string cl = audit_read_source_file("src/opencl/kernels/secp256k1_extended.cl");
        CHECK(!cl.empty(), "src/opencl/kernels/secp256k1_extended.cl must be readable (in-tree source)");
        if (!cl.empty()) {
            CHECK(cl.find("lbtc_scalar_ge_order(&sig->r)") != std::string::npos,
                  "[A] OpenCL ecdsa_verify_impl rejects r >= n (lbtc_scalar_ge_order guard)");
            CHECK(cl.find("lbtc_scalar_ge_order(&sig->s)") != std::string::npos,
                  "[A] OpenCL ecdsa_verify_impl rejects s >= n (lbtc_scalar_ge_order guard)");
        }
    }
}

// ---------------------------------------------------------------------------
// [B] On-device boundary-scalar differential (advisory; self-skips w/o GPU).
//     Builds one valid compressed compact row plus a family of boundary
//     encodings and asserts the batch/collect per-row verdicts are uniform with
//     the CPU strict oracle (ufsecp_ecdsa_verify: parse_compact_strict + low-S).
// ---------------------------------------------------------------------------
struct Row {
    const char* label;
    bool       want_valid;   // expected per the CPU strict oracle
};

// Deterministically produce a LOW-S compact signature accepted by the CPU
// strict oracle. Returns false if signing/verification infra is unavailable.
bool make_valid_row(ufsecp_ctx* sc, uint8_t msg[32], uint8_t pub33[33],
                    uint8_t sig64[64]) {
    for (int seed = 0; seed < 32; ++seed) {
        uint8_t sk[32] = {0};
        sk[31] = (uint8_t)(seed * 7 + 3);
        sk[30] = (uint8_t)(seed * 13 + 1);
        for (int j = 0; j < 32; ++j)
            msg[j] = (uint8_t)(seed * 29 + j * 11 + 5);
        if (ufsecp_pubkey_create(sc, sk, pub33) != UFSECP_OK) return false;
        if (ufsecp_ecdsa_sign(sc, msg, sk, sig64) != UFSECP_OK) continue;
        if (ufsecp_ecdsa_verify(sc, msg, sig64, pub33) == UFSECP_OK) return true;
    }
    return false;
}

// Build the boundary corpus and run batch+collect on one backend id.
void run_backend(uint32_t bid) {
    ufsecp_gpu_ctx* ctx = nullptr;
    if (ufsecp_gpu_ctx_create(&ctx, bid, 0) != UFSECP_OK || !ctx) {
        AUDIT_LOG("  (%s: ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        return;
    }
    ufsecp_ctx* sc = nullptr;
    if (ufsecp_ctx_create(&sc) != UFSECP_OK || !sc) {
        AUDIT_LOG("  (%s: cpu ctx create failed -- skipping)\n", ufsecp_gpu_backend_name(bid));
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }
    AUDIT_LOG("  Backend: %s\n", ufsecp_gpu_backend_name(bid));

    uint8_t msg[32], pub[33], base[64];
    if (!make_valid_row(sc, msg, pub, base)) {
        AUDIT_LOG("  (ecdsa signing infra unavailable -- skip boundary differential)\n");
        ufsecp_ctx_destroy(sc);
        ufsecp_gpu_ctx_destroy(ctx);
        return;
    }

    // Row corpus: (label, expected-by-CPU-strict-oracle).
    std::vector<Row> rows;
    rows.push_back({"valid base", true});
    rows.push_back({"s + n (congruent malleation; representable iff s < 2^256-n)", false});
    rows.push_back({"r = n", false});
    rows.push_back({"s = n", false});
    rows.push_back({"r = n-1", false});
    rows.push_back({"s = n-1", false});
    rows.push_back({"r = 0", false});
    rows.push_back({"s = 0", false});
    rows.push_back({"r = 2^256-1", false});
    rows.push_back({"s = 2^256-1", false});

    const size_t M = rows.size();
    std::vector<uint8_t> dig(M * 32), pks(M * 33), sigs(M * 64);

    // The s+n congruent-malleation row is representable only when the base
    // s < 2^256 - n (a random canonical s is effectively never that small).
    // When it does not fit we still exercise the row slot with a definite
    // > n encoding and assert uniformity; the flag only drives the label.
    bool has_sn_row = false;
    for (size_t i = 0; i < M; ++i) {
        std::memcpy(&dig[i * 32], msg, 32);
        std::memcpy(&pks[i * 33], pub, 33);
        uint8_t r[32], s[32];
        copy32(base, r);
        copy32(base + 32, s);
        switch (i) {
            case 0: break;  // base (valid low-S)
            case 1: has_sn_row = be_add_order(base + 32, s); break;  // s := s + n
            case 2: copy32(kOrderBe, r); break;
            case 3: copy32(kOrderBe, s); break;
            case 4: copy32(kOrderMinus1Be, r); break;
            case 5: copy32(kOrderMinus1Be, s); break;
            case 6: copy32(kZero32, r); break;
            case 7: copy32(kZero32, s); break;
            case 8: copy32(kMax32Be, r); break;
            case 9: copy32(kMax32Be, s); break;
            default: break;
        }
        if (i == 1 && !has_sn_row) copy32(kMax32Be, s);  // s >= n either way
        std::memcpy(&sigs[i * 64], r, 32);
        std::memcpy(&sigs[i * 64 + 32], s, 32);

        // CPU strict oracle for this row.
        rows[i].want_valid = (ufsecp_ecdsa_verify(sc, msg, &sigs[i * 64], pub) == UFSECP_OK);
    }

    // Sanity: the oracle itself is strict (base valid, malleation rejected).
    CHECK(rows[0].want_valid, "[B] CPU oracle accepts the valid base row (strict parse)");
    if (has_sn_row)
        CHECK(!rows[1].want_valid, "[B] CPU oracle rejects the s+n congruent-malleation row (strict parse)");
    else
        AUDIT_LOG("  (note: base s >= 2^256-n, s+n malleation unrealizable at this seed; row pinned with s>=n)\n");

    // ---- batch ----
    std::vector<uint8_t> batch(M, 2);
    ufsecp_error_t eb = ufsecp_gpu_ecdsa_verify_batch(ctx, dig.data(), pks.data(), sigs.data(), M, batch.data());
    if (is_skip_err(eb)) { AUDIT_LOG("  (ecdsa batch: runtime skip %d (%s))\n", eb, ufsecp_gpu_error_str(eb)); ufsecp_ctx_destroy(sc); ufsecp_gpu_ctx_destroy(ctx); return; }
    CHECK(eb == UFSECP_OK, "[B] verify_batch boundary corpus -> OK");

    // ---- collect ----
    const uint8_t kSeed = 0xEE;
    std::vector<uint8_t> key(M, kSeed);
    ufsecp_error_t ec = ufsecp_gpu_ecdsa_verify_collect(ctx, dig.data(), pks.data(), sigs.data(), M, key.data());
    if (is_skip_err(ec)) { AUDIT_LOG("  (ecdsa collect: runtime skip %d (%s))\n", ec, ufsecp_gpu_error_str(ec)); ufsecp_ctx_destroy(sc); ufsecp_gpu_ctx_destroy(ctx); return; }
    CHECK(ec == UFSECP_OK, "[B] ecdsa_verify_collect boundary corpus -> OK");

    if (eb == UFSECP_OK && ec == UFSECP_OK) {
        bool batch_match = true, collect_match = true, per_row = true, seeded = true;
        for (size_t i = 0; i < M; ++i) {
            const bool ok = rows[i].want_valid;
            if (batch[i] != (ok ? 1u : 0u)) batch_match = false;
            if ((key[i] == 0) != ok) collect_match = false;
            if ((key[i] == 0) != (batch[i] == 1)) per_row = false;
            if (!ok && key[i] != kSeed) seeded = false;   // invalid row stays seeded
        }
        CHECK(batch_match,   "[B] batch verdict == CPU strict oracle per boundary row");
        CHECK(collect_match, "[B] collect verdict == CPU strict oracle per boundary row");
        CHECK(per_row,       "[B] collect == verify_batch verdict per boundary row");
        CHECK(seeded,        "[B] invalid rows are left at the seeded marker (fail-closed)");
    }

    ufsecp_ctx_destroy(sc);
    ufsecp_gpu_ctx_destroy(ctx);
}

void test_backend_differential() {
    AUDIT_LOG("[gpu_ecdsa_compact_range] boundary-scalar differential (if GPU available)\n");
    uint32_t ids[8] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 8);
    bool any = false;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { any = true; run_backend(ids[i]); }
    }
    if (!any) AUDIT_LOG("  (no GPU available -- skipping on-device differential)\n");
}

}  // namespace

int test_regression_gpu_ecdsa_compact_range_run() {
    g_pass = 0; g_fail = 0;
    AUDIT_LOG("=== GPU ECDSA compact-sig strict-range regression ===\n");
    test_source_gate();
    test_backend_differential();
    AUDIT_LOG("[gpu_ecdsa_compact_range] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_gpu_ecdsa_compact_range_run(); }
#endif