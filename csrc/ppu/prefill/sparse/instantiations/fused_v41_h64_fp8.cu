// FP8-compute five-in-one prefill instantiation (h64).
//
// Mirrors decode's fused_v41_h64_fp8.cu: the whole TU is raised to
// ACOMPUTE_VERSION 10500 so the PPU1.5 E4M3 MMA atom
// (PPU0015_16x16x32_F32E4M3E4M3F32_TN) and its swizzled staging compile with
// the PPU1.5 configuration.  The wheel is otherwise built with
// -DACOMPUTE_VERSION=10000 (see the decode TU's header comment for the
// "silently degrades to the PPU1.0 configuration" trap).
//
// The entry point takes the same core_attn::Params as the BF16 prefill
// five-in-one, except that kv points at the API-quantized E4M3 workspace
// (the caller swaps it in before dispatch; see
// fused_norm_rope_attn_rope_cast_fwd.cpp).  Only the H64 / per-token-scale /
// sm89 combination is instantiated; the API entry rejects anything else so a
// stray flag cannot silently change kernels on a path that has no FP8
// variant (the decode five-in-one's fp8_compute contract).
#if defined(ACOMPUTE_VERSION)
#undef ACOMPUTE_VERSION
#endif
#define ACOMPUTE_VERSION 10500

#include "api/common.h"
#include "ppu/sparse/fused_v41.h"
#include "ppu/prefill/sparse/fused_v41_prefill_fp8.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn {

template<bool EnableNorm>
void run_fused_norm_rope_attn_rope_cast_prefill_hs64_fp8(const Params& p) {
    TORCH_CHECK(p.mode == FwdMode::Prefill && p.arch == 89 && p.h_q == 64,
                "fp8_compute requires the H64 five-in-one prefill path on sm89");
    TORCH_CHECK(p.per_token_scale,
                "fp8_compute requires per-token FP32 output scales (num_per_channels=4096)");
    TORCH_CHECK(p.topk >= 0 && p.topk <= 5312,
                "fp8_compute prefill topk must be within [0, 5312]");
    TORCH_CHECK(p.stride_kv_s_kv == 512,
                "fp8_compute expects the dense E4M3 workspace (row stride 512)");
    TORCH_CHECK(p.wv_group_size == 8 && p.n_wv_group == 8);
    if (p.s_q == 0) return;
    auto* out_sf = reinterpret_cast<float*>(p.out_sf);
    constexpr int BM = 64;
    if (p.topk <= 2176) {
        fp8_prefill::launch_fused_prefill_fp8<BM, EnableNorm, 2176>(
            p.q, p.kv, p.indices, p.topk_length, p.attn_sink, p.token_positions, p.cos_sin_cache,
            p.out_fp8, out_sf, p.max_logits, p.lse, p.sm_scale_log2, p.rms_norm_eps, p.topk,
            p.s_kv, p.s_q, p.n_wv_group, p.stride_q_s_q, p.stride_indices_s_q, p.stream);
    } else {
        // Wide sparse rows keep the same kernel; only the index staging grows,
        // which drops the launch to one CTA per CU.
        fp8_prefill::launch_fused_prefill_fp8<BM, EnableNorm, 5312>(
            p.q, p.kv, p.indices, p.topk_length, p.attn_sink, p.token_positions, p.cos_sin_cache,
            p.out_fp8, out_sf, p.max_logits, p.lse, p.sm_scale_log2, p.rms_norm_eps, p.topk,
            p.s_kv, p.s_q, p.n_wv_group, p.stride_q_s_q, p.stride_indices_s_q, p.stream);
    }
}

template void run_fused_norm_rope_attn_rope_cast_prefill_hs64_fp8<false>(const Params&);
template void run_fused_norm_rope_attn_rope_cast_prefill_hs64_fp8<true>(const Params&);

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::core_attn
