// Fused five-in-one FP8 prefill kernel: Q RMSNorm + Q RoPE + sparse attention
// (E4M3 compute) + O RoPE + O cast to FP8, for DeepSeek-V4.1 on PPU1.5 (sm89).
//
// V523-S61 (prefill mega q8).  The attention core is the validated FP8 sparse
// prefill pipeline of flash_fwd_sparse_prefill_fp8_kernel (csrc/ppu/prefill/
// sparse/native_fp8.cu -- the flashmla_sparse_quantized_fwd kernel measured by
// V523-S59): one CTA per query token, E4M3 QK on the PPU0015
// 16x16x32 F32E4M3E4M3F32 atom, online softmax over double-buffered swizzled
// KV stages, P staged as E4M3 with P_SCALE, PV on the same atom reading the
// same KV stages through ppu_mat_trans_b8.
//
// The fused hooks follow the decode five-in-one's FP8 semantics
// (csrc/ppu/decode/sparse/fused_v41_splitkv.cuh):
//   * Q is quantized to E4M3 LAST, i.e. strictly after RoPE (and the optional
//     RMS norm is applied to the scores, like the decode kernel's
//     scale_fused_scores, which is exact because the norm is per-row scalar).
//   * K/V arrive as E4M3: the API quantizes the BF16 workspace once
//     (prepare_sparse_prefill_fp8_kv, the same deployment cost the sparse q8
//     arm pays) and the kernel stages E4M3 without dequantizing.
//   * The epilogue publishes per-WV-group FP32 scales (PerTokenScale) and the
//     fused WV output layout: out[token, group, tile*256 + in_group*32 + i]
//     holds O[head=group*8+in_group][col=tile*32+i] (verified against the
//     official fused test's dequant view).
//
// Output contract matches the BF16 five-in-one exactly (natural-log domain
// max_logits/lse, +inf lse / -inf max / zero out for empty rows, attn_sink
// folded into the denominator without touching lse/max).
#pragma once

#include "ppu/sparse_native_fp8_attention.cuh"

namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::fp8_prefill {

using namespace cute;
using flashmla_fp8_std::ScoreRows;

// The Q/K/V staging arithmetic below assumes 1-byte E4M3 codes.
static_assert(sizeof(E4M3) == 1, "fused FP8 staging assumes 1-byte E4M3");

static constexpr int FUSED_BN = 64, FUSED_DV = 512;
static constexpr float NEG_INF = -1e30f;
static constexpr float P_SCALE = 448.f;
static constexpr float LOG2_P_SCALE = 8.807354922f;
using FusedAtomS8 = decltype(composition(Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

__forceinline__ __device__ float2 fused_apply_rope(float2 x, float c, float s) {
    return float2{fmaf(x.x, c, -x.y * s), fmaf(x.x, s, x.y * c)};
}

template <int BlockM, int MaxTopk> struct FusedFp8PrefillTraits : ScoreRows {
    static_assert(BlockM == 64, "the fused FP8 prefill path instantiates H64 only");
    static constexpr int Threads = 512;
    static constexpr int QKWarpN = 4;
    static constexpr int QKThreads = (BlockM / 16) * QKWarpN * 32;
    using MmaQK = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<Int<BlockM / 16>, _4, _1>>, Tile<Int<BlockM>, Int<FUSED_BN>, _32>>;
    using MmaPV = TiledMMA<MMA_Atom<PPU0015_16x16x32_F32E4M3E4M3F32_TN>,
        Layout<Shape<_1, Int<Threads / 32>, _1>>, Tile<_16, Int<Threads / 2>, _32>>;
    using LayoutP = decltype(tile_to_shape(FusedAtomS8{}, Shape<Int<BlockM>, Int<FUSED_BN>>{}));
    using CopyA = Copy_Atom<PPU_TSM_LD_SWZL<E4M3, BlockM, 64, false, false, 1>, E4M3>;
    using CopyB = Copy_Atom<PPU_TSM_LD_SWZL<E4M3, FUSED_BN, 64, true, false, 1>, E4M3>;

    CUTE_HOST_DEVICE static constexpr int output_row(int e) {
        return ((e >> 3) % (BlockM / 16) * 16) + ((e >> 1) & 1) * 8;
    }
};

// Shared-memory bytes: Q tile + 2 KV stages + P + indices + softmax/norm scratch.
// Float scratch: sPmax/sPsum (4BM each), sRunMax/sRunSum/sScale/sInvDen,
// sGroupAmax (16 warps x BM/8), sGroupScale (BM/8), sFlag + pad, and for the
// norm arm sNorm (BM) + sNormPartial (BM*32 granule partials).
template <int BM, int MaxTopk, bool EnableNorm>
constexpr size_t fused_fp8_prefill_smem() {
    constexpr size_t kQ = size_t(BM) * 512;
    constexpr size_t kK = 2 * size_t(FUSED_BN) * 512;
    constexpr size_t kP = size_t(BM) * FUSED_BN;
    constexpr size_t kIdx = size_t(MaxTopk) * 4;
    constexpr size_t kFloats = 12 * size_t(BM) + 16 * (BM / 8) + (BM / 8) + 4
        + (EnableNorm ? size_t(BM) * 33 : 0);
    return kQ + kK + kP + kIdx + kFloats * 4;
}

template <int BM, bool EnableNorm, int MaxTopk>
__global__ void __launch_bounds__(FusedFp8PrefillTraits<BM, MaxTopk>::Threads,
    (MaxTopk <= 2176 ? 2 : 1))
fused_norm_rope_attn_rope_cast_prefill_fp8_kernel(
    const void *__restrict__ gQRaw,          // [s_q, h_q=64, 512] bf16, row stride q_stride
    const E4M3 *__restrict__ gKV,            // [s_kv, 512] e4m3 (API-quantized workspace)
    const int *__restrict__ gIdx,            // [s_q, topk], row stride idx_stride
    const int *__restrict__ gTopkLength,    // [s_q] or null
    const float *__restrict__ gAttnSink,    // [h_q] or null
    const int *__restrict__ gTokenPositions, // [s_q]
    const float *__restrict__ gCosSinCache,  // [max_pos, 64]
    uint8_t *__restrict__ gOutFp8,          // [s_q, n_wv_group, 8*512]
    float *__restrict__ gOutSf,              // [s_q, n_wv_group, 1] fp32
    float *__restrict__ gMaxLogits,         // [s_q, h_q]
    float *__restrict__ gLse,                // [s_q, h_q]
    float sm_scale_log2, float rms_norm_eps, int topk, int s_kv, int n_wv_group,
    int64_t q_stride, int64_t idx_stride) {
    using Traits = FusedFp8PrefillTraits<BM, MaxTopk>;
    constexpr int NTHR = Traits::Threads, QK_THREADS = Traits::QKThreads;
    constexpr int BN = FUSED_BN, DV = FUSED_DV;
    using MmaQK = typename Traits::MmaQK;
    using MmaPV = typename Traits::MmaPV;
    using LayoutP8 = typename Traits::LayoutP;
    using CopyA8 = typename Traits::CopyA;
    using CopyB8 = typename Traits::CopyB;
    static constexpr int NK_QK = 512 / 32;
    static constexpr int QTILE = BM * 512, KTILE = BN * 512;
    using LayoutQ8 = decltype(tile_to_shape(FusedAtomS8{}, Shape<Int<BM>, Int<512>>{}));
    using LayoutK8 = decltype(tile_to_shape(FusedAtomS8{}, Shape<Int<BN>, Int<512>>{}));

    extern __shared__ char smem[];
    E4M3 *pQ = reinterpret_cast<E4M3 *>(smem);
    E4M3 *pK = pQ + QTILE;
    E4M3 *pP = pK + 2 * KTILE;
    int *sIdx = reinterpret_cast<int *>(pP + BM * BN);
    float *sf = reinterpret_cast<float *>(sIdx + MaxTopk);
    float *sPmax = sf, *sPsum = sf + 4 * BM, *sRunMax = sf + 8 * BM;
    float *sRunSum = sRunMax + BM, *sScale = sRunSum + BM, *sInvDen = sScale + BM;
    float *sGroupAmax = sInvDen + BM;
    float *sGroupScale = sGroupAmax + 16 * (BM / 8);
    int *sFlag = reinterpret_cast<int *>(sGroupScale + (BM / 8));
    float *sNorm = sGroupScale + (BM / 8) + 4;  // one slot for sFlag, 3 padding
    float *sNormPartial = sNorm + BM;

    const int tid = threadIdx.x, wp = tid / 32, lane = tid % 32;
    const int q_token = blockIdx.x;
    const int h_q = BM;  // the H64 instantiation owns all heads in one tile
    int topk_length = gTopkLength ? gTopkLength[q_token] : topk;
    topk_length = topk_length < 0 ? 0 : topk_length;
    topk_length = topk_length > topk ? topk : topk_length;
    // Keep one all-invalid tile for length zero so output/auxiliary behavior
    // matches the BF16 five-in-one (empty rows: max=-inf, lse=+inf, out=0).
    const int active_tiles = topk_length > 0 ? (topk_length + BN - 1) / BN : 1;    const int wm = wp % (BM / 16), wn = (wp / (BM / 16)) % 4, mi = lane / 4;
    const int r0 = wm * 16 + mi, r1 = r0 + 8;

    Tensor sQt = make_tensor(make_smem_ptr(pQ), LayoutQ8{});
    Tensor sKt0 = make_tensor(make_smem_ptr(pK), LayoutK8{});
    Tensor sPt = make_tensor(make_smem_ptr(pP), LayoutP8{});

    MmaQK mqk;
    auto tqk = mqk.get_slice(tid % QK_THREADS);
    MmaPV mpv;
    auto tpv = mpv.get_slice(tid);
    Tensor gOt = make_tensor(make_gmem_ptr((float *)nullptr), Shape<Int<BM>, Int<DV>>{}, GenRowMajor{});
    Tensor acc_o = tpv.partition_fragment_C(gOt);
    clear(acc_o);

    auto tcQ = make_tiled_copy_A(CopyA8{}, mqk);
    auto thQ = tcQ.get_thread_slice(tid % QK_THREADS);
    auto tcK = make_tiled_copy_B(CopyB8{}, mqk);
    auto thK = tcK.get_thread_slice(tid % QK_THREADS);
    auto tcP = make_tiled_copy_A(CopyA8{}, mpv);
    auto thP = tcP.get_thread_slice(tid);

    Tensor srcQ = thQ.partition_S(make_mix_tensor_like(sQt));
    Tensor srcK_ = thK.partition_S(make_mix_tensor_like(sKt0));
    Tensor srcP = thP.partition_S(make_mix_tensor_like(sPt));

    const int qk_lbo_a = 1, qk_sbo_a = 32;
    const int qk_lbo_b = 32, qk_sbo_b = 1;
    const int qtsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pQ) / 16) + wm * 64);
    const int ktsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pK) / 16) + wn * 64);
    const int ptsm0 = UNIF((int)(reinterpret_cast<uintptr_t>(pP) / 16));
    const int kstage = KTILE / 16;

    Tensor rQ = tqk.partition_fragment_A(sQt);
    Tensor rK = tqk.partition_fragment_B(sKt0);
    Tensor rP = tpv.partition_fragment_A(sPt);
    Tensor dQ = thQ.retile_D(rQ);
    Tensor dK = thK.retile_D(rK);
    Tensor dP = thP.retile_D(rP);

    // ---- fused Q prologue: load BF16, (norm), RoPE, quantize to E4M3 ----
    // Granule loop mirrors the sparse FP8 prefill kernel's Q staging exactly
    // (i-strided: granule i = (row = i/32, g = i%32), one 16-dim granule per
    // iteration).  The optional RMS norm runs as a first pass over the same
    // loop (gmem re-read; one-time per CTA).
    {
        const __ppu_bfloat16_raw *gQ = reinterpret_cast<const __ppu_bfloat16_raw *>(gQRaw)
            + (long)q_token * q_stride;
        const int position = __ldg(gTokenPositions + q_token);
        const float *rope = gCosSinCache + (long)position * 64;
        if constexpr (EnableNorm) {
            for (int i = tid; i < BM * 32; i += NTHR) {
                const int r = i / 32, g = i % 32;
                const uint4 *src = reinterpret_cast<const uint4 *>(gQ + (long)r * 512 + g * 16);
                const uint4 lo = src[0], hi = src[1];
                const auto *e = reinterpret_cast<const cutlass::bfloat16_t *>(&lo);
                const auto *e2 = reinterpret_cast<const cutlass::bfloat16_t *>(&hi);
                float sumsq = 0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 8; ++j) {
                    sumsq = fmaf(float(e[j]), float(e[j]), sumsq);
                    sumsq = fmaf(float(e2[j]), float(e2[j]), sumsq);
                }
                sNormPartial[i] = sumsq;  // 32 granule partials per row
            }
            __syncthreads();
            if (tid < BM) {
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 32; ++j) s += sNormPartial[tid * 32 + j];
                sNorm[tid] = rsqrtf(s / 512.f + rms_norm_eps);
            }
            __syncthreads();
        }
        // Segment walk mirrors the KV gather's addressing exactly: eight lanes
        // cover a whole 128-dim segment of one row, and every 16-dim granule is
        // written through the swizzled layout at (row, col).
        constexpr int Q_LANES = 8;
        constexpr int Q_SEGW = Q_LANES * 16;      // 128 dims per segment
        constexpr int Q_SEGS = 512 / Q_SEGW;      // four segments per row
        const int qrow = tid / Q_LANES, qlane = tid % Q_LANES, qcol0 = qlane * 16;
        for (int s = 0; s < Q_SEGS; ++s) {
            const int col = qcol0 + s * Q_SEGW;
            const uint4 *src = reinterpret_cast<const uint4 *>(gQ + (long)qrow * 512 + col);
            const uint4 lo = src[0], hi = src[1];
            const auto *e = reinterpret_cast<const cutlass::bfloat16_t *>(&lo);
            const auto *e2 = reinterpret_cast<const cutlass::bfloat16_t *>(&hi);
            float x[16];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j) {
                x[2 * j] = float(e[j]);
                x[2 * j + 1] = float(e2[j]);
            }
            const float qnorm = EnableNorm ? sNorm[qrow] : 1.f;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 16; ++j) x[j] *= qnorm;
            if (col >= 448) {  // rope dims [448, 512): rotate before quantizing
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 16; j += 2) {
                    const int pair = (col + j - 448) / 2;
                    const float2 rotated = fused_apply_rope(float2{x[j], x[j + 1]},
                        __ldg(rope + pair), __ldg(rope + 32 + pair));
                    x[j] = rotated.x;
                    x[j + 1] = rotated.y;
                }
            }
            // Scalar E4M3 conversion + 8-code (8-byte) stores, the same
            // granularity the decode five-in-one's FP8 loader uses.
            CUTLASS_PRAGMA_UNROLL
            for (int h = 0; h < 2; ++h) {
                alignas(16) cutlass::float_e4m3_t codes[8];
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 8; ++j)
#if FUSED_FP8_Q_NOQUANT
                    // Diagnostic: truncate the scaled float to a byte so the
                    // staging address is exercised without the converter.
                    codes[j] = cutlass::float_e4m3_t::bitcast(
                        uint8_t(int(x[h * 8 + j] * 16.0f) & 0xff));
#else
                    codes[j] = cutlass::float_e4m3_t::from_float(x[h * 8 + j]);
#endif
                *reinterpret_cast<uint2 *>(&sQt(qrow, col + h * 8)) =
                    *reinterpret_cast<const uint2 *>(codes);
            }
        }
    }
    if (tid < BM) {
        sRunMax[tid] = NEG_INF;
        sRunSum[tid] = 0.f;
    }
    const int *gIdxRow = gIdx + (long)q_token * idx_stride;
    for (int i = tid; i < active_tiles * BN; i += NTHR)
        sIdx[i] = i < topk_length ? gIdxRow[i] : -1;
    __syncthreads();

    // ---- KV gather (E4M3 straight from the quantized workspace) ----
    static constexpr int GATHER_LANES = 8;
    static constexpr int GATHER_WIDTH = GATHER_LANES * 16;
    static constexpr int KSEGS = (512 + GATHER_WIDTH - 1) / GATHER_WIDTH;
    const int grow = tid / GATHER_LANES, glane = tid % GATHER_LANES, col0 = glane * 16;
    int kdst_off[KSEGS];
    {
        Tensor sK0 = make_tensor(make_smem_ptr(pK), LayoutK8{});
        const uint8_t *kbase0 = reinterpret_cast<const uint8_t *>(pK);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < KSEGS; ++j) {
            const int col = col0 + j * GATHER_WIDTH;
            kdst_off[j] = (int)(reinterpret_cast<const uint8_t *>(&sK0(grow, col < 512 ? col : 0)) - kbase0);
        }
    }
    auto issue_K = [&](int t) {
        uint8_t *kdst = reinterpret_cast<uint8_t *>(pK + (t & 1) * KTILE);
        const int tok = sIdx[t * BN + grow];
        const bool ok = (unsigned)tok < (unsigned)s_kv;
        const uint8_t *src = reinterpret_cast<const uint8_t *>(gKV) + (long)(ok ? tok : 0) * 512 + col0;
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < KSEGS; ++j)
            cute::PPU_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>::copy(
                *reinterpret_cast<const cute::uint128_t *>(src + j * GATHER_WIDTH),
                *reinterpret_cast<cute::uint128_t *>(kdst + kdst_off[j]), ok);
    };
    const int vrow = (lane & 3) * 4 + ((lane >> 2) & 3);
    const int vcol = wp * 16 + (lane >> 4) * 4;
    const int voff = (int)(reinterpret_cast<const uint8_t *>(&sKt0(vrow, vcol)) -
                           reinterpret_cast<const uint8_t *>(pK));
    flashmla_fp8_std::prime_kv_pipeline(issue_K);

    // ---- main loop: E4M3 QK, online softmax, E4M3 P, E4M3 PV ----
    for (int t = 0; t < active_tiles; ++t) {
        flashmla_fp8_std::prefetch_next_kv(issue_K, t, active_tiles);

        Tensor acc_s = tqk.partition_fragment_C(sPt);
        Tensor acc_s1 = tqk.partition_fragment_C(sPt);
        clear(acc_s);
        clear(acc_s1);
        if (tid < QK_THREADS) {
            // Stream the Q/K fragments through the E4M3 copy atoms instead of
            // the raw PPU_LDM asm: the atoms carry the swizzle mode calibrated
            // for 1-byte operands, while a hard-coded mode aliases granules.
            Tensor rQv = tqk.partition_fragment_A(sQt);
            Tensor rKv = tqk.partition_fragment_B(sKt0);
            Tensor cQ = thQ.retile_D(rQv);
            Tensor cK = thK.retile_D(rKv);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < NK_QK; k += 2) {
                cute::copy(tcQ, srcQ(_, _, k), cQ(_, _, _0{}));
                cute::copy(tcK, srcK_(_, _, k), cK(_, _, _0{}));
                cute::gemm(mqk, cQ(_, _, _0{}), cK(_, _, _0{}), acc_s);
                if (k + 1 < NK_QK) {
                    cute::copy(tcQ, srcQ(_, _, k + 1), cQ(_, _, _0{}));
                    cute::copy(tcK, srcK_(_, _, k + 1), cK(_, _, _0{}));
                    cute::gemm(mqk, cQ(_, _, _0{}), cK(_, _, _0{}), acc_s1);
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int e = 0; e < 8; ++e) {
                const int col = wn * 16 + (lane % 4) * 2 + ((e >> 2) & 1) * 8 + (e & 1);
                const bool valid = (unsigned)sIdx[t * BN + col] < (unsigned)s_kv;
                // Per-row RMS norm is folded into the score (exact: the norm
                // is a positive per-row scalar, so (n*q).k == n*(q.k)).
                const float n = EnableNorm ? sNorm[r0 + Traits::score_row_offset(e)] : 1.f;
                acc_s(e) = valid ? (acc_s(e) + acc_s1(e)) * n : -INFINITY;
            }
            if (tid == 0)
                *sFlag = 0;
            flashmla_fp8_std::reduce_score_max<Traits, BM, 8>(acc_s, sPmax, lane, wn, r0, r1);
        }
        __syncthreads();
        if (tid < BM) {
            // Exact running max (no ceil quantization): the fused contract's
            // max_logits tolerance is 1e-5 against the true scaled maximum.
            float m = fmaxf(sPmax[tid], sPmax[BM + tid]);
            m = fmaxf(m, fmaxf(sPmax[2 * BM + tid], sPmax[3 * BM + tid]));
            m *= sm_scale_log2;
            float mprev = sRunMax[tid];
            float mnew = fmaxf(mprev, m);
            sRunMax[tid] = mnew;
            float sc = (mprev == NEG_INF) ? 1.f : exp2f(mprev - mnew);
            sScale[tid] = sc;
            if (sc != 1.f)
                *sFlag = 1;
        }
        __syncthreads();

        if (tid < QK_THREADS) {
            const float mx0 = sRunMax[r0], mx1 = sRunMax[r1];
            flashmla_fp8_std::quantize_probability<Traits, BM, 8>(
                acc_s, tqk, sPt, sPsum, lane, wn, r0, r1, mx0, mx1, sm_scale_log2);
        }
        if (*sFlag) {
            flashmla_fp8_std::rescale_output<Traits, BM>(acc_o, sScale, mi);
        }
        __syncthreads();
        if (tid < BM)
            flashmla_fp8_std::update_running_sum<Traits, BM, false>(sPsum, sRunSum, sScale, tid);

        flashmla_fp8_std::accumulate_pv<Traits, BM, BN>(
            tpv, rP, mpv, acc_o, pK, ptsm0, voff, (t & 1) * KTILE);

        flashmla_fp8_std::wait_next_kv(t, active_tiles);
    }
    __syncthreads();

    // ---- per-row epilogue scalars (natural-log domain, sink-aware) ----
    if (tid < BM) {
        const float m = sRunMax[tid];
        const float s = sRunSum[tid];
        const bool empty = !(s > 0.f);
        if (gMaxLogits)
            gMaxLogits[(long)q_token * h_q + tid] = empty ? -INFINITY : m * 0.6931471805599453f;
        if (gLse)
            gLse[(long)q_token * h_q + tid] =
                empty ? INFINITY : (m + log2f(s * (1.f / P_SCALE))) * 0.6931471805599453f;
        float den = s;
        if (gAttnSink) {
            const float sink_log2 = gAttnSink[tid] * 1.4426950408889634f;
            den += exp2f(sink_log2 - m + LOG2_P_SCALE);
        }
        // 1/den collapses to 0 for the empty row and for a +inf sink, which
        // are both "all-zero output" per the fused contract.
        sInvDen[tid] = (den == den && den > 0.f) ? 1.f / den : 0.f;
    }
    __syncthreads();

    // ---- fused O epilogue: normalize, O-RoPE, per-group amax, E4M3 + sf ----
    Tensor idO = make_identity_tensor(Shape<Int<BM>, Int<DV>>{});
    Tensor cOid = tpv.partition_C(idO);
    const int position = __ldg(gTokenPositions + q_token);
    const float *rope = gCosSinCache + (long)position * 64;
    float amax8[BM / 8];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < BM / 8; ++j) amax8[j] = 1e-4f;
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < size(acc_o); e += 2) {
        const int row = get<0>(cOid(e)), col = get<1>(cOid(e));
        float v0 = acc_o(e) * sInvDen[row], v1 = acc_o(e + 1) * sInvDen[row];
        if (col >= 448) {
            const int pair = (col - 448) >> 1;
            const float2 rotated = fused_apply_rope(float2{v0, v1},
                __ldg(rope + pair), -__ldg(rope + 32 + pair));
            v0 = rotated.x;
            v1 = rotated.y;
        }
        acc_o(e) = v0;
        acc_o(e + 1) = v1;
        const float mx = fmaxf(fabsf(v0), fabsf(v1));
        amax8[row >> 3] = fmaxf(amax8[row >> 3], mx);
    }
    // Reduce the per-group amax across the CTA (warp xor, then 16 warps).
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            amax8[j] = fmaxf(amax8[j], __shfl_xor_sync(0xffffffff, amax8[j], offset));
    }
    if (lane == 0)
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < BM / 8; ++j) sGroupAmax[wp * (BM / 8) + j] = amax8[j];
    __syncthreads();
    if (tid < BM / 8) {
        float amax = 1e-4f;
#pragma unroll
        for (int w = 0; w < 16; ++w) amax = fmaxf(amax, sGroupAmax[w * (BM / 8) + tid]);
        const float scale = amax / 448.f;
        sGroupScale[tid] = scale;
        gOutSf[(long)q_token * n_wv_group + tid] = scale;
    }
    __syncthreads();
    // Pack: out[token, group, tile*256 + in_group*32 + i] holds
    // O[head=group*8+in_group][col=tile*32+i] (fused WV layout).
    CUTLASS_PRAGMA_UNROLL
    for (int e = 0; e < size(acc_o); e += 2) {
        const int row = get<0>(cOid(e)), col = get<1>(cOid(e));
        const float mult = 1.f / sGroupScale[row >> 3];
        const uint16_t packed = flashmla::sparse_native_fp8::pack_e4m3_pair(
            float2{acc_o(e) * mult, acc_o(e + 1) * mult});
        const int64_t off = ((int64_t)q_token * n_wv_group + (row >> 3)) * 4096
            + (col >> 5) * 256 + (row & 7) * 32 + (col & 31);
        *reinterpret_cast<uint16_t *>(gOutFp8 + off) = packed;
    }
}

// Host launcher.  Q is [s_q, 64, 512] bf16; kv is the E4M3 workspace
// (the caller quantizes it once); indices are [s_q, topk] int32.
template <int BM, bool EnableNorm, int MaxTopk>
void launch_fused_prefill_fp8(
    const void *q, const void *kv, const int *idx, const int *lengths, const float *sink,
    const int *token_positions, const float *cos_sin_cache, uint8_t *out_fp8, float *out_sf,
    float *max_logits, float *lse, float sm_scale_log2, float rms_norm_eps, int topk, int s_kv,
    int s_q, int n_wv_group, int64_t q_stride, int64_t idx_stride, hggcStream_t stream) {
    constexpr int NTHR = FusedFp8PrefillTraits<BM, MaxTopk>::Threads;
    constexpr size_t smem = fused_fp8_prefill_smem<BM, MaxTopk, EnableNorm>();
    static_assert(smem <= 256 * 1024);
    auto kernel = fused_norm_rope_attn_rope_cast_prefill_fp8_kernel<BM, EnableNorm, MaxTopk>;
    C10_CUDA_CHECK(hggcFuncSetAttribute(kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<dim3(s_q, 1), NTHR, smem, stream>>>(
        q, reinterpret_cast<const E4M3 *>(kv), idx, lengths, sink, token_positions, cos_sin_cache,
        out_fp8, out_sf, max_logits, lse, sm_scale_log2, rms_norm_eps, topk, s_kv, n_wv_group,
        q_stride, idx_stride);
    CHECK_CUDA_KERNEL_LAUNCH();
}

} // namespace flashmla::fused_norm_rope_attn_rope_cast_fwd::fp8_prefill
