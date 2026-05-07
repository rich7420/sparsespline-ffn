// Backward v7 — CuTe-based TMA + WGMMA bwd kernel.
//
// This is the production successor to v5/v6.1a, using the CuTe layout pipeline
// validated in cute_oracle.cu Phase 1c (bit-exact parity vs torch.matmul):
//
//   - W operand : K-major SMEM, Layout_K_SW128_Atom<half>
//                 (M = BH*L_PAD, K = BLOCK_N; K-contig matches the manual write)
//   - g operand : MN-major SMEM, Layout_MN_SW64_Atom<half>
//                 (N = R, K = BLOCK_N; N-contig matches g_delta's row-major)
//   - WGMMA atom: SM90_64x32x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>
//
// Single-stage TMA path (no warp-spec, no multi-stage pipeline). Parity is
// the gate; speed optimizations are layered on after this lands.
//
// Math contract (BIT-EQUAL to v5 modulo precision floor):
//   dC[j, b, c] = Σ_n W[n, m=j_local*L_PAD+b] · g[n, c]
//   dz[n, j]    = scale · Σ_c g[n,c] · Σ_k dB_k(τ_nj) · C[j, bin+k, c]
//
// Note on requirements:
//   - sm_90a only.
//   - Requires CUTLASS PR #2171 patch on /opt/cutlass (cast_smem_ptr_to_uint
//     promoted to CUTE_HOST_DEVICE). The Modal launcher applies this perl
//     patch in the image build step.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstring>

// CuTe / CUTLASS headers (header-only, requires patched include path).
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

namespace v7_cute_impl {

using namespace cute;

// =============================================================================
// mbarrier + fence helpers (raw asm; CuTe wraps these but the wrappers vary
// by CUTLASS version, cleaner to keep our own).
// =============================================================================

__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t arrival) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(addr), "r"(arrival));
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* mbar, uint32_t bytes) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n\t"
        ".reg .pred P;\n\t"
        "WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@P bra DONE_%=;\n\t"
        "bra WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}\n" :: "r"(addr), "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// 2D TMA load using a CUtensorMap (for C — kept as v6.1a-style raw asm
// because we already know that path works).
__device__ __forceinline__ void cp_async_bulk_tensor_2d_g2s(
    void* smem_ptr, const void* tensor_map,
    int32_t coord0, int32_t coord1, uint64_t* mbar
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n"
        :: "r"(smem_addr), "l"(tensor_map), "r"(mbar_addr),
           "r"(coord0), "r"(coord1)
        : "memory"
    );
}

// =============================================================================
// fp helpers + B-spline math (same as v6)
// =============================================================================

__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float h2f(__half x) { return __half2float(x); }
__device__ __forceinline__ __half f2h(float x) { return __float2half(x); }

__device__ __forceinline__ void compute_B2(float tau, float& B0, float& B1, float& B2) {
    float omt = 1.0f - tau;
    B0 = 0.5f * omt * omt;
    B1 = 0.5f * (1.0f + 2.0f * tau - 2.0f * tau * tau);
    B2 = 0.5f * tau * tau;
}
__device__ __forceinline__ void compute_dB2(float tau, float& dB0, float& dB1, float& dB2) {
    dB0 = -(1.0f - tau);
    dB1 = 1.0f - 2.0f * tau;
    dB2 = tau;
}

// =============================================================================
// v7 backward kernel — CuTe-based.
//
// Tile contract (matches v5/v6 production):
//   BLOCK_N = K dim (128)
//   BLOCK_H * L_PAD = M dim (e.g., 8 * 24 = 192)
//   R       = N dim (32 or 64)
// =============================================================================

template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int N_PARTS,
          class TmaG>
__global__ void __launch_bounds__(128, 2)
spline_kv_bwd_v7_cute_kernel(
    const __nv_bfloat16* __restrict__ z,
    const __grid_constant__ CUtensorMap C_tma_map,
    CUTE_GRID_CONSTANT TmaG const tma_g,
    float* __restrict__ dC_scratch,
    float* __restrict__ dz,
    const int N, const int H, const int L,
    const float grid_lo, const float scale,
    const int chunks_per_block
) {
    constexpr int M = BLOCK_H * L_PAD;

    // ---- CuTe atoms + layouts (validated by Phase 1c microtest) ----
    // Single-stage TMA pipeline. A multi-stage attempt (Phase 5a-1) hit a
    // deadlock because we tried to reuse mbarrier stages without the
    // canonical CUTLASS full+empty acquire-release protocol; see
    // docs/plan_v7_postdeadline.md for the correct PipelineTmaAsync recipe
    // when this is picked up post-deadline.
    using AtomA = decltype(GMMA::Layout_K_SW128_Atom<half_t>{});
    using AtomB = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayA  = decltype(tile_to_shape(AtomA{},
                                         Shape<Int<M>, Int<BLOCK_N>>{}));
    using LayB  = decltype(tile_to_shape(AtomB{},
                                         Shape<Int<R>, Int<BLOCK_N>>{}));

    constexpr int sA_bytes = sizeof(half_t) * cosize_v<LayA>;
    constexpr int sB_bytes = sizeof(half_t) * cosize_v<LayB>;
    static_assert(sA_bytes % 1024 == 0, "sA byte size must be 1024B-aligned");

    extern __shared__ __align__(1024) char smem_buf[];
    half_t* sW_raw = reinterpret_cast<half_t*>(smem_buf);
    half_t* sg_raw = reinterpret_cast<half_t*>(smem_buf + sA_bytes);
    Tensor sW = make_tensor(make_smem_ptr(sW_raw), LayA{});
    Tensor sg = make_tensor(make_smem_ptr(sg_raw), LayB{});

    // C_smem unchanged from v6.1a — used only by Phase 3b dz inner, not WGMMA.
    __shared__ __align__(128) __half C_smem[BLOCK_H][L_PAD][R];

    __shared__ __align__(8) uint64_t mbar_g;
    __shared__ __align__(8) uint64_t mbar_C;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int pid_part = blockIdx.x;
    const int pid_h    = blockIdx.y;
    const int h_start  = pid_h * BLOCK_H;
    const int n_per_part = N / N_PARTS;
    const int n_part_start = pid_part * n_per_part;

    // ---- TiledMma (validated SM90_64x32x16_F32F16F16_SS<K, MN>) ----
    using TiledMma = decltype(make_tiled_mma(
        SM90_64x32x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>{}));
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    Tensor tCsA = thr_mma.partition_A(sW);
    Tensor tCsB = thr_mma.partition_B(sg);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);
    Tensor tCrC = partition_fragment_C(tiled_mma,
                                        Shape<Int<M>, Int<R>>{});
    clear(tCrC);

    // ---- Phase A: TMA-load C[h_start..h_start+BH] into C_smem ----
    if (tid == 0) {
        mbar_init(&mbar_C, BLOCK_H);
        fence_proxy_async_shared();
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t bytes_per_h = (uint32_t)L * (uint32_t)R * sizeof(__half);
        #pragma unroll
        for (int j_local = 0; j_local < BLOCK_H; j_local++) {
            mbar_arrive_expect(&mbar_C, bytes_per_h);
            cp_async_bulk_tensor_2d_g2s(
                /*smem_ptr=*/   &C_smem[j_local][0][0],
                /*tensor_map=*/ &C_tma_map,
                /*coord0=*/     0,
                /*coord1=*/     (h_start + j_local) * L,
                /*mbar=*/       &mbar_C);
        }
    }
    mbar_wait(&mbar_C, /*phase=*/ 0);
    __syncthreads();

    // ---- Initialize g mbarrier (will be flipped each chunk) ----
    if (tid == 0) {
        mbar_init(&mbar_g, 1);
        fence_proxy_async_shared();
    }
    __syncthreads();

    auto thr_tma_g = tma_g.get_slice(Int<0>{});

    Tensor gG_full   = tma_g.get_tma_tensor(make_shape(Int<R>{}, N));
    Tensor gG_tiled  = local_tile(gG_full,
                                   Shape<Int<R>, Int<BLOCK_N>>{},
                                   make_coord(_0{}, _));

    // ---- Main chunk loop (single-stage: load → compute → next) ----
    #pragma unroll 1
    for (int chunk = 0; chunk < chunks_per_block; chunk++) {
        const int n_start = n_part_start + chunk * BLOCK_N;
        const uint32_t g_phase = chunk & 1;

        // --- Phase 1: zero sW ---
        {
            constexpr int total_w_bytes = sizeof(half_t) * cosize_v<LayA>;
            constexpr int total_w_uint4 = total_w_bytes / sizeof(uint4);
            uint4* sW_u4 = reinterpret_cast<uint4*>(sW_raw);
            #pragma unroll
            for (int idx = tid; idx < total_w_uint4; idx += blockDim.x) {
                sW_u4[idx] = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // --- Issue TMA for this chunk's g ---
        const int global_chunk_idx = (n_part_start / BLOCK_N) + chunk;
        if (tid == 0) {
            constexpr uint32_t bytes_g = BLOCK_N * R * sizeof(half_t);
            mbar_arrive_expect(&mbar_g, bytes_g);
            Tensor gG_this = gG_tiled(_, _, global_chunk_idx);
            copy(tma_g.with(reinterpret_cast<uint64_t&>(mbar_g), 0),
                 thr_tma_g.partition_S(gG_this),
                 thr_tma_g.partition_D(sg));
        }
        __syncthreads();
        mbar_wait(&mbar_g, g_phase);
        __syncthreads();

        // --- Phase 3a: fill sW from z (B-spline values) ---
        // Each (n_local, j_local) writes 3 spline weights at columns
        // m_base+0/+1/+2 where m_base = j_local*L_PAD + bin_idx.
        // sW(m, n_local) — CuTe handles SW128 swizzle internally.
        {
            constexpr int total_pairs = BLOCK_N * BLOCK_H;
            for (int p = tid; p < total_pairs; p += blockDim.x) {
                const int n_local  = p / BLOCK_H;
                const int j_local  = p % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global >= N || j_global >= H) continue;

                const float z_val = bf2f(z[n_global * H + j_global]);
                const float u = (z_val - grid_lo) * scale;
                const float G_max = (float)(L - 2);
                const bool  in_range = (u >= 0.0f) && (u <= G_max);
                const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
                const int   bin_idx = (int)u_clip;
                const float tau = u_clip - (float)bin_idx;

                float B0, B1, B2;
                compute_B2(tau, B0, B1, B2);
                if (!in_range) { B0 = 0.0f; B1 = 0.0f; B2 = 0.0f; }

                const int col_base = j_local * L_PAD + bin_idx;
                // sW logical shape is (M, K=BLOCK_N). We're writing W[m, n_local].
                // CuTe Tensor of cutlass::half_t — assign via static_cast so
                // the float→half_t conversion goes through the right path.
                sW(col_base + 0, n_local) = static_cast<half_t>(B0);
                sW(col_base + 1, n_local) = static_cast<half_t>(B1);
                sW(col_base + 2, n_local) = static_cast<half_t>(B2);
            }
        }
        __syncthreads();
        fence_proxy_async_shared();

        // --- Phase 4: WGMMA (cute::gemm) ---
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
        warpgroup_commit_batch();

        // --- Phase 3b: dz inner (parallel with Phase 4 wgmma) ---
        {
            constexpr int total_pairs = BLOCK_N * BLOCK_H;
            for (int p = tid; p < total_pairs; p += blockDim.x) {
                const int n_local  = p / BLOCK_H;
                const int j_local  = p % BLOCK_H;
                const int n_global = n_start + n_local;
                const int j_global = h_start + j_local;
                if (n_global >= N || j_global >= H) continue;

                const float z_val = bf2f(z[n_global * H + j_global]);
                const float u = (z_val - grid_lo) * scale;
                const float G_max = (float)(L - 2);
                const bool clamp_active = (u >= 0.0f) && (u <= G_max - 1.0f);
                const float u_clip = fminf(fmaxf(u, 0.0f), G_max - 1.0f);
                const int   bin_idx = (int)u_clip;
                const float tau = u_clip - (float)bin_idx;

                float dB0, dB1, dB2;
                compute_dB2(tau, dB0, dB1, dB2);
                if (!clamp_active) { dB0 = 0.0f; dB1 = 0.0f; dB2 = 0.0f; }

                float inner = 0.0f;
                #pragma unroll
                for (int c = 0; c < R; c++) {
                    const float g  = static_cast<float>(sg(c, n_local));
                    const float c0 = h2f(C_smem[j_local][bin_idx + 0][c]);
                    const float c1 = h2f(C_smem[j_local][bin_idx + 1][c]);
                    const float c2 = h2f(C_smem[j_local][bin_idx + 2][c]);
                    inner += g * (dB0 * c0 + dB1 * c1 + dB2 * c2);
                }
                dz[n_global * H + j_global] = scale * inner;
            }
        }

        // --- Wait for Phase 4 wgmma ---
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
        __syncthreads();
    }  // end chunk loop

    // ---- Phase 5: store accumulated fragment to dC_scratch ----
    // partition_C gives the per-thread output fragment indexed as (M, N).
    // We translate (m, c) → (j_global, b, c) in dC_scratch coordinates.
    Tensor tCgD = thr_mma.partition_C(
        make_tensor(make_gmem_ptr<float>(nullptr),  // dummy — we use coords directly
                    Shape<Int<M>, Int<R>>{},
                    LayoutRight{}));
    // Actually, the cleanest way is to iterate the fragment with cute layout
    // navigation. But for parity-first simplicity, fall back to the v5-style
    // hand-decoded fragment store using lane_id / warp_id indexing — same
    // formula as v5/v6, applied to the cute::gemm output.
    //
    // m64n32 output fragment: 4 chunks × 4 elements per thread per WGMMA tile.
    // Per WGMMA tile (M=64 × N=32):
    //   tCrC layout per cute::gemm of an SM90_64x32x16: 16 fp32 per thread.
    // Multiple tiles in M direction: M_TILES = M / 64 = 3.
    // (N_TILES = R/32 = 1 for r=32; = 2 for r=64.)
    constexpr int M_TILES = M / 64;
    constexpr int N_TILES = R / 32;
    // Convert tCrC into a flat per-thread float pointer for the v5-style decode.
    // tCrC is a Tensor of fp32 with M_TILES * N_TILES * 16 elements per thread.
    float* acc_ptr = reinterpret_cast<float*>(&tCrC(0));

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
        #pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
            const int frag_base = (m_tile * N_TILES + n_tile) * 16;
            #pragma unroll
            for (int chunk_e = 0; chunk_e < 4; chunk_e++) {
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int frag_idx = chunk_e * 4 + e;
                    const int groupID  = lane_id / 4;
                    const int tigid    = lane_id % 4;
                    int row_in_warp, col_in_chunk;
                    switch (e) {
                        case 0: row_in_warp = groupID;     col_in_chunk = tigid*2 + 0; break;
                        case 1: row_in_warp = groupID;     col_in_chunk = tigid*2 + 1; break;
                        case 2: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 0; break;
                        case 3: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 1; break;
                    }
                    const int row_in_tile = warp_id * 16 + row_in_warp;
                    const int col_in_tile = chunk_e * 8 + col_in_chunk;
                    const int m_global    = m_tile * 64 + row_in_tile;
                    const int j_local     = m_global / L_PAD;
                    const int b           = m_global % L_PAD;
                    const int j_global    = h_start + j_local;
                    const int c           = n_tile * 32 + col_in_tile;
                    if (j_global < H && b < L) {
                        const long out_idx =
                            ((long)j_global * N_PARTS + pid_part) * L * R +
                            (long)b * R + c;
                        dC_scratch[out_idx] = acc_ptr[frag_base + frag_idx];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Reduce kernel (unchanged from v6).
// =============================================================================
__global__ void spline_kv_bwd_v7_reduce_kernel(
    const float* __restrict__ dC_scratch,
    __nv_bfloat16* __restrict__ dC_bf16,
    const int H, const int L, const int R, const int N_PARTS
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = H * L * R;
    if (idx >= total) return;
    const int h = idx / (L * R);
    const int rem = idx % (L * R);
    const int b = rem / R;
    const int c = rem % R;
    float sum = 0.0f;
    #pragma unroll 4
    for (int p = 0; p < N_PARTS; p++) {
        sum += dC_scratch[((long)h * N_PARTS + p) * L * R + b * R + c];
    }
    dC_bf16[idx] = __float2bfloat16(sum);
}

}  // namespace v7_cute_impl

// =============================================================================
// Host entry — builds CuTe TMA atoms, encodes C TMA descriptor, dispatches.
// =============================================================================

template <int BN, int BH, int LP, int RR, int NP>
static void launch_v7_cute(
    const __nv_bfloat16* z, const cute::half_t* g_fp16, void* C_ptr,
    float* dC_scratch_ptr, float* dz_ptr,
    int N, int H, int L, float grid_lo, float scale, int chunks_per_block,
    cudaStream_t stream
) {
    using namespace cute;
    using namespace v7_cute_impl;

    // ---- Build TMA atom for g ----
    // g_delta_fp16 is global half tensor [N, R] row-major.
    // CuTe view: (R, N) with R contig (stride 1).
    auto g_global = make_tensor(
        make_gmem_ptr(g_fp16),
        make_shape(Int<RR>{}, N),
        make_stride(Int<1>{}, Int<RR>{}));
    using AtomB = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayB  = decltype(tile_to_shape(AtomB{}, Shape<Int<RR>, Int<BN>>{}));
    auto tma_g = make_tma_copy(SM90_TMA_LOAD{}, g_global, LayB{});

    // ---- Build C TMA descriptor (raw cuTensorMapEncodeTiled, v6.1a-style) ----
    alignas(64) CUtensorMap C_tma_map;
    {
        const cuuint64_t global_dim[2] = {
            (cuuint64_t)RR,
            (cuuint64_t)((cuuint64_t)H * (cuuint64_t)L),
        };
        const cuuint64_t global_strides[1] = {
            (cuuint64_t)RR * sizeof(__half),
        };
        const cuuint32_t box_dim[2] = {
            (cuuint32_t)RR, (cuuint32_t)L,
        };
        const cuuint32_t element_strides[2] = { 1, 1 };
        CUresult err = cuTensorMapEncodeTiled(
            &C_tma_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            /*tensorRank=*/ 2,
            C_ptr,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(err == CUDA_SUCCESS,
                    "cuTensorMapEncodeTiled failed for C: code=", (int)err);
    }

    // ---- Compute dynamic shared memory size ----
    // sW (single-buffered) + NUM_STAGES * sg (3-stage TMA pipeline). NUM_STAGES
    // must match the kernel-side constant.
    constexpr int M = BH * LP;
    constexpr int kNumStages = 3;
    using AtomA = decltype(GMMA::Layout_K_SW128_Atom<half_t>{});
    using LayA  = decltype(tile_to_shape(AtomA{}, Shape<Int<M>, Int<BN>>{}));
    constexpr int sW_bytes = sizeof(half_t) * cosize_v<LayA>;
    constexpr int sg_bytes = sizeof(half_t) * cosize_v<LayB>;
    constexpr int dynamic_smem = sW_bytes + kNumStages * sg_bytes;

    const int blocks_h = (H + BH - 1) / BH;
    dim3 grid(NP, blocks_h, 1);
    dim3 block(128, 1, 1);

    // Increase max dynamic SMEM if needed (Hopper supports up to 228 KB).
    auto kernel = spline_kv_bwd_v7_cute_kernel<BN, BH, LP, RR, NP, decltype(tma_g)>;
    if (dynamic_smem > 48 * 1024) {
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem);
    }

    kernel<<<grid, block, dynamic_smem, stream>>>(
        z, C_tma_map, tma_g,
        dC_scratch_ptr, dz_ptr,
        N, H, L, grid_lo, scale, chunks_per_block);
}

std::vector<torch::Tensor> spline_kv_bwd_v7_cute_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    const torch::Tensor& g_delta,
    double grid_lo, double scale,
    int64_t L_arg
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16, "z must be bf16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 (we cast to fp16)");
    TORCH_CHECK(g_delta.dtype() == torch::kBFloat16, "g must be bf16 (we cast to fp16)");
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous() && g_delta.is_contiguous());

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = (int)L_arg;
    const int R = C.size(2);

    auto bf16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat32);

    torch::Tensor C_fp16 = C.to(torch::kFloat16);
    torch::Tensor g_fp16 = g_delta.to(torch::kFloat16);

    constexpr int N_PARTS = 4;
    torch::Tensor dC_bf16 = torch::zeros({H, L, R}, bf16_opts);
    torch::Tensor dz      = torch::zeros({N, H},   fp32_opts);
    torch::Tensor dC_scratch = torch::empty({H, N_PARTS, L, R}, fp32_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();
    using cute::half_t;
    const __nv_bfloat16* z_ptr = (const __nv_bfloat16*)z.data_ptr();
    const half_t*       g_ptr = (const half_t*)g_fp16.data_ptr();
    void*               C_ptr_v = C_fp16.data_ptr();
    float*              dC_scratch_ptr = dC_scratch.data_ptr<float>();
    float*              dz_ptr         = dz.data_ptr<float>();

    // Single dispatch for the production R=32 / L=22 / BLOCK_N=128 / BH=8 cell.
    if (R == 32 && L == 22) {
        const int chunks_per_block = (N / N_PARTS) / 128;
        TORCH_CHECK(chunks_per_block > 0,    "v7: needs N/NPARTS/BN > 0");
        TORCH_CHECK((N / N_PARTS) % 128 == 0, "v7: needs N/NPARTS divisible by 128");
        launch_v7_cute<128, 8, 24, 32, N_PARTS>(
            z_ptr, g_ptr, C_ptr_v,
            dC_scratch_ptr, dz_ptr,
            N, H, L, (float)grid_lo, (float)scale, chunks_per_block,
            stream);
    } else {
        TORCH_CHECK(false,
            "v7 cute: only (R=32, L=22) supported in this build. "
            "Got (R=", R, ", L=", L, ").");
    }

    // Reduce dC_scratch -> dC_bf16.
    const int total_dC = H * L * R;
    const int reduce_grid = (total_dC + 255) / 256;
    v7_cute_impl::spline_kv_bwd_v7_reduce_kernel<<<reduce_grid, 256, 0, stream>>>(
        dC_scratch.data_ptr<float>(),
        (__nv_bfloat16*)dC_bf16.data_ptr(),
        H, L, R, N_PARTS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {dC_bf16, dz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_v7_cute_cuda", &spline_kv_bwd_v7_cute_cuda,
          "v7 bwd: CuTe TMA + WGMMA. Returns (dC bf16, dz fp32).",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("grid_lo"), py::arg("scale"), py::arg("L"));
}
