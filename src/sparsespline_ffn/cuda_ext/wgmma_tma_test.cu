// Standalone TMA → WGMMA test (v6.1b training wheel).
//
// Purpose: validate that WGMMA m64n32k16 correctly reads SMEM that was
// written by TMA, IN ISOLATION from the bwd kernel. This pins down the
// silent-error-class question of WGMMA descriptor encoding (LBO/SBO/
// swizzle bits) for TMA-produced SMEM layouts BEFORE we touch g_cores
// in the production bwd kernel (v6.2).
//
// Test shape: A [M=64, K=16] half × B [K=16, N=32] half = D [M=64, N=32] fp32
// (= one wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 tile)
//
// Five descriptor variants exposed via Python:
//   variant=0 : no swizzle, naive row-major (LBO=K*2, SBO=2) — KNOWN WRONG
//   variant=1 : no swizzle, "core-major" v5 encoding (SBO=128) — KNOWN WRONG
//                (v5's manual layout differs from TMA row-major output)
//   variant=2 : 128B swizzle (SBO=1024, swizzle=3) — known to crash; m64n32k16
//                's N=32 may be too narrow for 128B sectors (need N≥64)
//   variant=3 : no swizzle, correctly-derived row-major canonical layout
//                LBO_B = 8 × (N × sizeof(half)) = 512 bytes
//                SBO_B = 8 × sizeof(half) = 16 bytes
//                LBO_A = 8 × (K × sizeof(half)) = 256 bytes
//                SBO_A = 8 × sizeof(half) = 16 bytes
//                Hypothesis: TMA writes row-major; wgmma's "core matrix" is
//                8×8 fp16 (=128 bytes); LBO is byte stride to next 8-row
//                K-tile; SBO is byte stride to next 8-col N-tile in same K-tile.
//   variant=4 : same as variant 3, but with column-major LBO/SBO (in case I
//                inverted M/N vs K conventions)
//                LBO_B = 8 × sizeof(half) = 16 bytes
//                SBO_B = 8 × (N × sizeof(half)) = 512 bytes
//
// Reference: torch.matmul(A.float(), B.float()) → host-side compare.
// Pass: max_abs_err < 1e-2 (fp16 GEMM tolerance).
//
// References for descriptor encoding:
//   PTX ISA §9.7.13 (wgmma matrix descriptor)
//   CUTLASS Discussion #2223 (128B swizzle: SBO = 128*8 = 1024 bytes)
//   Colfax CUTLASS WGMMA tutorial

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstring>

namespace {

// =============================================================================
// PTX helpers (duplicated from v6.cu to keep this file self-contained)
// =============================================================================

__device__ __forceinline__ uint64_t encode_smem_desc(
    void* smem_ptr, uint32_t leading_byte_offset,
    uint32_t stride_byte_offset, uint32_t swizzle = 0
) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    desc |= ((uint64_t)(smem_addr & 0x3FFFF) >> 4) << 0;
    desc |= ((uint64_t)(leading_byte_offset & 0x3FFFF) >> 4) << 16;
    desc |= ((uint64_t)(stride_byte_offset & 0x3FFFF) >> 4) << 32;
    desc |= ((uint64_t)(swizzle & 0x3)) << 62;
    return desc;
}

__device__ __forceinline__ void wgmma_fence()        { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit_group() { asm volatile("wgmma.commit_group.sync.aligned;\n"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group()   { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N)); }

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// f32.f16.f16, trans_a=1, trans_b=1 — matches v5's manual core-major SMEM.
__device__ __forceinline__ void wgmma_m64n32k16_f16(
    float* acc, uint64_t a_desc, uint64_t b_desc, bool scale_d
) {
    int sd = scale_d ? 1 : 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %18, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, p, 1, 1, 1, 1;\n\t"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15])
        : "l"(a_desc), "l"(b_desc), "r"(sd)
    );
}

// f32.f16.f16, trans_a=0, trans_b=0 — for natural row-major SMEM layouts
// (i.e., A is K-contiguous, B is N-contiguous), which is what TMA produces
// from a row-major global tensor.
__device__ __forceinline__ void wgmma_m64n32k16_f16_rm(
    float* acc, uint64_t a_desc, uint64_t b_desc, bool scale_d
) {
    int sd = scale_d ? 1 : 0;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %18, 0;\n\t"
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15}, "
        "%16, %17, p, 1, 1, 0, 0;\n\t"
        "}\n"
        : "+f"(acc[0]),  "+f"(acc[1]),  "+f"(acc[2]),  "+f"(acc[3]),
          "+f"(acc[4]),  "+f"(acc[5]),  "+f"(acc[6]),  "+f"(acc[7]),
          "+f"(acc[8]),  "+f"(acc[9]),  "+f"(acc[10]), "+f"(acc[11]),
          "+f"(acc[12]), "+f"(acc[13]), "+f"(acc[14]), "+f"(acc[15])
        : "l"(a_desc), "l"(b_desc), "r"(sd)
    );
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t arrival_count) {
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(mbar_addr), "r"(arrival_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 :: "r"(mbar_addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n\t"
        ".reg .pred                P1;\n\t"
        "LAB_WAIT_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1                       bra DONE_%=;\n\t"
        "bra                       LAB_WAIT_%=;\n\t"
        "DONE_%=:\n\t"
        "}\n" :: "r"(mbar_addr), "r"(phase)
    );
}

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
// Test kernel — m64n32k16 wgmma with TMA-loaded A and B
// =============================================================================

template <int VARIANT>
__global__ void __launch_bounds__(128, 1)
wgmma_tma_test_kernel(
    const __grid_constant__ CUtensorMap A_tma_map,
    const __grid_constant__ CUtensorMap B_tma_map,
    const __half* __restrict__ A_global,        // for variant 5 manual core-major
    float* __restrict__ D,                      // [M=64, N=32] fp32
    __half* __restrict__ B_smem_dump            // [K * N] fp16, debug dump of B SMEM
                                                //   contents post-TMA (variant 5 only)
) {
    constexpr int M = 64;
    constexpr int K = 16;
    constexpr int N = 32;

    // SMEM: row-major A [M, K] half, row-major B [K, N] half.
    // Variant 5 uses 1024-byte alignment so B_smem starts on a SW64 swizzle
    // tile boundary (matrix_base_offset = 0). Other variants use 128B which
    // is sufficient for their non-swizzled or SW128 paths.
    __shared__ __align__(1024) __half A_smem[M * K];   // 64 × 16 = 1024 half = 2 KB
    __shared__ __align__(1024) __half B_smem[K * N];   // 16 × 32 = 512 half = 1 KB
    __shared__ uint64_t mbar;

    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;

    if constexpr (VARIANT == 5 || VARIANT == 6) {
        // ---- variants 5/6: SW64-matched B, manual core-major A ----
        //   variant 5 = TMA + SMEM-dump ONLY (no WGMMA — pure layout probe)
        //   variant 6 = full path: TMA + SMEM-dump + WGMMA with SW64 descriptor
        //
        // A: copied from global by all threads into v5's core-major SMEM
        //    layout, the same layout that the production v5/v6.1a bwd kernel
        //    produces. This isolates the TMA→SW64→WGMMA path to the B
        //    operand only — which is what production cares about (g_cores
        //    is the operand we want to TMA-load; W_cores stays manual).
        // B: TMA loads under CU_TENSOR_MAP_SWIZZLE_64B into 1024-byte-aligned
        //    SMEM, then (variant 6 only) WGMMA descriptor swizzle bits = 2.
        constexpr int M_CORES = M / 8;   // 8
        const int total_A = M * K;
        for (int idx = tid; idx < total_A; idx += blockDim.x) {
            const int m = idx / K;
            const int k = idx % K;
            // Core-major: ((k>>3)*M_CORES + (m>>3))*64 + (k&7)*8 + (m&7)
            const int dst = ((k >> 3) * M_CORES + (m >> 3)) * 64
                          + (k & 7) * 8 + (m & 7);
            A_smem[dst] = A_global[m * K + k];
        }
        __syncthreads();

        if (tid == 0) {
            mbarrier_init(&mbar, 1);
            fence_proxy_async_shared_cta();
            const uint32_t bytes_B = K * N * sizeof(__half);
            mbarrier_arrive_expect_tx(&mbar, bytes_B);
            cp_async_bulk_tensor_2d_g2s(&B_smem[0], &B_tma_map, 0, 0, &mbar);
        }
        __syncthreads();
        mbarrier_wait(&mbar, 0);
        __syncthreads();

        // ---- Dump B_smem raw bytes for offline swizzle analysis ----
        // We read the 16 fp16 elements at every linear SMEM index; if SW64
        // is working as expected, this dump will be a SCRAMBLED version of
        // B (logical), and the host can XOR-decode it. If TMA failed, the
        // dump is all zeros / garbage / partial.
        if (B_smem_dump != nullptr) {
            const int total_B = K * N;
            for (int idx = tid; idx < total_B; idx += blockDim.x) {
                B_smem_dump[idx] = B_smem[idx];
            }
        }
        __syncthreads();

        // ---- Variant 5 stops here — D stays zero, no WGMMA executed ----
        // This isolates TMA correctness from WGMMA descriptor questions.
        // Host-side multiset_match on B_smem_dump tells us whether TMA
        // produced the right bytes (multiset_match = 0) regardless of
        // whether SW64 scrambled their positions.
        if constexpr (VARIANT == 5) return;
    } else {
        // ---- variants 0-4: original TMA-both path ----
        if (tid == 0) {
            mbarrier_init(&mbar, 2);
            fence_proxy_async_shared_cta();
            const uint32_t bytes_A = M * K * sizeof(__half);
            const uint32_t bytes_B = K * N * sizeof(__half);
            mbarrier_arrive_expect_tx(&mbar, bytes_A);
            mbarrier_arrive_expect_tx(&mbar, bytes_B);
            cp_async_bulk_tensor_2d_g2s(&A_smem[0], &A_tma_map, 0, 0, &mbar);
            cp_async_bulk_tensor_2d_g2s(&B_smem[0], &B_tma_map, 0, 0, &mbar);
        }
        __syncthreads();
        mbarrier_wait(&mbar, 0);
        __syncthreads();
    }

    // ---- WGMMA descriptor encoding (3 variants tested) ----
    uint64_t a_desc, b_desc;
    if constexpr (VARIANT == 0) {
        // Row-major encoding: stride byte offset = 1 element = 2 bytes,
        // leading byte offset = row stride = K (or N for B) * sizeof(half).
        // For non-swizzled wgmma input, the SMEM layout is canonical 8x8 cores;
        // row-major LBO interpretation: LBO = inner_dim_bytes between K-row groups.
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ K * sizeof(__half),
                                  /*SBO=*/ sizeof(__half), /*swizzle=*/ 0);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ N * sizeof(__half),
                                  /*SBO=*/ sizeof(__half), /*swizzle=*/ 0);
    } else if constexpr (VARIANT == 1) {
        // "Core-major" encoding (matches v5's existing g_cores layout where
        // SBO = 128 bytes, LBO = N_CORES * 128). For our shape:
        //   B SMEM size = K * N * 2 = 1024 bytes
        //   N_CORES (B's N direction) = N/8 = 4
        //   K_CORES (B's K direction) = K/8 = 2
        //   M_CORES (A's M direction) = M/8 = 8
        // v5-style: SBO = 128 (one 8x8 core = 64 elements = 128 bytes),
        //           LBO = N_CORES * 128 for B, M_CORES * 128 for A.
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ 8 * 128,
                                  /*SBO=*/ 128, /*swizzle=*/ 0);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ 4 * 128,
                                  /*SBO=*/ 128, /*swizzle=*/ 0);
    } else if constexpr (VARIANT == 2) {
        // 128B swizzle. Per PTX ISA Table 26 the swizzle field bits 62-63
        // are: 00=NONE, 01=128B, 10=64B, 11=32B. Earlier this variant set
        // swizzle=3 (= 32B) which contradicted CU_TENSOR_MAP_SWIZZLE_128B
        // on the TMA side; corrected to swizzle=1.
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ 1024,
                                  /*SBO=*/ 1024, /*swizzle=*/ 1);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ 1024,
                                  /*SBO=*/ 1024, /*swizzle=*/ 1);
    } else if constexpr (VARIANT == 3) {
        // Row-major canonical wgmma layout:
        //   A [M=64, K=16] half row-major in SMEM:
        //     row stride = K × 2 = 32 bytes
        //     LBO_A = stride between adjacent M-cores (8 M-rows) = 8 × 32 = 256 B
        //     SBO_A = stride between adjacent K-cores within same M-tile
        //           = 8 × 2 = 16 bytes
        //   B [K=16, N=32] half row-major in SMEM:
        //     row stride = N × 2 = 64 bytes
        //     LBO_B = stride between adjacent K-cores (8 K-rows) = 8 × 64 = 512 B
        //     SBO_B = stride between adjacent N-cores within same K-tile
        //           = 8 × 2 = 16 bytes
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ 256,
                                  /*SBO=*/ 16, /*swizzle=*/ 0);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ 512,
                                  /*SBO=*/ 16, /*swizzle=*/ 0);
    } else if constexpr (VARIANT == 4) {
        // Same as variant 3 but with LBO/SBO swapped (in case I inverted the
        // M/N vs K convention assignment).
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ 16,
                                  /*SBO=*/ 256, /*swizzle=*/ 0);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ 16,
                                  /*SBO=*/ 512, /*swizzle=*/ 0);
    } else if constexpr (VARIANT == 5 || VARIANT == 6) {
        // SW64-matched: B is TMA-loaded under CU_TENSOR_MAP_SWIZZLE_64B,
        // and the WGMMA descriptor advertises swizzle bits = 2 (= 64B).
        // A stays manual core-major (= v5 / v6.1a's W_cores layout).
        // (Variant 5 never reaches WGMMA so this descriptor is unused there.)
        //
        // PTX trans_a=1 / trans_b=1 convention (cross-checked against v5
        // production code which uses trans=1,1 + LBO=M_CORES*128 + SBO=128):
        //   LBO = stride between K-cores (the contracting-dim cores)
        //   SBO = stride between M-cores (or N-cores for B)
        //
        // A (manual core-major v5 layout):
        //   M_CORES=8, K_CORES=2; m_core+1 = +128 B, k_core+1 = +M_CORES*128 B.
        //   → LBO_A = 1024,  SBO_A = 128,  swizzle = 0.
        //
        // B (TMA SW64 row-major [K=16, N=32] fp16):
        //   row stride = N * sizeof(half) = 64 B = exactly SW64 swizzle width
        //   k_core+1 = 8 K-rows × 64 B/row = 512 B  → LBO_B = 512
        //   n_core+1 = 8 N-cols × 2 B/elem = 16 B   → SBO_B = 16
        //   swizzle  = 2 (= 64B) — matches CU_TENSOR_MAP_SWIZZLE_64B on TMA.
        a_desc = encode_smem_desc(&A_smem[0], /*LBO=*/ 1024,
                                  /*SBO=*/ 128, /*swizzle=*/ 0);
        b_desc = encode_smem_desc(&B_smem[0], /*LBO=*/ 512,
                                  /*SBO=*/ 16, /*swizzle=*/ 2);
    }

    // ---- WGMMA m64n32k16 ----
    // Each thread holds 16 fp32 fragment values (M_TILES=1, N_TILES=1).
    //
    // Variants 0/1/2 use the trans_a=1, trans_b=1 helper (col-major SMEM
    // interpretation, as in v5's manual core-major store). They are KNOWN
    // WRONG against TMA's row-major output; kept as negative controls.
    //
    // Variants 3/4 use the trans_a=0, trans_b=0 helper (natural row-major
    // SMEM, which is what TMA produces). Variant 3's LBO/SBO is canonical;
    // variant 4 has them swapped as a sanity-check negative control.
    float acc[16] = {0};

    wgmma_fence();
    if constexpr (VARIANT == 3 || VARIANT == 4) {
        wgmma_m64n32k16_f16_rm(acc, a_desc, b_desc, /*scale_d=*/ false);
    } else {
        // Variants 0/1/2/6 all use trans_a=1, trans_b=1.
        // - 0/1: col-major-style descriptor over TMA row-major SMEM (broken).
        // - 2:   128B swizzle (broken: B too narrow for 128B).
        // - 6:   manual core-major A (M-leading) + SW64 row-major B (K-leading).
        wgmma_m64n32k16_f16(acc, a_desc, b_desc, /*scale_d=*/ false);
    }
    wgmma_commit_group();
    wgmma_wait_group<0>();
    __syncthreads();

    // ---- Store fragment to D[M=64, N=32] fp32 ----
    // Fragment layout: m64n32 has 4 "n_chunks" of 8 cols each.
    // Each thread holds 16 values arranged as 4 chunks × 4 elements.
    // Per chunk: e=0,1,2,3 → 4 elements at specific (row, col) positions.
    //
    // Position formula (matches v5's Phase 5 store):
    //   groupID = lane_id / 4
    //   tigid   = lane_id % 4
    //   For e ∈ {0,1,2,3}:
    //     row_in_warp = (e < 2 ? groupID : groupID + 8)
    //     col_in_chunk = tigid * 2 + (e % 2)
    //   row_in_tile = warp_id * 16 + row_in_warp
    //   col_in_tile = chunk_e * 8 + col_in_chunk
    #pragma unroll
    for (int chunk_e = 0; chunk_e < 4; chunk_e++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const int frag_idx = chunk_e * 4 + e;
            const int groupID = lane_id / 4;
            const int tigid   = lane_id % 4;
            int row_in_warp, col_in_chunk;
            switch (e) {
                case 0: row_in_warp = groupID;     col_in_chunk = tigid*2 + 0; break;
                case 1: row_in_warp = groupID;     col_in_chunk = tigid*2 + 1; break;
                case 2: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 0; break;
                case 3: row_in_warp = groupID + 8; col_in_chunk = tigid*2 + 1; break;
            }
            const int row = warp_id * 16 + row_in_warp;
            const int col = chunk_e * 8 + col_in_chunk;
            if (row < M && col < N) {
                D[row * N + col] = acc[frag_idx];
            }
        }
    }
}

}  // namespace

// =============================================================================
// PyTorch entry point — runs the test for a chosen variant.
// =============================================================================

std::vector<torch::Tensor> wgmma_tma_test(
    const torch::Tensor& A,    // [64, 16] half
    const torch::Tensor& B,    // [16, 32] half
    int64_t variant            // 0..5
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda());
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be fp16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be fp16");
    TORCH_CHECK(A.size(0) == 64 && A.size(1) == 16, "A must be [64, 16]");
    TORCH_CHECK(B.size(0) == 16 && B.size(1) == 32, "B must be [16, 32]");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous());
    TORCH_CHECK(variant >= 0 && variant <= 6, "variant must be 0..6");

    auto fp32_opts = torch::TensorOptions().device(A.device()).dtype(torch::kFloat32);
    auto fp16_opts = torch::TensorOptions().device(A.device()).dtype(torch::kFloat16);
    torch::Tensor D = torch::zeros({64, 32}, fp32_opts);
    // B_smem_dump: variant 5 writes its 1024-byte SMEM here for offline
    // swizzle inspection. Other variants leave it zero.
    torch::Tensor B_smem_dump = torch::zeros({16, 32}, fp16_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Build CUtensorMaps for A [64, 16] and B [16, 32].
    // For TMA, the descriptor sees the global tensor as 2D row-major.
    // - innermost dim (dim 0 in CUtensorMap convention) = trailing tensor dim
    // - global stride[0] = innermost row stride in bytes
    alignas(64) CUtensorMap A_tma_map;
    alignas(64) CUtensorMap B_tma_map;

    // Per-variant TMA swizzle setting. Driver enforces:
    //   interleave=NONE, swizzle=S → boxDim[0] * elem_size ≤ S bytes.
    //   variant 2 (SW128) and variant 5 (SW64) both satisfy this for B.
    CUtensorMapSwizzle tma_swizzle_A = CU_TENSOR_MAP_SWIZZLE_NONE;
    CUtensorMapSwizzle tma_swizzle_B = CU_TENSOR_MAP_SWIZZLE_NONE;
    if (variant == 2) tma_swizzle_B = CU_TENSOR_MAP_SWIZZLE_128B;
    if (variant == 5 || variant == 6) tma_swizzle_B = CU_TENSOR_MAP_SWIZZLE_64B;

    {
        // A is [64, 16] half, row-major.
        const cuuint64_t global_dim_A[2]     = { 16, 64 };           // (innermost K, outer M)
        const cuuint64_t global_strides_A[1] = { 16 * sizeof(__half) };
        const cuuint32_t box_dim_A[2]        = { 16, 64 };
        const cuuint32_t element_strides_A[2] = { 1, 1 };
        CUresult err = cuTensorMapEncodeTiled(
            &A_tma_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            /*tensorRank=*/ 2,
            A.data_ptr(),
            global_dim_A,
            global_strides_A,
            box_dim_A,
            element_strides_A,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            tma_swizzle_A,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(err == CUDA_SUCCESS,
                    "cuTensorMapEncodeTiled A failed: code=", (int)err);
    }
    {
        // B is [16, 32] half, row-major.
        const cuuint64_t global_dim_B[2]     = { 32, 16 };           // (innermost N, outer K)
        const cuuint64_t global_strides_B[1] = { 32 * sizeof(__half) };
        const cuuint32_t box_dim_B[2]        = { 32, 16 };
        const cuuint32_t element_strides_B[2] = { 1, 1 };
        CUresult err = cuTensorMapEncodeTiled(
            &B_tma_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            /*tensorRank=*/ 2,
            B.data_ptr(),
            global_dim_B,
            global_strides_B,
            box_dim_B,
            element_strides_B,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            tma_swizzle_B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        TORCH_CHECK(err == CUDA_SUCCESS,
                    "cuTensorMapEncodeTiled B failed: code=", (int)err);
    }

    const __half* A_global   = (const __half*)A.data_ptr();
    __half*       dump_ptr   = (__half*)B_smem_dump.data_ptr();
    float*        D_ptr      = D.data_ptr<float>();

    // Launch test kernel (1 CTA, 128 threads = 1 warpgroup).
    if (variant == 0) {
        wgmma_tma_test_kernel<0><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else if (variant == 1) {
        wgmma_tma_test_kernel<1><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else if (variant == 2) {
        wgmma_tma_test_kernel<2><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else if (variant == 3) {
        wgmma_tma_test_kernel<3><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else if (variant == 4) {
        wgmma_tma_test_kernel<4><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else if (variant == 5) {
        wgmma_tma_test_kernel<5><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    } else {
        wgmma_tma_test_kernel<6><<<1, 128, 0, stream>>>(
            A_tma_map, B_tma_map, A_global, D_ptr, dump_ptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {D, B_smem_dump};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wgmma_tma_test", &wgmma_tma_test,
          "Standalone TMA→WGMMA m64n32k16 test. variant ∈ {0..5}. Returns "
          "(D [64, 32] fp32, B_smem_dump [16, 32] fp16).",
          py::arg("A"), py::arg("B"), py::arg("variant"));
}
