// Backward v6 — FA3-pattern bwd kernel (incremental TMA + warp-spec roadmap).
//
// PHASE STATUS: v6.0 — v5-equivalent clone. Build pipeline + binding test.
//   Same kernel body as v5 (3-deep cp.async + register-resident dC + fp16
//   wgmma + XOR-swizzle on C_smem). Only renames (v5 → v6). Speed and
//   parity must match v5 within ±2% — this is the v6.0 gate.
//
// PHASES PLANNED (one per chat-turn iteration):
//   v6.0 (THIS): clone v5 → v6, validate build pipeline + parity
//   v6.1a:       TMA plumbing on C_smem only (validates cuTensorMapEncodeTiled
//                + mbarrier + extension linking with CUDA Driver API; not a
//                wgmma operand so layout-risk-free)
//   v6.1b:       standalone TMA→WGMMA tiny GEMM test (resolves descriptor /
//                LBO / SBO / swizzle math BEFORE touching production bwd)
//   v6.2:        TMA for g_cores per-chunk + 128B swizzle + matched WGMMA
//                descriptor swizzle bits — gated on v6.1b passing
//   v6.3:        warp specialization (1 producer + 3 consumer warps,
//                mbarrier circular pipeline, drop monolithic __syncthreads)
//   v6.4:        setmaxnreg.dec/inc register rebalancing (producer 40,
//                consumer 232) — only meaningful with warp specialization
//
// v6 ships only if at end of phases: dC max_rel ≤ 5e-3 at N=32768/65536
// AND v6/Triton ≥ 1.15× at both shapes. Otherwise v5 stays default.
//
// Inherited goals (from v5):
//   1. Match v11 fwd's precision strategy: cast B to fp16 (not bf16) before
//      wgmma so we keep ~3 more mantissa bits than v1 bwd's bf16(B) path.
//   2. Hold dC accumulator in registers across chunks within a block —
//      eliminates SMEM/HBM round-trip for fragment.
//   3. No global atomic on dC: each block uniquely owns (h_tile, n_part)
//      and writes to dC_scratch[h_tile, n_part, m, c] in fp32.  A reduce
//      kernel then sums across n_part.
//
// Grid layout (PRECISION + STRUCTURE differs from v1, MATH identical):
//   Block: 128 threads (1 warpgroup)
//   grid.x = N_PARTS         (split N for occupancy, e.g. 4)
//   grid.y = ceil(H, BLOCK_H)
//
//   Each block iterates CHUNKS = (N / N_PARTS) / BLOCK_N chunks of N
//   sequentially, accumulating fragments in register `dC_acc[16]`.  After
//   all chunks, the fragment is stored to dC_scratch[h_tile, n_part, ...].
//
//   dz output: each (n, h) is unique → plain store, no atomic.
//
// Math contract (BIT-EQUAL to v1 modulo precision-floor):
//   dC[j, b, c] = Σ_n W[n, m=j_local*L_PAD+b] · g[n, c]
//   dz[n, j]    = scale · Σ_c g[n,c] · Σ_k dB_k(τ_nj) · C[j, bin+k, c]
//
// Precision: fp32 acc / fp32 dC_scratch / fp32 dz.  Only the *operand*
// types change (bf16→fp16) — accumulators stay fp32 throughout.

#include <cuda.h>          // CUtensorMap + cuTensorMapEncodeTiled (driver API)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstring>

namespace {

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

// f32.f16.f16 with trans_a=1, trans_b=1 (matches v1 bwd's W[K][M] / g[K][N] layouts)
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

__device__ __forceinline__ void cp_async_16_h(__half* dst, const __half* src, bool valid) {
    unsigned d = __cvta_generic_to_shared(dst);
    if (valid) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(d), "l"(src));
    } else {
        #pragma unroll
        for (int i = 0; i < 8; i++) dst[i] = __float2half(0.0f);
    }
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}
// Wait until at most N cp.async commit groups are still outstanding.
// Used by the double-buffered g_cores pipelining: after issuing chunk c+1's
// prefetch, wait_group<1> drains chunk c (older) while keeping chunk c+1 in
// flight for the next iteration.
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

// =============================================================================
// v6.1a — Hopper TMA + mbarrier helpers (replaces cp.async for C_smem load).
// Refs: PTX ISA 9.7.3.2 (cp.async.bulk.tensor), 9.7.5 (mbarrier);
//       Colfax TMA tutorial; CUTLASS 3 cute/atom/copy_traits_sm90_tma.hpp.
// =============================================================================

// Initialize a SMEM mbarrier with target arrival count.
// For TMA loads: the byte-count signaled via arrive_expect_tx accounts for
// 1 implicit arrival on completion, so total_arrives_expected = arrival_count.
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t arrival_count) {
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(mbar_addr), "r"(arrival_count)
    );
}

// Arrive on barrier and register expected-transaction byte count.
// Producer thread calls this BEFORE issuing the TMA — TMA hardware will
// credit the bytes upon completion, signaling the barrier.
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(mbar_addr), "r"(bytes)
    );
}

// Wait until the mbarrier's parity bit flips to `phase`.
// Spin on try_wait.parity until predicate becomes 1 (= barrier reached new phase).
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
        "}\n"
        :: "r"(mbar_addr), "r"(phase)
    );
}

// Issue 2D TMA load: global tensor (per CUtensorMap) → SMEM, signaling mbar.
// Only ONE thread per CTA should issue (TMA is a single-thread op).
//   smem_ptr  : 16-byte aligned SMEM destination
//   tensor_map: pointer to CUtensorMap (host-encoded via cuTensorMapEncodeTiled)
//   coord0    : innermost dimension coordinate (in TENSOR ELEMENTS, not bytes)
//   coord1    : outer dimension coordinate
//   mbar      : SMEM mbarrier — TMA hardware signals on completion
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
// v5 Backward kernel — register-resident dC + fp16 wgmma.
//
// Tile contract:
//   BLOCK_N = K dim (per-chunk), e.g. 128
//   BLOCK_H * L_PAD = M dim, multiple of 64
//   R = N dim (32 or 64)
//
// Grid: (N_PARTS, ceil(H, BLOCK_H))
//   N_PARTS splits the N axis for occupancy.  CHUNKS_PER_BLOCK = N / N_PARTS / BLOCK_N
// =============================================================================
// `chunks_per_block` is a RUNTIME arg (not a template parameter): the
// chunk loop has `#pragma unroll 1` so the compiler emits identical SASS
// whether the bound is a compile-time constant or a register int. Making
// it runtime collapses the previous {2,4,8} dispatch tree into a single
// kernel binary that handles any (N, NPARTS, BLOCK_N) combination
// satisfying N % (NPARTS × BLOCK_N) == 0 — including the production d20
// microbatch (N=65536, NPARTS=4, BN=128 → chunks_per_block=128).
template <int BLOCK_N, int BLOCK_H, int L_PAD, int R, int N_PARTS>
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor):
//   (128, 2) — empirically optimal. Tested (128, 1) on 2026-05-03: parity at
//   N=32768 (~6.77 ms, no change), but catastrophic regression at N=65536
//   (>5 s/call timeout vs (128, 2)'s 13.56 ms). Hypothesis: with relaxed
//   hint, the compiler raises per-thread register usage; at chunks_per_block
//   ≥ 128 the long-lived `acc[48 fp32]` accumulator + per-chunk transients
//   cross the spill threshold, and N=65536 amplifies the spill cost across
//   128 serial chunks. (128, 2) keeps the budget tight enough to avoid that.
__global__ void __launch_bounds__(128, 2)
spline_kv_bwd_v6_kernel(
    const __nv_bfloat16* __restrict__ z,
    // v6.1a: C_fp16 pointer replaced by a CUtensorMap describing the global
    // C_fp16 [H, L, R] tensor.  Pass-by-value as __grid_constant__ so all
    // CTAs share the read-only descriptor.  The 128-byte struct is set up
    // host-side via cuTensorMapEncodeTiled (see wrapper below).
    const __grid_constant__ CUtensorMap C_tma_map,
    const __half* __restrict__ g_delta_fp16,   // [N, R]    fp16 (caller cast)
    float* __restrict__ dC_scratch,            // [H, N_PARTS, L, R] fp32
    float* __restrict__ dz,                     // [N, H] fp32
    const int N, const int H, const int L,
    const float grid_lo, const float scale,
    const int chunks_per_block
) {
    constexpr int M       = BLOCK_H * L_PAD;
    constexpr int M_TILES = M / 64;
    constexpr int N_TILES = R / 32;
    constexpr int K_TILES = BLOCK_N / 16;
    static_assert(M % 64 == 0,        "M must be multiple of 64");
    static_assert(BLOCK_N % 16 == 0,  "BLOCK_N must be multiple of 16");
    static_assert(R % 32 == 0,        "R must be multiple of 32");

    constexpr int M_CORES = M / 8;
    constexpr int N_CORES = R / 8;
    constexpr int K_CORES = BLOCK_N / 8;

    const int pid_part = blockIdx.x;          // 0..N_PARTS-1
    const int pid_h    = blockIdx.y;
    const int h_start  = pid_h * BLOCK_H;
    const int n_per_part = N / N_PARTS;       // tokens this block owns
    const int n_part_start = pid_part * n_per_part;

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    __shared__ __align__(128) __half W_cores[K_CORES * M_CORES * 64];
    // Triple-buffered g_cores (3-deep cp.async pipeline): prefetch chunk c+2
    // while chunk c is computing.  At any point up to 2 prefetches are in
    // flight, fully hiding cp.async latency.  Cost: +4 KB SMEM total
    // (g_cores is 2 KB/buffer × 3 = 6 KB; was 4 KB at 2-buffer).
    __shared__ __align__(128) __half g_cores[3][K_CORES * N_CORES * 64];
    // C_smem with XOR-based 2-way bank swizzle (Phase 3b conflict mitigation).
    // The original [BLOCK_H][L_PAD][R] layout has j_local row stride
    // 24 × 32 × 2 = 1536 bytes = 384 word-stride mod 32 banks = 0 → all
    // j_locals within a warp hit the SAME 32-bank line, causing 4-way
    // conflict in Phase 3b (each warp serves 8 j_local values per pass).
    //
    // Mitigation: XOR the c index by ((j_local & 3) << 3), i.e. flip bits
    // [3..4] of c based on j_local's low 2 bits. This is bit-symmetric, so
    // Phase A (writes) and Phase 3b (reads) round-trip to the same data.
    //
    // Why this preserves cp.async 16-byte alignment: cp.async writes 8-fp16
    // contiguous blocks (16 bytes = c_start ∈ {0, 8, 16, 24}). The XOR
    // affects bits ≥ 3 of c, so c_start is mapped to a different aligned
    // 8-fp16 block (e.g. j=1 swaps c_start=0 ↔ c_start=8). Each cp.async
    // still hits one contiguous 16-byte destination.
    //
    // Bank conflict outcome:
    //   Original:    8 j_local values → 1 bank slot  (4-way conflict)
    //   This XOR:    8 j_local values → 4 bank slots (2-way conflict)
    //   Full elim. (TMA 128B): 8 → 8 banks. Requires TMA, deferred to v6.
    //
    // C_smem is read only in Phase 3b's scalar fp32 dz inner; not a WGMMA
    // operand → no impact on WGMMA descriptor encoding.
    __shared__ __half C_smem[BLOCK_H][L_PAD][R];

    // Swizzle c-index based on j_local. Symmetric: A ⊕ B ⊕ B = A so
    // Phase A writes and Phase 3b reads with the same swz_c() round-trip
    // to the same value at the same logical (j, b, c).
    auto swz_c = [&](int j, int c) -> int {
        return c ^ ((j & 3) << 3);
    };

    #define W_OFF(k, m)  (((k) >> 3) * M_CORES * 64 + ((m) >> 3) * 64 + ((k) & 7) * 8 + ((m) & 7))
    #define G_OFF(k, n)  (((k) >> 3) * N_CORES * 64 + ((n) >> 3) * 64 + ((k) & 7) * 8 + ((n) & 7))

    constexpr uint32_t LBO_A = M_CORES * 128;
    constexpr uint32_t SBO_A = 128;
    constexpr uint32_t LBO_B = N_CORES * 128;
    constexpr uint32_t SBO_B = 128;

    // Issue cp.async load of g[n_start..n_start+BLOCK_N] into g_cores[buf].
    // Caller is responsible for calling cp_async_commit() afterwards.
    auto issue_g_load = [&](int buf, int n_start_local) {
        constexpr int total_rows = K_CORES * N_CORES * 8;
        for (int idx = tid; idx < total_rows; idx += blockDim.x) {
            const int k_core = idx / (N_CORES * 8);
            const int rem    = idx % (N_CORES * 8);
            const int n_core = rem / 8;
            const int k_in   = rem % 8;
            const int k_global = k_core * 8 + k_in;
            const int n_start_in_core = n_core * 8;
            const int n_global_idx = n_start_local + k_global;
            __half* dst =
                &g_cores[buf][((k_core * N_CORES) + n_core) * 64 + k_in * 8];
            const __half* src = (n_global_idx < N)
                ? &g_delta_fp16[n_global_idx * R + n_start_in_core]
                : nullptr;
            cp_async_16_h(dst, src, n_global_idx < N);
        }
    };

    // ---- Phase A (v6.1a): TMA-load C[h_start..h_start+BH] into SMEM ----
    //
    // Each block loads BH 2D tiles of shape [L, R] half from global C, one
    // per (h_start + j_local).  The C global tensor is described by a 2D
    // CUtensorMap with shape [R, H*L] (innermost first) and box [R, L].
    // For j_local in 0..BH-1, the call coord1 = (h_start + j_local) * L picks
    // the right slab of L rows.
    //
    // OOB handling: if h_start + j_local >= H, the descriptor's
    // CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE setting fills the destination with
    // zeros — matches v5's `valid=false` cp_async_16_h zero behavior.
    //
    // SMEM layout: each TMA call writes its [L, R] tile contiguously starting
    // at &C_smem[j_local][0][0]; positions [j_local][L..L_PAD-1][...] remain
    // uninitialized (Phase 3b never reads them; bin_idx + k ≤ L-1 always).
    //
    // NOTE: v5's XOR swizzle on c is dropped here because TMA writes a
    // contiguous tile (can't apply per-element swizzle on the store path).
    // Phase 3b reverts to non-swizzled access — gives back ~0.5-1% speed
    // vs v5+XOR but isolates the TMA correctness signal. v6.1c can restore
    // by switching to TMA's native 128B swizzle (auto bank-conflict-free).
    {
        // Mbarrier with arrival count = BH (one per TMA call).
        __shared__ uint64_t C_mbar;
        if (tid == 0) {
            mbarrier_init(&C_mbar, BLOCK_H);
            // Fence: ensures the mbarrier init is visible to the async proxy
            // (TMA hardware) before issuing transfers.
            fence_proxy_async_shared_cta();
        }
        __syncthreads();

        if (tid == 0) {
            const uint32_t bytes_per_h = (uint32_t)L * (uint32_t)R * sizeof(__half);
            #pragma unroll
            for (int j_local = 0; j_local < BLOCK_H; j_local++) {
                mbarrier_arrive_expect_tx(&C_mbar, bytes_per_h);
                cp_async_bulk_tensor_2d_g2s(
                    /*smem_ptr=*/   &C_smem[j_local][0][0],
                    /*tensor_map=*/ &C_tma_map,
                    /*coord0 (R inner)=*/ 0,
                    /*coord1 (H*L outer)=*/ (h_start + j_local) * L,
                    /*mbar=*/       &C_mbar);
            }
        }

        // All threads wait for TMA completion (phase 0 = first crossing).
        mbarrier_wait(&C_mbar, /*phase=*/ 0);
        __syncthreads();
    }
    // (No cp_async_commit() here — TMA uses mbarrier signaling, not the
    // legacy commit_group/wait_group pattern. The g_cores pre-loop below
    // resumes the cp.async path for g loads.)

    // ---- Pre-loop: kick off chunk-0 AND chunk-1 g prefetches (3-deep pipeline) ----
    // We always have at most 2 prefetches in flight (chunks c+1, c+2) plus the
    // current chunk being computed (already drained). The 3 g_cores buffers
    // rotate by `chunk % 3`.
    if (chunks_per_block > 0) {
        issue_g_load(0, n_part_start);
        cp_async_commit();    // group 1 = chunk 0 g
    }
    if (chunks_per_block > 1) {
        issue_g_load(1, n_part_start + BLOCK_N);
        cp_async_commit();    // group 2 = chunk 1 g
    }

    // Drain chunk 0's g (cp.async group 0); leave chunk 1's g (group 1) in
    // flight so iter 0 starts with one prefetch already in motion.
    // (v6.1a: Phase A no longer uses a cp.async group — TMA + mbarrier
    // already waited on its own signaling.)
    if (chunks_per_block > 1) {
        cp_async_wait_group<1>();
    } else {
        cp_async_wait_group<0>();
    }
    __syncthreads();

    // ---- dC fragment register accumulator (across CHUNKS) ----
    // M_TILES * N_TILES * 16 fp32 per thread.  For M=192, M_TILES=3,
    // N_TILES=1 → 48 fp32 / thread.  Lives in registers across all chunks.
    float acc[M_TILES * N_TILES * 16];
    #pragma unroll
    for (int i = 0; i < M_TILES * N_TILES * 16; i++) acc[i] = 0.0f;

    // ---- Main chunk loop (3-stage pipeline) ----
    // For each chunk c:
    //   Phase 1  : zero W_cores
    //   Phase 2  : issue prefetch for chunk c+1 → g_cores[(c+1)&1]
    //   Wait     : drain chunk c's g (1 group still in flight if prefetched)
    //   Phase 3a : fill W_cores from z (B-spline values, SMEM stores only)
    //   Phase 4  : ISSUE wgmma async (multi-stage commit, no per-tile wait)
    //   Phase 3b : compute dz inner — runs in parallel with Phase 4 wgmma
    //              (Phase 3b reads C_smem + g_cores[buf] + writes dz to global;
    //               does NOT touch W_cores or acc[] — safe alongside wgmma)
    //   Wait     : wgmma_wait_group<0>
    //
    // chunks_per_block is runtime; #pragma unroll 1 keeps codegen identical
    // to a compile-time bound.
    #pragma unroll 1
    for (int chunk = 0; chunk < chunks_per_block; chunk++) {
        const int n_start    = n_part_start + chunk * BLOCK_N;
        const int buf        = chunk % 3;                          // 3-deep rotation
        const bool has_next1 = (chunk + 1 < chunks_per_block);
        const bool has_next2 = (chunk + 2 < chunks_per_block);

        // --- Phase 1: zero W_cores ---
        {
            constexpr int total_elems = K_CORES * M_CORES * 64;
            #pragma unroll
            for (int idx = tid * 8; idx < total_elems; idx += blockDim.x * 8) {
                uint4* p = reinterpret_cast<uint4*>(&W_cores[idx]);
                *p = make_uint4(0, 0, 0, 0);
            }
        }

        // --- Phase 2: issue prefetch for chunk c+2 into the buffer it will use ---
        // (chunk c+1 was already prefetched in the previous iteration / pre-loop.)
        // Runs in parallel with Phase 1's W_cores zeroing.
        if (has_next2) {
            issue_g_load((chunk + 2) % 3,
                         n_part_start + (chunk + 2) * BLOCK_N);
            cp_async_commit();
        }

        // --- Wait for THIS chunk's g (oldest in-flight group) ---
        //   has_next2: groups in flight = {c, c+1, c+2} → wait_group<2> drains c
        //   has_next1 only (i.e. c == N-2): {c, c+1} → wait_group<1> drains c
        //   neither (last chunk): {c} → wait_group<0> drains c
        if (has_next2) {
            cp_async_wait_group<2>();
        } else if (has_next1) {
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // --- Phase 3a: fill W_cores from z (B-spline values only, no dz) ---
        // Fast pass: 1024 (n,j) pairs × 3 SMEM stores; no R-loop.
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
                W_cores[W_OFF(n_local, col_base + 0)] = f2h(B0);
                W_cores[W_OFF(n_local, col_base + 1)] = f2h(B1);
                W_cores[W_OFF(n_local, col_base + 2)] = f2h(B2);
            }
        }
        __syncthreads();
        fence_proxy_async_shared_cta();

        // --- Phase 4: ISSUE wgmma async, multi-stage commit (no per-tile wait) ---
        // Hopper allows multiple commit groups in flight; each (m_tile, n_tile)
        // writes to a DIFFERENT fragment in acc[], so they're independent.
        // Single fence at start, single wait at end of chunk — far less serial
        // than the previous per-tile wait pattern.
        wgmma_fence();
        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; m_tile++) {
            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; n_tile++) {
                const int frag_base = (m_tile * N_TILES + n_tile) * 16;
                #pragma unroll
                for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                    __half* a_ptr =
                        &W_cores[((k_tile * 2) * M_CORES + m_tile * 8) * 64];
                    __half* b_ptr =
                        &g_cores[buf][((k_tile * 2) * N_CORES + n_tile * 4) * 64];
                    uint64_t a_desc = encode_smem_desc(a_ptr, LBO_A, SBO_A, 0);
                    uint64_t b_desc = encode_smem_desc(b_ptr, LBO_B, SBO_B, 0);
                    // scale_d = false only at the very first wgmma op of the very
                    // first chunk (overwrite); accumulate thereafter.
                    bool scale_d = (chunk > 0) || (k_tile > 0);
                    wgmma_m64n32k16_f16(&acc[frag_base], a_desc, b_desc, scale_d);
                }
                wgmma_commit_group();    // commit (m_tile, n_tile) group; do NOT wait
            }
        }
        // wgmma is now in flight — Phase 3b runs concurrently below.

        // --- Phase 3b: dz inner (parallel with Phase 4 wgmma) ---
        // Reads z (re-read; second pass, hits L1 cache from Phase 3a),
        // C_smem, and g_cores[buf].  Writes dz to global memory.
        // Does NOT touch W_cores or acc[] → safe alongside wgmma.
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
                    // v6.1a: C_smem accesses non-swizzled (XOR removed
                    // along with the cp.async path). 4-way bank conflict
                    // returns; will be addressed by TMA 128B swizzle in v6.2.
                    const float g  = h2f(g_cores[buf][G_OFF(n_local, c)]);
                    const float c0 = h2f(C_smem[j_local][bin_idx + 0][c]);
                    const float c1 = h2f(C_smem[j_local][bin_idx + 1][c]);
                    const float c2 = h2f(C_smem[j_local][bin_idx + 2][c]);
                    inner += g * (dB0 * c0 + dB1 * c1 + dB2 * c2);
                }
                dz[n_global * H + j_global] = scale * inner;
            }
        }

        // --- Wait for Phase 4 wgmma (all M_TILES × N_TILES commit groups) ---
        wgmma_wait_group<0>();
        __syncthreads();
    }  // end chunk loop

    // ---- Phase 5: store accumulated fragments to dC_scratch (fp32, no atomic) ----
    // Layout: dC_scratch[h_tile, n_part, m_global, c]  size [H, N_PARTS, L, R]
    // Per fragment we know (m_tile, n_tile, frag layout).
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
                        dC_scratch[out_idx] = acc[frag_base + frag_idx];
                    }
                }
            }
        }
    }
    #undef W_OFF
    #undef G_OFF
}

// Reduce dC_scratch[H, N_PARTS, L, R] → dC[H, L, R] bf16
__global__ void spline_kv_bwd_v6_reduce_kernel(
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

// Single launch — chunks_per_block is now a runtime kernel arg, so we no
// longer need the {2,4,8} dispatch tree. NPARTS stays templated because
// the kernel uses it in indexing arithmetic; chunks_per_block is only the
// loop bound, so it's free to be runtime.
#define LAUNCH_BWD_V6(BN, BH, LP, RR, NPARTS) \
    do { \
        const int chunks_per_block = (N / NPARTS) / BN; \
        TORCH_CHECK(chunks_per_block > 0,    "v6 needs N/NPARTS/BN > 0"); \
        TORCH_CHECK((N / NPARTS) % BN == 0,  "v6 needs N/NPARTS divisible by BN"); \
        const int blocks_h = (H + (BH) - 1) / (BH); \
        dim3 grid(NPARTS, blocks_h, 1); \
        dim3 block(128, 1, 1); \
        auto stream = c10::cuda::getCurrentCUDAStream(); \
        spline_kv_bwd_v6_kernel<BN, BH, LP, RR, NPARTS> \
            <<<grid, block, 0, stream>>>( \
                (const __nv_bfloat16*)z_ptr, \
                C_tma_map, \
                (const __half*)g_ptr, \
                dC_scratch_ptr, dz_ptr, \
                N, H, L, grid_lo, scale, chunks_per_block); \
        const int total_dC = H * L * RR; \
        const int reduce_grid = (total_dC + 255) / 256; \
        spline_kv_bwd_v6_reduce_kernel<<<reduce_grid, 256, 0, stream>>>( \
            dC_scratch_ptr, (__nv_bfloat16*)dC_bf16_ptr, H, L, RR, NPARTS); \
    } while(0)

}  // namespace


// =============================================================================
// PyTorch entry point.  Caller passes z (bf16), C (bf16 — we cast to fp16),
// g_delta (bf16 — we cast to fp16).  Returns (dC bf16, dz fp32).
// =============================================================================
std::vector<torch::Tensor> spline_kv_bwd_v6_cuda(
    const torch::Tensor& z,
    const torch::Tensor& C,
    const torch::Tensor& g_delta,
    double grid_lo, double scale,
    int64_t L_arg
) {
    TORCH_CHECK(z.is_cuda() && C.is_cuda() && g_delta.is_cuda());
    TORCH_CHECK(z.dtype() == torch::kBFloat16, "z must be bf16");
    TORCH_CHECK(C.dtype() == torch::kBFloat16, "C must be bf16 (we cast)");
    TORCH_CHECK(g_delta.dtype() == torch::kBFloat16, "g must be bf16 (we cast)");
    TORCH_CHECK(z.is_contiguous() && C.is_contiguous() && g_delta.is_contiguous());

    const int N = z.size(0);
    const int H = z.size(1);
    const int L = (int)L_arg;
    const int R = C.size(2);

    auto bf16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kBFloat16);
    auto fp32_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat32);
    auto fp16_opts = torch::TensorOptions().device(z.device()).dtype(torch::kFloat16);

    // Cast C and g to fp16 for the wgmma f32.f16.f16 path.
    torch::Tensor C_fp16 = C.to(torch::kFloat16);
    torch::Tensor g_fp16 = g_delta.to(torch::kFloat16);

    torch::Tensor dC_bf16 = torch::zeros({H, L, R}, bf16_opts);
    torch::Tensor dz      = torch::zeros({N, H},   fp32_opts);

    void* z_ptr = z.data_ptr();
    void* C_ptr = C_fp16.data_ptr();    // kept for fallback paths
    void* g_ptr = g_fp16.data_ptr();
    void* dC_bf16_ptr = dC_bf16.data_ptr();
    float* dz_ptr = dz.data_ptr<float>();

    // ---- v6.1a: build CUtensorMap for C_fp16 ----
    // Treat C as 2D [H*L, R] with stride [R*sizeof(half), sizeof(half)] —
    // this lets a single descriptor service per-h tile loads via 2D TMA
    // calls with coord1 = (h_start + j_local) * L.  Per-call box [L, R]
    // stays well under 256 (L≤32 in production cells).
    //
    // Driver API requirements:
    //   - tensorMap struct: 64-byte aligned, 128 bytes total
    //   - globalAddress: ≥16-byte aligned (PyTorch CUDA tensors satisfy this)
    //   - boxDim each ≤ 256
    //   - swizzle = NONE for v6.1a (validates plumbing first; v6.2 will
    //     switch to 128B swizzle with matched WGMMA descriptor)
    //   - oobFill = NONE → OOB regions zero-filled (matches v5's
    //     `valid=false` cp_async_16_h zero behavior for j_global ≥ H)
    alignas(64) CUtensorMap C_tma_map;
    {
        const cuuint64_t global_dim[2] = {
            (cuuint64_t)R,            // innermost
            (cuuint64_t)((cuuint64_t)H * (cuuint64_t)L),  // outer (rows = H*L)
        };
        // globalStrides excludes the innermost (which has unit element stride
        // by definition).  For 2-D, only the outer-row byte stride is needed,
        // and that's R * sizeof(half).  Must be a multiple of 16 bytes.
        const cuuint64_t global_strides[1] = {
            (cuuint64_t)R * sizeof(__half),
        };
        const cuuint32_t box_dim[2] = {
            (cuuint32_t)R,            // innermost — full row width
            (cuuint32_t)L,            // outer — L rows per call
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
            CU_TENSOR_MAP_SWIZZLE_NONE,        // v6.1a: no swizzle (v6.2 → 128B)
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE  // OOB → 0 (matches v5 valid=false)
        );
        TORCH_CHECK(err == CUDA_SUCCESS,
                    "cuTensorMapEncodeTiled failed for C_fp16: code=", (int)err);
    }

    // N_PARTS=4 is fixed: dC_scratch = [H, NPARTS, L, R] fp32 = 28.8 MB
    // per layer at H=2560/L=22/R=32, independent of N. Larger NPARTS would
    // grow scratch memory linearly without speeding the kernel (grid is
    // already saturated at 4×ceil(H/BH) blocks ≫ concurrent SM capacity),
    // so we let chunks_per_block grow with N instead. The chunk loop body
    // is hot-path-optimized once (register-resident dC accumulator), so
    // per-token throughput is preserved at any N.
    constexpr int N_PARTS_DEFAULT = 4;
    const int N_PARTS = N_PARTS_DEFAULT;

    torch::Tensor dC_scratch = torch::empty({H, N_PARTS, L, R}, fp32_opts);
    float* dC_scratch_ptr = dC_scratch.data_ptr<float>();

    if (R == 32 && L == 22) {
        LAUNCH_BWD_V6(128, 8, 24, 32, N_PARTS_DEFAULT);
    } else if (R == 32 && L == 16) {
        LAUNCH_BWD_V6(128, 8, 16, 32, N_PARTS_DEFAULT);
    } else if (R == 32 && L == 32) {
        LAUNCH_BWD_V6(128, 8, 32, 32, N_PARTS_DEFAULT);
    } else if (R == 64 && L == 22) {
        LAUNCH_BWD_V6(128, 8, 24, 64, N_PARTS_DEFAULT);
    } else {
        TORCH_CHECK(false, "spline_kv_bwd_v6: unsupported (R, L)");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dC_bf16, dz};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spline_kv_bwd_v6_cuda", &spline_kv_bwd_v6_cuda,
          "v6 bwd: register-resident dC + fp16 wgmma. Returns (dC bf16, dz fp32).",
          py::arg("z"), py::arg("C"), py::arg("g_delta"),
          py::arg("grid_lo"), py::arg("scale"), py::arg("L"));
}
