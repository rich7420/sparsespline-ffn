// CuTe TMA→WGMMA oracle (v7.0 — establishes layout-pipeline truth-table).
//
// Phase 0: smoke test that <cute/tensor.hpp> compiles and CUTLASS headers
//          link cleanly with our build configuration.
// Phase 1a: TMA-only SW64 round-trip — TMA loads B[16,32] half via
//           Layout_MN_SW64_Atom + make_tma_copy, dump SMEM to global, host
//           validates multiset_match==0.
// Phase 1b: WGMMA-only SW64 — manual SMEM fill in CuTe-derived layout, run
//           cute::gemm with MN_SW64 atom; D parity vs torch.matmul.
// Phase 1c: TMA→WGMMA fused — both paths combined, parity gate.
//
// Each phase gated on the previous; entry point dispatches by `phase` arg.
//
// The reason this kernel uses CuTe (not hand-written descriptors) is that the
// hand-written wgmma_tma_test.cu hit silent layout-mismatch errors across 6
// variants. CuTe atoms are pre-validated against PTX descriptor math, so they
// give us a known-good encoding to anchor v7 production against.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// CuTe / CUTLASS headers (header-only, no library link required).
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

// NOTE: cannot use anonymous namespace here — CUTLASS headers also declare
// anonymous-namespace symbols, and NVCC's host stub generation produces
// "reference to ‘_GLOBAL__N__...’ is ambiguous" when both coexist. Use a
// named namespace instead.
namespace cute_oracle_impl {

using namespace cute;

// =============================================================================
// Phase 0 — smoke kernel: prove CuTe templates instantiate on sm_90a.
//
// We pick the same SMEM atom we'll use in Phase 1+ (Layout_MN_SW64_Atom for
// fp16) so that any compile-time issue surfaces here, not later.
// =============================================================================

// =============================================================================
// mbarrier helpers (raw asm — CuTe wraps these but the wrappers vary by
// CUTLASS version; cleaner to keep our own).
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

__global__ void cute_smoke_kernel(int* sentinel) {
    // Instantiate the four GMMA SW atoms we care about — both MN- and K-major
    // forms for both fp16 and bf16 — then dump their dimensions so we can pick
    // a compatible target shape for Phase 1's tile_to_shape.
    using A_MN_SW64 = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using A_K_SW64  = decltype(GMMA::Layout_K_SW64_Atom <half_t>{});
    using A_MN_SW128 = decltype(GMMA::Layout_MN_SW128_Atom<half_t>{});
    using A_K_SW128  = decltype(GMMA::Layout_K_SW128_Atom <half_t>{});

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Pack {M_extent, N_extent} for each atom into pairs in the output:
        //   sentinel[0..1]  = MN_SW64  (M, N)
        //   sentinel[2..3]  = K_SW64   (M, K) or (K, N) depending on side
        //   sentinel[4..5]  = MN_SW128 (M, N)
        //   sentinel[6..7]  = K_SW128  (M, K) or (K, N)
        //   sentinel[8]     = total size of MN_SW64 atom
        //   sentinel[9]     = magic 0xC0DECAFE (success marker)
        sentinel[0] = (int)size<0>(A_MN_SW64{});
        sentinel[1] = (int)size<1>(A_MN_SW64{});
        sentinel[2] = (int)size<0>(A_K_SW64{});
        sentinel[3] = (int)size<1>(A_K_SW64{});
        sentinel[4] = (int)size<0>(A_MN_SW128{});
        sentinel[5] = (int)size<1>(A_MN_SW128{});
        sentinel[6] = (int)size<0>(A_K_SW128{});
        sentinel[7] = (int)size<1>(A_K_SW128{});
        sentinel[8] = (int)size(A_MN_SW64{});
        sentinel[9] = (int)0xC0DECAFE;
    }
}

// =============================================================================
// Phase 1a — TMA-only SW64 round-trip.
//
// Goal: prove that TMA + SW64 SMEM layout speak the same swizzle language.
// 1. TMA loads B[K=16, N=32] half from global into SW64-swizzled SMEM.
// 2. All threads dump SMEM linearly to a global debug buffer.
// 3. Host compares multiset(dump) vs multiset(B). Must be equal even though
//    the linear ordering of dump is scrambled by SW64 swizzle.
// =============================================================================

template <class TmaCopyAtomB>
__global__ void cute_phase1a_kernel(
    CUTE_GRID_CONSTANT TmaCopyAtomB const tma_b,
    half_t* __restrict__ dump_global   // [K * N] linear, host-side reads raw
) {
    using AtomB_T = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayB    = decltype(tile_to_shape(AtomB_T{}, Shape<_32, _16>{}));
    constexpr int K = 16, N = 32;
    constexpr int total_elem = N * K;

    // SMEM allocation. Layout_MN_SW64 wants 1024-byte alignment for the
    // swizzle pattern to land on a clean boundary; use raw bytes + reinterpret.
    __shared__ __align__(1024) uint8_t smem_buf[total_elem * sizeof(half_t)];
    half_t* sB_raw = reinterpret_cast<half_t*>(smem_buf);
    Tensor sB = make_tensor(make_smem_ptr(sB_raw), LayB{});

    __shared__ __align__(8) uint64_t mbar;

    const int tid = threadIdx.x;

    // ---- TMA load via CuTe TMA atom ----
    if (tid == 0) {
        mbar_init(&mbar, 1);
        fence_proxy_async_shared();
        constexpr uint32_t bytes = total_elem * sizeof(half_t);
        mbar_arrive_expect(&mbar, bytes);

        // CuTe TMA pattern: build the global tile view, then partition it
        // through the TMA atom's slice; partition_S/D rearrange the src/dst
        // tensors to match the CopyAtom's instruction layout (single coord
        // per TMA descriptor for the whole tile). Without partition_S/D the
        // CopyAtom static asserts on shape mismatch.
        Tensor gB = tma_b.get_tma_tensor(make_shape(Int<N>{}, Int<K>{}));
        auto thread_tma = tma_b.get_slice(Int<0>{});
        Tensor tBgB = thread_tma.partition_S(gB);
        Tensor tBsB = thread_tma.partition_D(sB);
        // Issue the actual cp.async.bulk.tensor.2d through the atom.
        copy(tma_b.with(reinterpret_cast<uint64_t&>(mbar), 0),
             tBgB, tBsB);
    }
    __syncthreads();
    mbar_wait(&mbar, 0);
    __syncthreads();

    // ---- Dump SMEM linearly ----
    // We read the raw byte buffer (not through sB(...) which would apply
    // the swizzle inverse). The dump captures TMA's actual byte placement;
    // host-side multiset_match reveals whether SW64 was honoured.
    for (int idx = tid; idx < total_elem; idx += blockDim.x) {
        dump_global[idx] = sB_raw[idx];
    }
}

// =============================================================================
// Phase 1c — TMA→WGMMA fused, m64n32k16 f32.f16.f16.
//
// 1. TMA loads A[M=64, K=16] half and B[K=16, N=32] half into SW64-swizzled
//    SMEM via two separate TMA atoms.
// 2. CuTe TiledMma issues a single WGMMA m64n32k16 instruction over the
//    SMEM operands.
// 3. Output fragment is stored back to D_global [64, 32] fp32.
//
// Pass criterion: max_abs_err(D_kernel, torch.matmul(A.float(), B.float())) < 1e-2.
// =============================================================================

template <class TmaCopyAtomA, class TmaCopyAtomB>
__global__ void __launch_bounds__(128, 1)
cute_phase1c_kernel(
    CUTE_GRID_CONSTANT TmaCopyAtomA const tma_a,
    CUTE_GRID_CONSTANT TmaCopyAtomB const tma_b,
    float* __restrict__ D_global   // [M=64, N=32] row-major fp32
) {
    constexpr int M = 64, K = 16, N = 32;

    // A is row-major [M=64, K=16] half: K is the contig dim (stride 1) →
    // K-MAJOR from CuTe's perspective. Use Layout_K_SW32_Atom (Shape<_8, _16>)
    // because K=16 = 32 bytes fits SW32 exactly (SW64's atom K=32 doesn't
    // divide our K=16). Target tile <_64, _16> = (8, 1) atoms.
    //
    // B is row-major [K=16, N=32] half: N is contig → MN-major. Layout_MN_SW64
    // (Shape<_32, _8>) fits N=32=64B and target tile <_32, _16> = (1, 2) atoms.
    using AtomA = decltype(GMMA::Layout_K_SW32_Atom<half_t>{});
    using LayA  = decltype(tile_to_shape(AtomA{}, Shape<_64, _16>{}));
    using AtomB = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayB  = decltype(tile_to_shape(AtomB{}, Shape<_32, _16>{}));

    __shared__ __align__(1024) uint8_t smem_a[size(LayA{}) * sizeof(half_t)];
    __shared__ __align__(1024) uint8_t smem_b[size(LayB{}) * sizeof(half_t)];
    half_t* sA_raw = reinterpret_cast<half_t*>(smem_a);
    half_t* sB_raw = reinterpret_cast<half_t*>(smem_b);

    Tensor sA = make_tensor(make_smem_ptr(sA_raw), LayA{});
    Tensor sB = make_tensor(make_smem_ptr(sB_raw), LayB{});

    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x;

    // ---- TMA load A and B ----
    if (tid == 0) {
        mbar_init(&mbar, 1);
        fence_proxy_async_shared();
        constexpr uint32_t bytes_total =
            (size(LayA{}) + size(LayB{})) * sizeof(half_t);
        mbar_arrive_expect(&mbar, bytes_total);

        Tensor gA = tma_a.get_tma_tensor(make_shape(Int<M>{}, Int<K>{}));
        auto thr_tma_a = tma_a.get_slice(Int<0>{});
        copy(tma_a.with(reinterpret_cast<uint64_t&>(mbar), 0),
             thr_tma_a.partition_S(gA),
             thr_tma_a.partition_D(sA));

        Tensor gB = tma_b.get_tma_tensor(make_shape(Int<N>{}, Int<K>{}));
        auto thr_tma_b = tma_b.get_slice(Int<0>{});
        copy(tma_b.with(reinterpret_cast<uint64_t&>(mbar), 0),
             thr_tma_b.partition_S(gB),
             thr_tma_b.partition_D(sB));
    }
    __syncthreads();
    mbar_wait(&mbar, 0);
    __syncthreads();

    // ---- WGMMA via cute::gemm ----
    // SM90_64x32x16_F32F16F16_SS is the SS (shared-shared) variant: both
    // operands come from SMEM. Major::MN means M is contig for A and N is
    // contig for B — matches our SW64 layouts.
    // WGMMA atom: A is K-major (K-contig in row-major mem), B is MN-major
    // (N-contig in row-major mem). The Major::K / Major::MN template params
    // must match the SmemLayout atoms above.
    using TiledMma = decltype(make_tiled_mma(
        SM90_64x32x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::MN>{}));
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // Partition SMEM views for the WGMMA. For the SS (shared-shared) atom the
    // SMEM views need to be wrapped through make_fragment_A/B which converts
    // them into GMMA::DescriptorIterator views — that's what cute::gemm
    // expects. Calling cute::gemm directly on partition_A(...) raw tensors
    // fails with "no instance of overloaded function cute::max_alignment
    // matches the argument list (GMMA::DescriptorIterator)".
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    // Output fragment: each thread holds its share of [M=64, N=32] in regs.
    Tensor tCrC = partition_fragment_C(tiled_mma,
                                        Shape<Int<M>, Int<N>>{});
    clear(tCrC);

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // ---- Store fragment to D_global ----
    Tensor gD = make_tensor(
        make_gmem_ptr(D_global),
        make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<N>{}, Int<1>{}));
    Tensor tCgD = thr_mma.partition_C(gD);
    cute::copy(tCrC, tCgD);
}

}  // namespace cute_oracle_impl

// =============================================================================
// PyTorch entry point. `phase` selects which CuTe-based test to run.
//   0 = smoke (just compile + launch a 1-thread kernel that touches an atom)
//   1a..1c = future phases (not yet implemented; will be added incrementally
//            as the previous phase's gate passes)
// =============================================================================

torch::Tensor cute_oracle_phase0() {
    auto i32_opts = torch::TensorOptions()
        .device(torch::kCUDA)
        .dtype(torch::kInt32);
    torch::Tensor sentinel = torch::zeros({10}, i32_opts);
    auto stream = c10::cuda::getCurrentCUDAStream();
    cute_oracle_impl::cute_smoke_kernel<<<1, 1, 0, stream>>>(
        sentinel.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return sentinel;
}

// Phase 1a: takes B[K=16, N=32] half, returns SMEM dump as fp16 [K*N].
// Caller compares multiset(dump) vs multiset(B); equal means TMA + SW64
// layout did a clean byte-level round-trip.
torch::Tensor cute_oracle_phase1a(const torch::Tensor& B) {
    using namespace cute;
    using namespace cute_oracle_impl;

    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat16);
    TORCH_CHECK(B.size(0) == 16 && B.size(1) == 32, "B must be [16, 32] fp16");
    TORCH_CHECK(B.is_contiguous());

    auto fp16_opts = torch::TensorOptions().device(B.device()).dtype(torch::kFloat16);
    torch::Tensor dump = torch::zeros({16, 32}, fp16_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Construct CuTe global tensor view of B. Memory layout is row-major
    // [K=16, N=32], so element B[k][n] sits at offset k*N + n. In CuTe (N, K)
    // order with N inner, that's stride (1, N).
    auto B_global = make_tensor(
        make_gmem_ptr(reinterpret_cast<half_t const*>(B.data_ptr())),
        make_shape(_32{}, _16{}),
        make_stride(_1{}, _32{}));

    using AtomB = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayB  = decltype(tile_to_shape(AtomB{}, Shape<_32, _16>{}));

    // Build the TMA copy atom — this is the pre-validated descriptor.
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, B_global, LayB{});

    cute_oracle_impl::cute_phase1a_kernel
        <<<1, 32, 0, stream>>>(
            tma_b,
            reinterpret_cast<half_t*>(dump.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return dump;
}

// Phase 1c: takes A[M=64, K=16] half + B[K=16, N=32] half, returns D fp32.
// Caller compares D vs torch.matmul(A.float(), B.float()).
torch::Tensor cute_oracle_phase1c(const torch::Tensor& A,
                                  const torch::Tensor& B) {
    using namespace cute;
    using namespace cute_oracle_impl;

    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kFloat16);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kFloat16);
    TORCH_CHECK(A.size(0) == 64 && A.size(1) == 16, "A must be [64, 16] fp16");
    TORCH_CHECK(B.size(0) == 16 && B.size(1) == 32, "B must be [16, 32] fp16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous());

    auto fp32_opts = torch::TensorOptions().device(A.device()).dtype(torch::kFloat32);
    torch::Tensor D = torch::zeros({64, 32}, fp32_opts);

    auto stream = c10::cuda::getCurrentCUDAStream();

    // Global views.
    // A is row-major [M=64, K=16]; CuTe shape (M, K) with K inner stride=1.
    auto A_global = make_tensor(
        make_gmem_ptr(reinterpret_cast<half_t const*>(A.data_ptr())),
        make_shape(_64{}, _16{}),
        make_stride(_16{}, _1{}));
    // B is row-major [K=16, N=32]; CuTe (N, K) with N inner stride=1.
    auto B_global = make_tensor(
        make_gmem_ptr(reinterpret_cast<half_t const*>(B.data_ptr())),
        make_shape(_32{}, _16{}),
        make_stride(_1{}, _32{}));

    // MUST match the kernel-side SmemLayout atoms exactly. A is K-major in
    // memory (row-major M×K with K contig), B is MN-major (row-major K×N
    // with N contig). See cute_phase1c_kernel for the full atom rationale.
    using AtomA = decltype(GMMA::Layout_K_SW32_Atom<half_t>{});
    using LayA  = decltype(tile_to_shape(AtomA{}, Shape<_64, _16>{}));
    using AtomB = decltype(GMMA::Layout_MN_SW64_Atom<half_t>{});
    using LayB  = decltype(tile_to_shape(AtomB{}, Shape<_32, _16>{}));

    auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, A_global, LayA{});
    auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, B_global, LayB{});

    cute_oracle_impl::cute_phase1c_kernel
        <<<1, 128, 0, stream>>>(
            tma_a, tma_b,
            D.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return D;
}

// Dispatcher to keep one PYBIND entry point.
std::vector<torch::Tensor> cute_oracle(int64_t phase,
                                        const torch::Tensor& A,
                                        const torch::Tensor& B) {
    if (phase == 0) {
        return {cute_oracle_phase0()};
    } else if (phase == 11) {        // phase 1a
        return {cute_oracle_phase1a(B)};
    } else if (phase == 13) {        // phase 1c
        return {cute_oracle_phase1c(A, B)};
    } else {
        TORCH_CHECK(false,
                    "Unsupported phase. Use 0/11/13.");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cute_oracle", &cute_oracle,
          "CuTe TMA->WGMMA oracle. phase ∈ {0=smoke, 11=phase1a, 13=phase1c}.",
          py::arg("phase"), py::arg("A"), py::arg("B"));
}
