# Megakernel Plan — Custom Fused RL-Spline-KV FFN

**Goal**: implement a single fused CUDA kernel per direction (forward & backward) that fuses the entire RL-Spline-KV FFN — `x → z = Kx → a, δ via spline → f = [a; λδ] → y = W_out·f` — into one kernel.

**Why**: empirical analysis shows the per-step gap to MLP (105 ms vs 46 ms per step on H100) is dominated by **launch overhead + framework overhead + intermediate tensor materialization**, not by the bare matmul/spline FLOPs (microbench says these are 3.6 ms / step). A megakernel eliminates all of these.

**Target**: total step time **≤ MLP** (≤ 46 ms / step on H100, ideally beat it). Stretch goal: 0.7× MLP via FP8 + TMA + asynchrony.

References used while planning:
- FlashAttention-3 paper + source ([Dao-AILab/flash-attention/hopper](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)) — canonical Hopper megakernel pattern
- ThunderKittens ([HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)) — producer/consumer + TMA + wgmma at >700 TFLOPS
- CUTLASS Hopper Collective MMA + Pipeline ([NVIDIA/cutlass](https://github.com/NVIDIA/cutlass))
- PTX 8.5 §9.7.14 (wgmma) + §9.7.8.24.9 (cp.async.bulk.tensor)
- v7 theory: `docs/THEORY_v7_RL_SPLINE_KV.md` — the architecture this kernel implements

---

## 0a. Theoretical context (what we are accelerating)

RL-Spline-KV is **not** a generic FFN. It is a **two-path FFN with a continuous key-value
memory** (v7 §R.1). For one token x ∈ ℝ^d:

```
z = K x                                              # h scalar "keys"
y = W_a · ReLU²(z)         (base path)               # standard MLP
  + λ · W_δ · δ(z, C)      (spline residual)         # local-value-memory
where  δ(z, C) = Σ_{j=1..h} Σ_{b ∈ A(z_j)}  B_b(z_j) · c_{j,b}     # c_{j,b} ∈ ℝ^r
```

The framing (v7 §R.1.3): the base path is the standard Geva-2021 "FFN as key-value memory"
where key `k_j` retrieves a single fixed value `W_a[:,j]`. The spline path **upgrades each key
to a continuous local value curve** `b ↦ c_{j,b}`, smoothly interpolated by the B-spline basis
`B_b`. Same key, retrieves *different* values depending on activation magnitude `z_j`.

What makes this distinct (v7 §R.1.5 / §R.5):
- Not FullMix-Tucker rebranded — that has a single output-rank bottleneck `R_o`. RL-KV writes
  `y` as **two separable paths sharing one z = Kx**, no bottleneck on the base path.
- Not DeepSpline / KAT — those learn a scalar activation `φ_j(z_j)`. We learn an **r-dim
  residual** `Σ_b B_b(z_j) · c_{j,b}` per key.
- Not PKM (Lample 2019) — that uses sparse top-k retrieval over a flat memory. We are
  **dense over keys, sparse over bins** (every key contributes via p+1 active basis indices).

The kernel-relevant invariants we have to preserve in the megakernel:

| Invariant | Source | Implication for kernel |
|---|---|---|
| `B-spline partition of unity`: Σ_b B_b(z) = 1 in range | v7 §R.3.0 | basis weights must sum to 1 in fp32 accumulator |
| `B2 is C¹ continuous`, `B1 is C⁰` | v7 §R.3.0 | spline derivative `dB_b/dτ` needed in backward path |
| `out-of-range mask`: B_b = 0 when z ∉ [grid_lo, grid_hi] | v7 §R.3.0 | clamp + mask, both in fwd basis and bwd derivative |
| `cold-start init`: C = 0 ⇒ δ = 0 | v7 §R.5 | numerical: gradient through `c_{j,b}` must flow even when C=0 |
| `paper claim threshold`: `ρ_δ = ‖W_δ·δ‖ / ‖W_a·a‖ ≥ 0.20` in ≥ 8/12 layers | v7 §R.1.4 | not a kernel correctness issue but explains why both paths matter — kernel must keep δ accurate enough that ρ_δ doesn't get masked by quantization noise |

The active-basis count is the spline degree's `p+1`:
- B1 (linear): 2 active basis per z_j (`bin, bin+1`), `L = G + 1`
- B2 (quadratic, default): 3 active basis (`bin, bin+1, bin+2`), `L = G + 2`

The megakernel below targets **B2 + r=32** as primary (matches our shipping cell
`rl_kv_B2_r32_L22`). B1 path can be added as a constexpr template parameter later.

---

## 0. Architecture refresher (per layer, per token)

**Forward** (one token x ∈ ℝ^d, d=h, with h_ratio=1):
```
z   = K x                                        # K: ℝ^{h×d}
a   = ReLU²(z)                                   # ∈ ℝ^h
bin, τ, B0..B2 = b2_basis(z)                    # per-channel
δ   = Σ_j Σ_{k∈{0,1,2}} B_k(z_j) · C[j, bin_j+k, :]   # ∈ ℝ^r
f   = [a;  λ·δ]                                  # ∈ ℝ^{h+r}
y   = W_out · f                                  # W_out: ℝ^{d×(h+r)}
```

**Backward** (given g_y = ∂L/∂y):
```
g_f       = W_out^T · g_y                                 # ∈ ℝ^{h+r}
g_a       = g_f[:h]
g_λδ      = g_f[h:]
g_δ       = λ · g_λδ
g_W_out   = g_y · f^T                                     # outer product, atomic-add
phi'(z_j) = 2·z_j · 1[z_j>0]                              # ReLU² derivative
g_z_a     = g_a · phi'(z)
g_z_spline= scale · Σ_k dB_k(τ) · ⟨g_δ, C[j, bin+k, :]⟩    # via dot product over r
g_z       = g_z_a + g_z_spline
g_K       = g_z · x^T                                     # outer product, atomic-add
g_C[j,b,:]+= B_b(z_j) · g_δ                                # scatter, atomic-add
g_x       = K^T · g_z
```

Per layer, per step (B*T = 2048 tokens, d=h=768, r=32, L=22):
- 6 GEMMs (z=Kx, y=W_out·f, g_f=W_out^T·g_y, g_K=g_z·x^T, g_W_out=g_y·f^T, g_x=K^T·g_z) — these are the FLOPs
- spline forward: per token sum of 3 vectors of length r (small)
- spline backward dC: scatter atomic-add (~6h*L*r = ~400K atomics per layer per batch)

---

## 1. Why a megakernel wins (root-cause analysis)

Current path (B2 wgmmaCUDA: 209 ms/step):

| Stage | Source | Observed cost |
|---|---|---|
| K linear (cuBLAS) | per-call ~30 μs × 6 layers = 180 μs | small |
| FlashSplineFeature.apply forward | autograd Function dispatch ~50 μs + Triton fwd kernel ~150 μs × 6 = 1.2 ms | medium |
| W_out linear (cuBLAS) | ~30 μs × 6 = 180 μs | small |
| FlashSplineFeature.apply backward | autograd recall + Triton bwd / our CUDA bwd | medium |
| **Framework overhead** (autograd graph, save_for_backward, dispatch) | **~50 μs × 6 layers × 2 (fwd+bwd) = 600 μs** | **medium** |
| **Kernel launch overhead** (~10 μs × ~30 launches/step) | **~300 μs** | **medium** |
| **Memory bandwidth** (z, a, δ, f tensors materialized in HBM) | f tensor [2048, 800] bf16 = 3.2 MB read+write × 6 layers × 2 = 38 MB / step / layer | **medium-large** |
| **Optimizer + remaining nanochat blocks** | ~30 ms (same as MLP) | — |

The remaining ~150 ms/step is unaccounted; we believe it is **wave quantization + wave shutdown latency** — many small kernels each take a wave then idle SMs while the next launches.

A megakernel:
1. **Eliminates dispatch / autograd graph overhead** → save ~600 μs / step
2. **Eliminates intra-layer HBM round-trips** for z, a, δ, f → save ~10-20 ms / step
3. **Allows producer/consumer overlap** of TMA + wgmma → save ~10-20 ms / step
4. **Enables CUDA Graph capture trivially** since the layer is one launch → save another ~10-30 ms / step from launch amortization

Realistic target: **40-60 ms / step**, i.e., 0.85-1.3× MLP.

---

## 2. Tile / shape decisions

For B*T = 2048 tokens, d = 768, h = 768, r = 32, L = 22:

- **Per-block work unit**: M_TILE = **64** tokens (one wgmma m=64 row group; zero waste).
  See §10.1 for the full SM-saturation analysis (24% util at B=2 T=1024 — accept; we
  are launch-bound, not SM-bound at this batch).
- **Grid**: persistent, 132 blocks (one CTA per H100 SM), grid-stride static work
  assignment (`work_idx = sm_id + k·132`). M-tiles total = ⌈2048/64⌉ = 32; well under
  one wave.

- **K dim of GEMMs**: d = 768. Each wgmma k=16 → 48 k-iterations for the z = Kx GEMM.
- **N dim**:
  - z = K x: M=64, K=d=768, N=H_BLOCK=8 per inner step (we sweep h as h_blocks of 8 to
    keep the z fragment register footprint inside the 160-reg consumer budget; see §10.9).
  - y = W_out f: M=64, K=h+r=800, N=N_BLOCK=64; outer-loop over N_TILES=12 (§10.2).

- **SMEM budget on H100**: 228 KB / SM dynamic, 167 KB used. Per §10.6/10.7:
  - x stage buffer: 3 stages × [M_TILE=64, K_BLOCK=64] bf16 = 24 KB
  - K weight stage buffer: 3 stages × [K_BLOCK=64, H_BLOCK=8] bf16 = 3 KB
  - C stage buffer: 2 stages × [H_BLOCK=8, L_PAD=24, R=32] bf16 = 24 KB
  - W_out stage buffer: 2 stages × [N_BLOCK=64, K_BLOCK_WOUT=32] bf16 = 8 KB
  - z handoff (consumer-1 → consumer-2): 2-ring × [M_TILE=64, H_BLOCK=8] fp32 = 4 KB
  - **f_smem persistent** (consumer-2 → consumer-3): [M_TILE=64, h+r=800] bf16 = **100 KB**
  - y SMEM tile (consumer-3 → producer TMA-store): 2-ring × [M_TILE=64, N_BLOCK=64] bf16 = 16 KB
  - mbarriers + scratch: ~4 KB
  - **Total**: ~183 KB ≤ 228 KB ✓ (with margin for alignment padding).

> **Note**: an earlier draft staged `a` through a 32 KB ring buffer and kept M_TILE=128
> with 232 regs/consumer. Both decisions are obsolete: M_TILE=64 + persistent f_smem +
> NCWG=3 (160 regs/consumer) is the current spec; do not implement from §2 paragraphs
> that may still hint at the old shape.

---

## 3. Pipeline plan

We follow the **FA3 / TK pattern**: warp specialization with TMA producer + wgmma consumers + mbarrier-based pipeline. PTX 8.5 §9.7.14.5 mandates `wgmma.fence + fence.proxy.async + commit_group + wait_group` per epoch.

### 3.1 Warpgroup roles (forward kernel)

H100 register file = **65536 regs/SM**. With `launch_bounds(512, 1)` (1 CTA/SM), per-CTA
budget is 65536 regs ÷ 512 threads = **128 regs/thread average**, with `setmaxnreg`
allowing skewed splits.

ThunderKittens formula (`group.cuh`): `consumer_registers<NCWG>() = 480/NCWG -
8·(NCWG>3) - 224·(NCWG==1)`. Consumer counts:
- NCWG=2 → **240 regs/consumer thread** (1 producer + 2 consumers)
- NCWG=3 → **160 regs/consumer thread** (1 producer + 3 consumers)

Sanity check totals (per CTA, all 512 threads):
- NCWG=2: 24×128 + 240×128×2 = 64512 ≤ 65536 ✓
- NCWG=3: 24×128 + 160×128×3 = 64512 ≤ 65536 ✓

**Decision: NCWG=3 with 160 regs/consumer thread.** The three roles map cleanly to z-gemm,
spline, y-gemm specialization, and the 160-reg budget is enough for our largest accumulator
fragment (16 fp32/thread = 16 regs for one m64n64 wgmma cell).

Block has 4 warpgroups (16 warps, 512 threads). Roles:

| WG | Threads | Role | `setmaxnreg` |
|---|---|---|---|
| WG0 | 0..127 | **Producer**: issue TMA loads for x, K, C, W_out tiles | `dec 24` |
| WG1 | 128..255 | **Consumer-1**: z = x @ K^T (wgmma m64n8k16) → z accumulator in regs | `inc 160` |
| WG2 | 256..383 | **Consumer-2**: spline forward — bin/τ/B; gather C; accumulate δ in registers | `inc 160` |
| WG3 | 384..511 | **Consumer-3**: y = f @ W_out^T (wgmma m64n64k16) → TMA-store y per n_tile | `inc 160` |

**Total registers: 24·128 + 160·128·3 = 64512 ≤ 65536 ✓**

If consumer-2 register pressure spills (NCU `smsp__sass_local_load.sum` > 0), fall back
to NCWG=2 (merge consumer-1 and consumer-2 into one WG sharing z register state directly,
no SMEM handoff needed; consumer-3 takes the full 240-reg consumer budget for the y GEMM).

### 3.2 Pipeline depth & buffers

(Authoritative SMEM table is in §2; this subsection only covers the per-h_block control flow.)

**Why we sweep h in H_BLOCK=8 chunks**: with M_TILE=64 and the 160-reg consumer budget,
keeping the full z[:, h] = [64, 768] fp32 in registers is not feasible. Instead each
"h_block" iteration handles 8 of the 768 h channels, allowing the z fragment to fit in
~16 fp32/thread (one m64n8 wgmma cell).

Each iteration of the outer `h_blk = 0..h/H_BLOCK` loop:
1. Producer issues TMA loads for K[:, h_blk*8:(h_blk+1)*8] and C[h_blk*8:(h_blk+1)*8, :, :].
2. Consumer-1 computes z[:, h_blk*8:(h_blk+1)*8] = x · K[:, …]^T via wgmma m64n8k16,
   sweeping the K dim in K_BLOCK=64 stages.
3. Consumer-1 stores its z fragment to the z handoff SMEM ring (§10.6); arrives on
   `z_full[stage]`.
4. Consumer-2 waits on `z_full[stage]`, reads z, computes B/τ/bin, gathers C[h_blk, :, :],
   accumulates λ·δ into a register fragment, and writes
   `a[:, h_blk*8:(h_blk+1)*8] = ReLU²(z)` into f_smem (§10.7).
5. After all h_blocks done, consumer-2 writes the final λ·δ slice into f_smem at offset h
   and arrives on `f_ready`.
6. Consumer-3 waits on `f_ready` once, then runs y = f @ W_out^T as N_TILES=12 outer
   iterations (§10.2) reading from the persistent f_smem.

### 3.3 Synchronization primitives

- **`mbarrier`** in SMEM, one per (buffer × stage). Producer arrives `expect_bytes` after TMA issue; consumer waits with `try_wait` / `wait_parity`.
- **`fence.proxy.async.shared::cta`** between any generic-proxy SMEM write and any wgmma read (PTX 8.5 §9.7.14.5 step 2). Already a known requirement from our earlier wgmma fix.
- **`wgmma.fence.sync.aligned`** between accumulator init and first wgmma op.
- **Named barriers** (`bar.sync 1, NumThreads`) for cross-WG sync points where mbarrier is awkward (e.g. consumer-1 → consumer-2 handoff of z register state).
- **`cluster_barrier` / `cluster_arrive`** if we use cluster mode (size 2 cluster for cross-CTA TMA). Skip in v1.

### 3.4 Forward pseudocode

```cpp
template <int M_TILE=64, int K_BLOCK=64, int H_BLOCK=8, int R=32, int L=22, int STAGES=3>
__global__ void __launch_bounds__(512, 1)
rl_kv_fwd_megakernel(
    TensorMap x_map,        // [B*T, d]
    TensorMap K_map,        // [d, h]
    TensorMap C_map,        // [h, L, r]
    TensorMap Wout_map,     // [d, h+r]
    bf16* y_global,         // [B*T, d]
    int N_tokens, int d, int h, int r, int L,
    float grid_lo, float grid_hi, int G, float lambda_scale)
{
    extern __shared__ char smem_raw[];
    auto& smem = *reinterpret_cast<SmemLayout*>(smem_raw);
    int wg_id = (threadIdx.x / 128);  // 0=producer, 1..3=consumers
    int wg_lane = threadIdx.x % 128;

    if (wg_id == 0) {
        // PRODUCER
        warpgroup::decrease_registers<24>();
        if (cute::elect_one_sync()) {
            for (int h_blk = 0; h_blk < h / H_BLOCK; ++h_blk) {
                for (int k_blk = 0; k_blk < d / K_BLOCK; ++k_blk) {
                    auto stage = (h_blk * (d/K_BLOCK) + k_blk) % STAGES;
                    // Wait for empty buffer
                    smem.full_K[stage].wait_for_empty(...);
                    // Issue TMA load: x[block_start:block_start+M_TILE, k_blk*K_BLOCK:+K_BLOCK]
                    cp_async_bulk_tensor_2d(smem.x_buf[stage], x_map,
                                             {block_token_start, k_blk*K_BLOCK},
                                             smem.full_K[stage] /* mbarrier */);
                    // Issue TMA load: K[k_blk*K_BLOCK:+K_BLOCK, h_blk*H_BLOCK:+H_BLOCK]
                    cp_async_bulk_tensor_2d(smem.K_buf[stage], K_map,
                                             {k_blk*K_BLOCK, h_blk*H_BLOCK},
                                             smem.full_K[stage]);
                }
                // Issue TMA load for C[h_blk*H_BLOCK:+H_BLOCK, :, :]
                cp_async_bulk_tensor_3d(smem.C_buf[h_blk%2], C_map,
                                         {h_blk*H_BLOCK, 0, 0},
                                         smem.full_C[h_blk%2]);
            }
            // Then load W_out tiles (similar)
            for (int n_blk = 0; n_blk < d / N_BLOCK; ++n_blk) {
                cp_async_bulk_tensor_2d(smem.Wout_buf[n_blk%2], Wout_map,
                                         {n_blk*N_BLOCK, 0},
                                         smem.full_Wout[n_blk%2]);
            }
        }
    } else {
        // CONSUMERS (NCWG=3 → 160 regs/thread per ThunderKittens formula; see §3.1)
        warpgroup::increase_registers<160>();
        float z_acc[Z_FRAG_SIZE];
        float delta_acc[DELTA_FRAG_SIZE];
        // ... per-WG specialized code
        if (wg_id == 1) {
            // Consumer-1: GEMM z = x @ K^T over h_blocks
            for (int h_blk = 0; h_blk < h/H_BLOCK; ++h_blk) {
                // Reset z accumulator for this h_block
                fill_zero(z_acc);
                for (int k_blk = 0; k_blk < d/K_BLOCK; ++k_blk) {
                    auto stage = (h_blk * (d/K_BLOCK) + k_blk) % STAGES;
                    smem.full_K[stage].wait_for_full(...);
                    fence_proxy_async_shared_cta();
                    wgmma_fence();
                    wgmma_m64n64k16(z_acc, smem.x_buf[stage], smem.K_buf[stage]);
                    wgmma_commit_group();
                    wgmma_wait_group<0>();
                    smem.full_K[stage].arrive_empty();
                }
                // Hand z_acc off to consumer-2 via SMEM (small, just M_TILE × H_BLOCK fp32)
                store_z_to_smem(z_acc, smem.z_buf[h_blk]);
                NamedBarrier::arrive(2 * 128, named_barrier::Z_READY);
            }
        } else if (wg_id == 2) {
            // Consumer-2: spline forward
            for (int h_blk = 0; h_blk < h/H_BLOCK; ++h_blk) {
                NamedBarrier::wait(2 * 128, named_barrier::Z_READY);
                load_z_from_smem(z_local, smem.z_buf[h_blk]);
                smem.full_C[h_blk%2].wait_for_full(...);
                fence_proxy_async_shared_cta();
                spline_accumulate(delta_acc, z_local, smem.C_buf[h_blk%2]);
                smem.full_C[h_blk%2].arrive_empty();
                // Compute a = ReLU²(z), write to SMEM for consumer-3
                relu_sq_to_smem(z_local, smem.a_buf[h_blk]);
            }
            // Pack δ and a into f-buffer
            store_delta_to_smem(delta_acc, smem.delta_buf);
            NamedBarrier::arrive(2 * 128, named_barrier::F_READY);
        } else {
            // Consumer-3: y = f @ W_out^T (wgmma over h+r dim)
            NamedBarrier::wait(2 * 128, named_barrier::F_READY);
            float y_acc[Y_FRAG_SIZE] = {0};
            for (int n_blk = 0; n_blk < d / N_BLOCK; ++n_blk) {
                smem.full_Wout[n_blk%2].wait_for_full(...);
                fence_proxy_async_shared_cta();
                wgmma_fence();
                wgmma_m64n64k16(y_acc, smem.f_buf, smem.Wout_buf[n_blk%2]);
                wgmma_commit_group();
                wgmma_wait_group<0>();
                smem.full_Wout[n_blk%2].arrive_empty();
            }
            // Store y to global (one-writer, no atomic needed)
            store_y_to_global(y_acc, y_global, block_token_start);
        }
    }
}
```

### 3.5 Backward — see §Phase 4 for the 3-kernel split

**Earlier drafts** described a single fused backward megakernel mirroring forward,
with atomic-add into g_K / g_W_out / g_C. That approach is **abandoned** — see Phase 4
for the current spec.

**Current design** (Phase 4): three separate backward kernels, each grid-by-output-tile
so every parameter-gradient write is uncontested (zero atomics, zero workspace):
- **`bwd_y`**: g_f = W_out^T · g_y (per-token); g_W_out = g_y^T · f (grid by `[d, h+r]` output tile).
- **`bwd_spline`**: g_C scatter (grid by `[h_block, L]` output tile, atomics local to SMEM only); g_z_spline (per-token).
- **`bwd_K`**: g_z = g_a · phi'(z) + g_z_spline; g_x = g_z · K (per-token); g_K = g_z^T · x (grid by `[d, h]` output tile).

**Saved state for backward**: see §10.3 — `bin_idx (uint8) + τ (bf16) + z_bf16 + in_range_mask (1 bit)`
per element, NOT just z_bf16 (bin-boundary round-trip risk).

Each of the three kernels reuses the forward kernel's WG specialization machinery
(producer + 3 consumers, mbarrier-coordinated) but with smaller scope per kernel.
3-kernel total launch overhead is ~30 μs vs the ~5–10 ms cost of atomic-add contention
that a fused single-kernel design would incur.

### 3.6 SMEM layout (forward) — see §10.6 / §10.7 for the authoritative spec

The persistent `f_smem[M_TILE=64][h+r=800]` (100 KB) replaces the earlier "stream a
through ring buffer" sketch — the ring approach forced consumer-3 to recompute the
spline coefficients ~12× (once per N_TILE), which is unacceptable.

```cpp
struct SmemLayout {
    // Producer-fed staging
    alignas(128) bf16 x_stage   [3][M_TILE  ][K_BLOCK  ];        // 24 KB
    alignas(128) bf16 K_stage   [3][K_BLOCK ][H_BLOCK  ];        //  3 KB
    alignas(128) bf16 C_stage   [2][H_BLOCK ][L_PAD    ][R];     // 24 KB
    alignas(128) bf16 Wout_stage[2][N_BLOCK ][K_BLOCK_WOUT];     //  8 KB

    // Inter-consumer handoff
    alignas(128) fp32 z_handoff [2][M_TILE  ][H_BLOCK  ];        //  4 KB  — consumer-1 → consumer-2
    alignas(128) bf16 f_smem       [M_TILE  ][H_PLUS_R ];        // 100 KB — consumer-2 → consumer-3 (PERSISTENT, not ringed)
    alignas(128) bf16 y_smem    [2][M_TILE  ][N_BLOCK  ];        // 16 KB — consumer-3 → producer TMA-store

    // mbarriers (see §10.5b for full table)
    alignas(8) uint64_t x_full[3], x_empty[3];
    alignas(8) uint64_t K_full[3], K_empty[3];
    alignas(8) uint64_t C_full[2], C_empty[2];
    alignas(8) uint64_t Wout_full[2], Wout_empty[2];
    alignas(8) uint64_t z_full[2], z_empty[2];
    alignas(8) uint64_t f_ready;
    alignas(8) uint64_t y_full[2], y_done[2];
};
// Total ≈ 183 KB ≤ 228 KB H100 SM dynamic SMEM ✓
```

**Decision**: keep `f_smem` persistent for the whole y = f @ W_out^T loop; do NOT ring-buffer
it. Confirmed in §10.7.

---

## 4. Implementation phases (de-risked, incremental)

Build incrementally with a hard verification gate at each step — the goal is to catch regressions early rather than ship a single big PR that silently misbehaves.

### Phase 0 — scaffold
- Set up `spline_kv_fwd_mega.cu` with empty kernel, CUTLASS includes, TMA descriptor builders.
- Write Python autograd Function `RLSplineKVMega` that wraps both fwd & bwd CUDA kernels.
- Write a unit test that compares fwd output to PyTorch reference on a tiny shape (B=2, T=64).
- **Pass criterion**: kernel launches, returns zeros (no crash).

### Phase 1 — single-WG forward, no producer
- Implement forward as a single warpgroup that does z = x@K, then spline, then y = f@W_out, all synchronous (no TMA, just cp.async, no warp specialization).
- Use existing wgmma kernel as starting point; extend to chained GEMMs.
- **Pass criterion**: rel_err ≤ 1.7e-3 vs PyTorch reference for B=8 T=1024 d=768 h=768 r=32.

### Phase 2 — add TMA loads
- Replace cp.async with cp.async.bulk.tensor for x, K, C, W_out.
- Build host-side `cuTensorMapEncodeTiled` for each tensor (4 descriptors total).
- Use `mbarrier::expect_tx` / `mbarrier::try_wait` for completion.
- **Pass criterion**: same rel_err, wall ≥ 1.5× faster than Phase 1 in isolation.

### Phase 3 — warp specialization + pipeline
- Split into 4 WGs as planned in §3.1.
- Use `setmaxnreg` (producer `dec 24`, all 3 consumers `inc 160` per §3.1 NCWG=3 split).
- Pipeline depth STAGES=3 for x/K, STAGES=2 for C/Wout.
- Validate: rel_err same, no deadlock, no race.
- Use NCU to verify SM occupancy ≥ 50%, no instruction stalls.
- **Pass criterion**: forward kernel achieves ≥ 200 TFLOPS effective (vs 989 TFLOPS H100 peak BF16).

### Phase 4 — backward as 3 separate kernels (no atomics, no workspace)

**Why separate kernels, not one mega-bwd**: backward has three distinct output shapes
(`[d, h+r]` for g_W_out, `[h, L, r]` for g_C, `[d, h]` for g_K) plus per-token outputs
(g_x, g_z_spline). One kernel cannot make all three "data-parallel along output" at the
same time without atomics on at least one. Three separate kernels each pick the optimal
grid for their own output, all atomic-free.

**Phase 4.1** — kernel **bwd_y**: backward through `y = f @ W_out`:
- Inputs: `g_y [N, d]`, `f` (recomputed in-kernel from saved z + C, OR loaded from scratch
  if forward saved it — see decision below)
- Outputs: `g_f [N, h+r]` (per-token, scratch global, no atomics);
  `g_W_out [d, h+r]` (parameter, grid by output tile, no atomics)
- Grid choice: **two grids in one kernel via cooperative groups**, OR **two cooperating
  blocks** — the simpler path is launch a separate sub-kernel for g_W_out.
  - Sub-kernel A1: grid `(N_tiles, M_tiles)` for `g_f = g_y @ W_out`, per-token output.
  - Sub-kernel A2: grid `(d/64, (h+r)/64)` for `g_W_out = g_y^T @ f` accumulating over N.
  - Or unified: grid by `g_W_out` output tile, each block re-reads its slab of g_y/f
    (simpler, slight redundant reads).
- Verify: rel_err of `g_f` and `g_W_out` ≤ 2e-3 vs PyTorch autograd reference.

**Phase 4.2** — kernel **bwd_spline**: spline backward (g_C + g_z_spline):
- Inputs: saved `bin_idx + τ` (see §10.3 z_saved revision), `C`, `g_f [N, h+r]` (from 4.1),
  `λ`
- Outputs: `g_C [h, L, r]` (per-h_block grid, no atomics); `g_z_spline [N, h]` (per-token
  scratch, no atomics)
- Grid: `(h_block_tiles=12, b_tiles=L)` so each block owns a unique slice of g_C. Token
  axis is reduced inside each block by atomic-add into block-local SMEM accumulator.
- Verify: `g_C` rel_err ≤ 2e-3.

**Phase 4.3** — kernel **bwd_K**: backward through `z = K x` (g_K + g_x):
- Inputs: `x [N, d]`, `K [d, h]`, `g_a = g_f[:, :h]` (from 4.1), `g_z_spline [N, h]` (from
  4.2), saved z (for `phi'`)
- Compute: `g_z = g_a · phi'(z) + g_z_spline` (in-kernel pointwise)
- Outputs: `g_x = g_z @ K [N, d]` (per-token, no atomics);
  `g_K = g_z^T @ x [d, h]` (parameter, grid by output tile, no atomics)
- Verify: `g_K, g_x` rel_err ≤ 2e-3.

**Final pass criterion**: end-to-end gradient rel_err ≤ 2e-3 vs PyTorch reference;
total backward wall ≤ 1.2× MLP backward wall (Phase 6 CUDA Graphs amortizes the 3
kernel launches into 1 graph submit).

**Tradeoff vs single mega-bwd**: 3 launches add ~30 μs overhead vs unified, but
eliminate the atomic-add cost which would otherwise be ~5-10 ms. Net win.

**Future stitching (post-v1, only if 30 μs becomes the bottleneck)**: merge 4.2 + 4.3
into one kernel since both consume g_f-derived inputs and produce per-token outputs;
keep 4.1 separate because its g_W_out output dim (`d × (h+r)`) is incompatible with the
others' grid layouts. Skip in v1.

### Phase 5 — wire into RL-Spline-KV
- Add `bwd_kernel="megakernel"` option in autograd Function.
- Wire into RLKVAdapter / RLSplineKVConfig.
- Run smoke test (500 steps) on H100 modal.

### Phase 6 — CUDA Graphs + persistent kernel
- Wrap step in `torch.cuda.CUDAGraph`.
- Optionally make megakernel persistent (SM-resident, work-stealing).
- **Pass criterion**: total step time ≤ MLP × 1.0.

### Phase 7 — FP8 (stretch, optional)
- Convert C, K, W_out to fp8 (e4m3) with per-tensor scaling.
- Use wgmma fp8 variant (m64n*k32 with .e4m3.e4m3).
- 2× peak FLOPS over BF16 → potentially 0.5× MLP wall.
- Only attempt after Phase 6 hits the wall target on BF16.

---

## 5. Risk register & fallback

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TMA descriptor encoding wrong | high (1st time) | high — silent garbage | First test with cp.async (Phase 1); switch to TMA in Phase 2 with diff against reference at each step |
| Producer-consumer deadlock | medium | high — kernel hangs | Single-WG Phase 1 first; add WG split incrementally with NCU profiling |
| SMEM budget overshoot | medium | hard — kernel won't launch | Compute SMEM at compile time with `static_assert`; have FP8 fallback to halve sizes |
| Atomic-add contention on g_K / g_W_out | medium | medium — slow | Block the M dimension across SMs; use one block per (M_tile, k_chunk) so atomics don't collide |
| Numerics in FP8 (Phase 7) | high | medium | Skip Phase 7 if BF16 already beats MLP |
| ncu overhead / unable to profile | low | low — debug-only | Use `cuda::cooperative_groups` profile counters as fallback |

**Hard fallback**: if Phase 4 (backward kernel) fails, ship Phase 0-3 (forward megakernel only) + keep Triton v3 backward. We'd still get ~30% speedup from the forward fusion alone.

---

## 6. Verification strategy

1. **Numerical**: at every phase, compare against `flash_spline_feature_reference()` in fp64 mode at tiny scale. Tolerance: `rel_dC, rel_dz ≤ 2e-3` (bf16 wgmma precision).
2. **Determinism**: same input → same output bit-exact (within fp32 acc round-off).
3. **Smoke training**: 500-step nanochat smoke; compare loss curve to existing B2 wgmmaCUDA. Should match to within 0.005 val_loss at step 500.
4. **Profiling**: NCU SM occupancy ≥ 50%, memory throughput ≥ 80% peak HBM, wgmma issue rate ≥ 75% peak.
5. **Wall time**: measured against MLP at the same B, T, d settings.

---

## 7. Open questions (decide before starting)

1. **Forward-backward fusion**: split into two kernels (one fwd, one bwd) or one big kernel? Two kernels is simpler and lets us reuse fwd alone if bwd fails. **Decision: two kernels.**
2. **Save z in forward, or recompute in backward?** Saving = 12 MB extra mem; recomputing = extra 1.6 GFLOPS. **Decision: save z (memory cheap, recompute unnecessary).**
3. **Cluster mode (size 2)**: enables cross-CTA TMA multicast. Not needed for our shape; defer.
4. **FP8 on what**: wait until BF16 megakernel is correct + fast, then add FP8 as optional path. C tensor is the biggest candidate (h*L*r = 540K params per layer at bf16 = 1 MB/layer → fp8 = 0.5 MB).
5. **All-12-layers**: with kernel cost lowered, all-12 RL-KV becomes feasible (12 megakernel calls instead of 6). Run capacity ladder afterwards.

---

## 8. Success criteria

- ✅ **Numerical**: bit-equivalent (within bf16 precision) to reference path at every layer.
- ✅ **Wall**: total step ≤ MLP × 1.0 on H100 (i.e., ≤ 46 ms / step at our reference shape).
- ✅ **VRAM**: peak ≤ MLP × 1.0 (i.e., ≤ 5006 MB).
- ✅ **Quality**: 500-step val_loss ≤ B2 wgmmaCUDA's (≤ 6.870, currently 6.867).
- 🚀 **Stretch (FP8)**: total step ≤ MLP × 0.7 (≤ 32 ms / step).

---

## 9. References (consulted while planning)

- FlashAttention-3, Shah et al., NeurIPS 2024. [arxiv:2407.08608](https://arxiv.org/abs/2407.08608) — uses 2-stage pipeline on H100, 75% peak FLOPS.
- ThunderKittens — `producer_registers() = 24`, `consumer_registers() = 480/NCWG - …`, persistent grid.
- CUTLASS Hopper Collective MMA — `cutlass::pipeline::PipelineTmaAsync` for TMA + mbarrier coordination.
- PTX 8.5 §9.7.14.5 — wgmma SMEM matrix descriptor format, `wgmma.fence + fence.proxy.async` requirement.
- PTX 8.5 §9.7.8.24.9 — cp.async.bulk.tensor syntax for TMA.
- our own `spline_kv_bwd_wgmma.cu` — already-correct wgmma + core-major SMEM (rel_dC=1.68e-3) — basis for megakernel.

---

## 10. Detailed implementation specs (post-self-review)

This section closes every critical / important gap from the self-review. Each subsection
is **prescriptive** — when implementing, follow these specs verbatim.

### 10.1 Persistent kernel + tile scheduler

**Problem**: 16 blocks × 1 SM each leaves 116/132 SMs idle (12% utilization).

**Fix** (CUTLASS SM90 persistent scheduler pattern — see `cutlass/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp`):

```cpp
constexpr int M_TILE  = 64;     // wgmma m=64 native
constexpr int NUM_SMS = 132;    // H100 SXM5
constexpr int GRID    = NUM_SMS;  // launch exactly 132 blocks (one per SM)
// Total work units = ceil(B*T / M_TILE).
//   B=2, T=1024: 32 work units → 32/132 = 24% SM utilization.
//   B=4, T=1024: 64 work units → 48% utilization.
//   B=2, T=1024, all-12-layers: layers run sequentially, but each layer has 32 units,
//     so over a full step we use 24% of SM-time × 12 = 288% SM-cycles, i.e. each SM
//     gets ~2.2 layers of work in a row. Fine.
// For B*T ≥ 132·M_TILE = 8448 (e.g. B=8 T=1024): full saturation.

__global__ void __launch_bounds__(512, 1) rl_kv_fwd_megakernel(
    const __grid_constant__ CUtensorMap tm_x, ...)
{
    // SPMD entry: all 512 threads enter together. WGs split below.
    const int sm_id      = blockIdx.x;            // 0..131
    const int total_work = (N + M_TILE - 1) / M_TILE;
    const int wg_id      = threadIdx.x / 128;     // 0=producer, 1..3=consumers
    const int wg_lane    = threadIdx.x % 128;

    // Per-WG register reshape (PTX 8.5 §9.7.13.6 setmaxnreg):
    // NCWG=3 → 160 regs/consumer; total 24*128 + 160*128*3 = 64512 ≤ 65536/SM. See §3.1.
    if (wg_id == 0) asm("setmaxnreg.dec.sync.aligned.u32 24;");
    else            asm("setmaxnreg.inc.sync.aligned.u32 160;");

    // Init mbarriers in SMEM (one thread per CTA, see §10.G)
    if (threadIdx.x == 0) {
        init_all_mbarriers();
    }
    asm("barrier.sync.aligned 0, 512;");  // wait for init before use

    // Grid-stride work loop. All 4 WGs cooperate on each work_idx.
    for (int work_idx = sm_id; work_idx < total_work; work_idx += GRID) {
        const int token_start = work_idx * M_TILE;
        const int token_count = min(M_TILE, N - token_start);  // tail-tile guard

        if (wg_id == 0) {
            run_producer(tm_x, tm_K, tm_C, tm_Wout, token_start, token_count, ...);
        } else if (wg_id == 1) {
            run_consumer_z_gemm(token_start, token_count, ...);          // z = x @ K
        } else if (wg_id == 2) {
            run_consumer_spline(token_start, token_count, ...);          // δ + a + f assembly
        } else {
            run_consumer_y_gemm(tm_y, tm_z, token_start, token_count, ...); // y = f @ W_out
        }

        // End-of-tile sync: all WGs must finish before next iteration to avoid race
        // on shared mbarriers / SMEM buffers.
        asm("barrier.sync.aligned 0, 512;");
    }
}
```

**SM saturation analysis** (must read before assuming megakernel beats MLP at small B):

| Workload (B, T) | work units = ⌈B·T/64⌉ | SM utilization | bottleneck likely |
|---|---|---|---|
| B=2, T=1024 | 32 | **24%** | producer/launch overhead masks underutil; megakernel still beats current |
| B=4, T=1024 | 64 | 48% | producer parallelism becomes the limit |
| B=8, T=1024 | 128 | 97% | full saturation; megakernel optimal |
| B=2, T=2048 | 64 | 48% | same as B=4, T=1024 |

**Honest assessment**: at our reference shape (B=2, T=1024), the megakernel runs only 32
of 132 SMs. It still wins because **the gap to MLP is launch overhead + autograd graph,
not arithmetic**. Once those are eliminated (single launch + CUDA Graph), 24% saturation
of a 989 TFLOPS peak still gives ~240 TFLOPS effective for the spline path — far above
what our current 100+ ms/step achieves.

**If we need full saturation at B=2 T=1024**: split each work unit across N_TILE too.
- New grid: `(M_tiles=32, N_tiles_grid=12)` = 384 blocks, ~3 waves on 132 SMs.
- Trade-off: each block computes 1 (M_TILE, N_TILE) cell of y, but **z + δ are needed
  by all 12 N_TILEs of one M_TILE**. Either:
  - (i) recompute z + δ in each block (12× work) — bad
  - (ii) use a 2-kernel scheme: kernel A produces z + a + δ to scratch global,
    kernel B reads scratch and runs y = f @ W_out per (M, N) cell — **breaks fusion**
- **Decision for v1**: accept 24% saturation at small batch. Fusion + CUDA Graph wins
  outweigh the lost parallelism. Revisit if profiling shows we are SM-bound (not the
  case at small batch — we are launch-bound).

**Why M_TILE=64 instead of 128**:
- wgmma m=64 native; M_TILE=64 = 1 wgmma row group, 0 wasted.
- M_TILE=128 would mean 16 work units, 12% utilization at (B=2, T=1024) — worse.

**Work assignment is static stride** (`work_idx = sm_id + k·GRID`), not work-stealing.
Every tile has the same shape and runtime → no straggler problem. If we ever introduce
variable-cost tiles (e.g. masked/padded tail), switch to atomic-counter work-stealing
(CUTLASS Stream-K pattern); not needed for v1.

**Decision: M_TILE=64, persistent grid of 132 blocks, grid-stride static assignment,
end-of-tile barrier; accept 24% SM utilization at B=2 T=1024.**

**Reference**: [Colfax CUTLASS persistent kernel + Stream-K tutorial](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) — confirms "one CTA per SM, each computes multiple tiles over its lifetime".

> **Note on mbarrier API**: pseudocode in §10.2–10.8 uses shorthand
> `mbar.wait_for_full() / mbar.arrive_empty()`. These map to the real PTX
> `mbarrier.try_wait.parity + mbarrier.arrive` per §10.5b. Treat the shorthand as a
> CUTLASS-style helper wrapper.

### 10.2 Streaming f through W_out GEMM (no full materialization)

**Problem**: f = [a; λδ] has shape `[M_TILE=64, h+r=800]` = 100 KB bf16. We can't keep all
of f in SMEM AND keep all of y in registers (12 N_TILES × 16 fp32/thread = 192 regs/thread,
exceeds the 160-reg consumer budget — even before bookkeeping).

**Fix**: do y = f @ W_out^T as **outer-loop-over-N_TILE, inner-loop-over-K**. Each N_TILE
gets its own wgmma sweep over the full K dimension, then the y_acc fragment is TMA-stored
to global, freeing registers for the next N_TILE.

```cpp
// y[m, n] = sum_k f[m, k] * W_out[n, k]    where m∈[0,M_TILE), n∈[0,d), k∈[0,h+r)
constexpr int K_BLOCK_WOUT = 32;   // see §10.8 — chosen so it never straddles h/r boundary
constexpr int N_BLOCK      = 64;   // wgmma n=64 native
constexpr int F_RING_STAGES = 2;
constexpr int N_TILES      = d / N_BLOCK;          // 12
constexpr int K_BLOCKS_F   = (h + r) / K_BLOCK_WOUT;  // 25 = 24 h-blocks + 1 r-block
// SMEM for f ring: 2 stages × M_TILE × K_BLOCK_WOUT bf16 = 2 × 4 KB = 8 KB

// **Key**: y_acc holds ONE N_TILE worth of fragment at a time = 16 fp32/thread.
// After each N_TILE is done, we TMA-store it and reuse the registers for the next.
float y_acc[16];

for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
    fill_zero(y_acc);
    wgmma_fence();
    for (int k_blk = 0; k_blk < K_BLOCKS_F; ++k_blk) {
        int stage = k_blk % F_RING_STAGES;
        f_full[stage].wait_for_full();        // f chunk produced by consumer-2
        Wout_full[k_blk_global][n_tile].wait_for_full();   // see note on Wout buffering below
        fence_proxy_async_shared_cta();
        for (int kk = 0; kk < K_BLOCK_WOUT / 16; ++kk) {
            wgmma_m64n64k16(y_acc,
                             smem.f_ring[stage] + kk*16,
                             smem.Wout_ring[stage_n][n_tile] + kk*16,
                             /*scale_d=*/1);
        }
        f_empty[stage].arrive();              // (release once per n_tile? see note below)
    }
    wgmma_commit_group();
    wgmma_wait_group<0>();

    // Store y[:, n_tile*N_BLOCK : +N_BLOCK] to global via TMA store.
    // No atomic needed: each block owns a unique M_TILE×d slice of y.
    store_y_n_tile_to_global(y_acc, n_tile);
}
```

**`f` buffering wrinkle** — f is consumed `N_TILES=12` times (once per n_tile), so the simple
"consumer-2 fills, consumer-3 drains, recycle" pattern doesn't work directly. Two options:

1. **Replay-from-SMEM**: keep f in SMEM persistently for the whole y GEMM. SMEM cost
   `M_TILE × (h+r) × 2 = 100 KB` — uses most of our budget but works. Skip the ring.
2. **Recompute f per n_tile**: consumer-2 re-emits f into ring across n_tiles. Cheaper SMEM
   (~8 KB ring) but 12× the work in consumer-2. Bad tradeoff.

**Decision: option 1 — keep f in SMEM persistently.** Allocate `f_smem[M_TILE][h+r]` = 100 KB
once; consumer-2 fills it incrementally as h_blocks complete; consumer-3 reads it
N_TILES times. Use a single `f_ready` named barrier to gate consumer-3 from starting
until f is fully populated.

```cpp
__shared__ alignas(128) bf16 f_smem[M_TILE][h + r];
__shared__ uint64_t f_ready;        // arrives once when consumer-2 finishes all h_blocks + δ

// Consumer-2 (writer): writes f_smem[:, h_blk*H_BLOCK : +H_BLOCK] = ReLU²(z[h_blk]) per h_blk,
//   then writes f_smem[:, h:] = λ·δ, then arrives on f_ready.
// Consumer-3 (reader): waits on f_ready once, then reads f_smem N_TILES times via wgmma.
```

**Synchronization for `Wout_ring`** (producer loads ↔ consumer-3 reads):

Each n_tile reuses the SAME K_BLOCK of W_out across all m_tiles handled by this CTA, but a
DIFFERENT slab per n_tile (Wout has shape [d, h+r] so each n_tile is a unique row stripe).
Producer must load `Wout[n_tile*N_BLOCK : +N_BLOCK, k_blk*K_BLOCK_WOUT : +K_BLOCK_WOUT]` for
all (n_tile, k_blk) pairs.

- 2 mbarriers: `Wout_full[2]`, `Wout_empty[2]`, init `arrival_count=1` (one producer arrival).
- Outer order: consumer-3 loops `n_tile` outer, `k_blk` inner. Producer must match this
  schedule and tile-load in the same outer/inner order.
- Producer arrives `Wout_full[k_blk%2]` after each TMA load; consumer-3 arrives
  `Wout_empty[k_blk%2]` after the last wgmma using that buffer.

**SMEM total revised** (M_TILE=64, K_BLOCK_WOUT=32):
- x_ring: 3 × 64 × 64 × 2 = **24 KB**
- K_ring: 3 × 64 × 8 × 2 = **3 KB** (H_BLOCK is the load slab; see §10.4)
- C_ring: 2 × 8 × 24 × 32 × 2 = **24 KB**
- f_smem (persistent, not ring): 64 × 800 × 2 = **100 KB**
- Wout_ring: 2 × 64 × 32 × 2 = **8 KB**
- z_handoff: 2 × 64 × 8 × 4 (fp32, ring) = **4 KB**
- mbarriers + scratch ≈ **4 KB**
- **Total ≈ 167 KB** (within 228 KB budget, ~60 KB headroom)

If the budget is too tight (e.g., when adding W_out load buffers for n_tile prefetch), swap
`f_smem` to bf16-quantized δ + bf16 a (already bf16) and accept option 2's recompute path
for the largest 6 n_tiles only. Skip in v1.

### 10.3 Saved-z global layout (forward → backward)

**Allocation**: PyTorch autograd Function allocates output + saved tensors and passes
their `data_ptr()`s to a C++ entry point. **TMA descriptors are built inside the C++
launcher** because `CUtensorMap` is an opaque host-side struct that PyBind11 cannot
serialize; only the underlying tensor pointers cross the Python/C++ boundary.

**Python side** (only deals with torch.Tensors):

```python
class RLSplineKVMega(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, K, C, W_out, grid_lo, grid_hi, G, lambda_scale, spline_order):
        # Validate
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        assert K.dtype == torch.bfloat16 and K.is_contiguous()
        assert C.dtype == torch.bfloat16 and C.is_contiguous()
        assert W_out.dtype == torch.bfloat16 and W_out.is_contiguous()
        N, d = x.shape
        h = K.shape[1]; r = C.shape[2]; L = C.shape[1]
        assert h == K.shape[1] == W_out.shape[1] - r
        assert d == K.shape[0] == W_out.shape[0] == x.shape[1]

        # Allocate outputs
        y       = torch.empty((N, d), device=x.device, dtype=torch.bfloat16)
        z_saved = torch.empty((N, h), device=x.device, dtype=torch.bfloat16)

        # Single C++ entry point — builds TMA descriptors internally
        _ext.rl_kv_fwd_megakernel(
            x, K, C, W_out, y, z_saved,           # tensors (data_ptr + shape from torch::Tensor)
            float(grid_lo), float(grid_hi), int(G),
            float(lambda_scale), int(spline_order),
        )

        ctx.save_for_backward(x, K, C, W_out, z_saved)
        ctx.grid_lo = float(grid_lo); ctx.grid_hi = float(grid_hi)
        ctx.G = int(G); ctx.lambda_scale = float(lambda_scale)
        ctx.spline_order = int(spline_order)
        return y

    @staticmethod
    def backward(ctx, g_y):
        x, K, C, W_out, z = ctx.saved_tensors
        g_x     = torch.empty_like(x)              if ctx.needs_input_grad[0] else None
        g_K     = torch.zeros_like(K)              if ctx.needs_input_grad[1] else None
        g_C     = torch.zeros_like(C)              if ctx.needs_input_grad[2] else None
        g_W_out = torch.zeros_like(W_out)          if ctx.needs_input_grad[3] else None

        _ext.rl_kv_bwd_megakernel(
            g_y, x, K, C, W_out, z,
            g_x, g_K, g_C, g_W_out,
            ctx.grid_lo, ctx.grid_hi, ctx.G,
            ctx.lambda_scale, ctx.spline_order,
        )
        return g_x, g_K, g_C, g_W_out, None, None, None, None, None
```

**C++ side** (builds TMA descriptors fresh per call — descriptors are cheap to construct,
~few microseconds each, dominated by host-side validation; CUDA Graphs capture freezes
them after first run):

```cpp
void rl_kv_fwd_megakernel(
    const torch::Tensor& x, const torch::Tensor& K,
    const torch::Tensor& C, const torch::Tensor& W_out,
    torch::Tensor& y, torch::Tensor& z_saved,
    double grid_lo, double grid_hi, int64_t G,
    double lambda_scale, int64_t spline_order)
{
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    // ... other validation ...

    int64_t N = x.size(0), d = x.size(1), h = K.size(1), r = C.size(2), L = C.size(1);

    // Build 6 TMA descriptors on host (see §10.4 for params).
    // Descriptors live on the stack; passed by value to kernel via __grid_constant__.
    CUtensorMap tm_x    = make_tma_2d_bf16(x.data_ptr<__nv_bfloat16>(),    N, d, /*box=*/{64, 64}, SWIZZLE_128B);
    CUtensorMap tm_K    = make_tma_2d_bf16(K.data_ptr<__nv_bfloat16>(),    h, d, /*box=*/{8, 64},  SWIZZLE_128B);
    CUtensorMap tm_C    = make_tma_3d_bf16(C.data_ptr<__nv_bfloat16>(),    h, L, r, /*box=*/{8, 24, 32}, SWIZZLE_64B);
    CUtensorMap tm_Wout = make_tma_2d_bf16(W_out.data_ptr<__nv_bfloat16>(),d, h+r, /*box=*/{64, 32}, SWIZZLE_64B);
    CUtensorMap tm_y    = make_tma_2d_bf16(y.data_ptr<__nv_bfloat16>(),    N, d, /*box=*/{64, 64}, SWIZZLE_128B);
    CUtensorMap tm_z    = make_tma_2d_bf16(z_saved.data_ptr<__nv_bfloat16>(),N,h,/*box=*/{64, 8}, SWIZZLE_NONE);

    dim3 grid(132); dim3 block(512);
    int smem_bytes = 167 * 1024;
    // Required for SMEM > 48 KB:
    cudaFuncSetAttribute(rl_kv_fwd_megakernel_impl<2>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);

    if (spline_order == 2) {
        rl_kv_fwd_megakernel_impl<2><<<grid, block, smem_bytes>>>(
            tm_x, tm_K, tm_C, tm_Wout, tm_y, tm_z,
            (int)N, (int)d, (int)h, (int)r, (int)L,
            (float)grid_lo, (float)grid_hi, (int)G, (float)lambda_scale);
    } else {
        rl_kv_fwd_megakernel_impl<1><<<grid, block, smem_bytes>>>(...);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

**Why C++ owns the descriptor**:
- `CUtensorMap` is a 128-byte opaque struct (PTX 8.5 §B "Tensor-map"). It cannot be
  pickled / passed via PyBind11 cleanly.
- `cuTensorMapEncodeTiled` is CUDA Driver API; cleanest to call from the same TU as the
  kernel launch.
- Descriptor build cost is amortized once per `forward()` call (microseconds). Under
  CUDA Graph capture, the descriptor on the host stack is recorded as part of the graph
  metadata, so re-encoding is zero-cost on replay.

**Decisions on saved state for backward**:

**Risk with naive bf16 z** (rejected): if z lies near a bin boundary (e.g. z = bin_lo + 1e-4),
the bf16 round-trip in `z_saved` may push it across the boundary in backward. Then forward
used `bin_idx = b`, backward uses `bin_idx = b+1`, basis weights flip, **gradient
becomes inconsistent with the actual forward output**. Even ~1% of tokens flipping bin
can blow up g_C rel_err.

**Risk with bin_idx + τ alone** (also rejected): reconstructing
`z = grid_lo + (bin_idx + τ) / scale` from saved bin/τ loses information for
**out-of-range z** (z < grid_lo or z > grid_hi). Forward clamps τ to [0, 1] and pins
bin_idx to {0, G-1}, so the clamped (bin, τ) state reconstructs to the *clamp boundary*,
not the original z. The activation derivative `phi'(z) = 2·z·1[z>0]` for ReLU² needs
the **unclamped** z to be correct. Reconstructing from (bin, τ) would silently zero the
activation gradient on out-of-range tokens that should have non-zero phi'(z).

**Fix — save all four**: `bin_idx + τ + z_bf16 + in_range` per element:

| Per-element | bytes | dtype | role |
|---|---|---|---|
| `bin_idx` | 1 | uint8 | spline path: exact forward-matching bin (no boundary flip) |
| `τ` | 2 | bf16 | spline path: exact forward-matching basis weights |
| `z_bf16` | 2 | bf16 | activation path: actual unclamped z for `phi'(z)` |
| `in_range` | (packed bit) | uint8 → 1 bit | Bool: was z in [grid_lo, grid_hi]? Drives spline-path gradient masking |
| **total** | **5 bytes/elem unpacked, 5⅛ packed** | | (vs naive bf16 z = 2 bytes) |

The `in_range` bit is packed 8-elements-per-byte into a separate `[N, h/8] uint8` tensor,
so per-tensor overhead is `(1 + 2 + 2) × N × h + N × h / 8 = 5.125 N h bytes`.

**Per-tensor cost**: at N=2048, h=768 → **~7.7 MB** (vs naive bf16 z = 3 MB; vs
bin_idx+τ-only = 4.5 MB). Extra ~3-5 MB to guarantee correctness of both spline
gradient AND activation gradient on every token, including out-of-range ones.

**Why we save z_bf16 instead of recomputing from x@K**: re-running `z = x@K` in
backward would cost an extra full GEMM (~1.6 GFLOPS / layer × 12 layers ≈ 20 GFLOPS
/ step) — that re-GEMM is bigger than the entire spline kernel cost we're trying to
optimize. Saving 3 MB of z_bf16 instead is a clear win.

**Layout**: each of the 4 tensors row-major `[N, h]` (or `[N, h/8]` for in_range).
TMA-load with 4 separate descriptors. swizzle = NONE for all (innermost dim H_BLOCK=8
is below the smallest 32B swizzle granularity for the per-element tensors; in_range
is too narrow to swizzle).

**Stress test (gate for Phase 4)**: at Phase 4 verification, generate inputs with z values
clustered ±ε around bin boundaries (ε = 1e-3) AND clustered just outside [grid_lo, grid_hi]
(z = grid_hi + δ for δ ∈ {1e-3, 1e-2, 0.1, 1.0}). Confirm `g_C` and `g_z` rel_err ≤ 2e-3
on both edge sets. **Without this test we cannot rule out the silent boundary-flip /
out-of-range gradient bugs.**

### 10.4 TMA descriptors — exact specs

**Reference**: CUDA C++ Programming Guide §4.11 "Asynchronous Data Copies", and the
`cuTensorMapEncodeTiled` API in `cuda.h`. Followed [TMA introduction blog](https://veitner.bearblog.dev/tma-introduction/) for concrete parameters.

For each tensor we need:
- **rank**, **dataType**, **globalAddress**, **globalDim[]**, **globalStrides[]** (in bytes,
  except first; first stride is implicit 1 element),
- **boxDim[]** (the SMEM tile shape), **elementStrides[]** (always all 1 for our case),
- **interleave** = `CU_TENSOR_MAP_INTERLEAVE_NONE`,
- **swizzle** = per-tensor (see table below; depends on innermost box dim)
- **l2Promotion** = `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`,
- **oobFill** = `CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE` (we mask in-kernel; no OOB tokens
  for production shapes).

Concrete table (M_TILE=64, K_BLOCK=64, H_BLOCK=8, K_BLOCK_WOUT=32, N_BLOCK=64):

| Tensor | rank | shape | boxDim (innermost-first) | swizzle | dtype | direction |
|---|---|---|---|---|---|---|
| **x** | 2 | `(N_tokens, d)` = `(B*T, 768)` | `(K_BLOCK=64, M_TILE=64)` | 128B | bf16 | load |
| **K** | 2 | `(h, d)` (logical [d, h] but descriptor says [h, d] for transpose-on-read) | `(K_BLOCK=64, H_BLOCK=8)` | 128B | bf16 | load |
| **C** | 3 | `(h, L, r)` = `(768, 22, 32)` | `(r=32, L=22→pad-to-24, H_BLOCK=8)` | **64B** | bf16 | load |
| **W_out** | 2 | `(d, h+r)` = `(768, 800)` | `(K_BLOCK_WOUT=32, N_BLOCK=64)` | **64B** | bf16 | load |
| **y** | 2 | `(N_tokens, d)` | `(N_BLOCK=64, M_TILE=64)` | 128B | bf16 | store |
| **z_saved** | 2 | `(N_tokens, h)` | `(H_BLOCK=8, M_TILE=64)` | **none** | bf16 | store |

**Box constraint** (PTX 8.5 §9.7.14.5.1.7 + CUTLASS docs): swizzle mode requires the
innermost contiguous dimension to be **exactly** the swizzle byte count (or a multiple of it):
- **128B swizzle**: innermost = 128 bytes = 64 bf16 elements
- **64B swizzle**: innermost = 64 bytes = 32 bf16 elements
- **32B swizzle**: innermost = 32 bytes = 16 bf16 elements

Per-tensor swizzle decision:

| Tensor | innermost (bf16) | bytes | swizzle |
|---|---|---|---|
| **x** | K_BLOCK = 64 | 128 | 128B ✓ |
| **K** | K_BLOCK = 64 (after transpose, see below) | 128 | 128B ✓ |
| **C** | r = 32 | 64 | **64B** (not 128B!) |
| **W_out** | K_BLOCK_WOUT = 32 | 64 | **64B** |
| **y** | N_BLOCK = 64 | 128 | 128B ✓ |
| **z_saved** | H_BLOCK = 8 | 16 | **none** (smaller than 32B) |

**K transpose-on-load**: PyTorch stores `K` as `[d, h]` row-major (d-stride = h*2 bytes,
h-stride = 2 bytes). For wgmma `z = x @ K^T`, we want K loaded as `[K_BLOCK=64,
H_BLOCK=8]` with K_BLOCK innermost → innermost stride 2 bytes (= 1 K-element). That
matches loading the K matrix's d axis as innermost. So TMA box for K = `(K_BLOCK=64,
H_BLOCK=8)` reading K as if `K[d, h]` is stored col-major — equivalent to telling TMA
the matrix is `[h, d]` row-major and asking for a `(H_BLOCK=8, K_BLOCK=64)` box. Either
flavor works; pick whichever lets us pass `K.T.contiguous()` from Python without a copy
(it forces a copy; better to flip the descriptor's globalDim order on the C++ side).

**C box innermost = r = 32 (64 bytes) requires 64B swizzle**, not 128B. The wgmma
descriptor for C just needs to match (use 64B swizzle bits in the SMEM descriptor).

**z_saved write skips swizzle** (`CU_TENSOR_MAP_SWIZZLE_NONE`) since H_BLOCK=8 = 16 bytes
is below the smallest 32B swizzle. The z_saved tensor is plain row-major; backward
re-reads it without any swizzle decoding.

**Host-side construction** (pseudocode for `build_tma_2d`):

```cpp
CUtensorMap make_tma_2d_bf16_load(const __nv_bfloat16* gptr,
                                    cuuint64_t rows, cuuint64_t cols,
                                    cuuint32_t box_rows, cuuint32_t box_cols,
                                    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {
    CUtensorMap tm;
    cuuint64_t global_dim[2]      = {cols, rows};         // innermost first
    cuuint64_t global_strides[2]  = {cols * sizeof(__nv_bfloat16)};  // only outer stride
    cuuint32_t box_dim[2]         = {box_cols, box_rows}; // innermost first
    cuuint32_t element_strides[2] = {1, 1};
    CUresult res = cuTensorMapEncodeTiled(
        &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, /*rank=*/2,
        (void*)gptr, global_dim, global_strides + 1, /* skip implicit innermost */
        box_dim, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(res == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed");
    return tm;
}
```

**Kernel signature** (must use `__grid_constant__`):

```cpp
__global__ void __launch_bounds__(512, 1)
rl_kv_fwd_megakernel(
    const __grid_constant__ CUtensorMap tm_x,
    const __grid_constant__ CUtensorMap tm_K,
    const __grid_constant__ CUtensorMap tm_C,
    const __grid_constant__ CUtensorMap tm_Wout,
    const __grid_constant__ CUtensorMap tm_y,
    const __grid_constant__ CUtensorMap tm_z,
    int N, int d, int h, int r, int L,
    float grid_lo, float grid_hi, int G, float lambda_scale)
```

### 10.5 Parameter-gradient reduction (no workspace blowup)

**Problem**: g_K[d=768, h=768] full-rank atomic. With token-tiled grid (32 token tiles for
B*T=2048), each token tile contributes a full [d, h] partial. Naive atomic-add: 32 ×
590K = 18M atomics ≈ 18 ms. Split-K replication workspace: 132 × 590K × 2 B × 6 layers
= 940 MB — too heavy.

**Fix**: split the **output** (parameter) dimension across blocks, not the K (token)
dimension. Each block owns a unique slice of the parameter gradient — no replication, no
inter-block atomics, just intra-block reduction in SMEM.

**Backward grid is 2D**: `(token_tile, output_tile)`. Each block:
- iterates over its own subset of tokens (token_tile)
- reduces partial outer-products into SMEM accumulator owned by output_tile
- at end: writes its (token_tile, output_tile) slice to global once, with a single
  inter-token-tile reduction (atomic-add ONLY across the few token_tiles that share an
  output_tile)

For our shape:
- g_K[d=768, h=768]: split into 12×12 = 144 output tiles of (64, 64). Grid:
  (token_tiles=32, n_tile_K=12, m_tile_K=12) — but that's 4608 blocks, 35× the SM count.
  Reduce token_tile dim: assign each (m_tile_K, n_tile_K) to ⌈32/132⌉ = 1-2 blocks via
  persistent grid.
- Practical layout: **single grid (132 blocks)** with each block processing a stripe of
  (token_tiles) × all (output_tiles). Each block accumulates [d × h] in SMEM (590KB —
  too big for SMEM!) → need finer slicing.

**Refined: hierarchical reduction.**

```cpp
// BWD grid = (token_tiles, n_tile_param)
//   token_tiles: ⌈B*T / M_TILE⌉  = 32 (for B*T=2048)
//   n_tile_param: d / N_TILE_K  = 12 (for d=768, N_TILE_K=64)
// Each block handles ONE token_tile's contribution to ONE n_tile_param-slice of g_K.
// → block accumulates [d, N_TILE_K=64] of g_K = 768 × 64 fp32 = 192 KB → too big for SMEM
// → Further slice m: block also takes m_tile_param, accumulates [M_TILE_K=64, N_TILE_K=64]
//                     × num_token_tiles_per_block in fp32 = 16 KB.

// Final BWD grid: (n_tile_param=12, m_tile_param=12) = 144 blocks; each block iterates
//   over ALL token_tiles, accumulating into its private SMEM slab, writes ONE slab at end.
// Each block touches the same n/m slab from EVERY token, so does the full token-axis reduction
// itself in SMEM — no inter-block atomics for g_K at all.
```

**Decision: 2D grid `(m_tile_param=12, n_tile_param=12)` = 144 blocks**:
- Each block reads ALL tokens × its own (m_slab=64, n_slab=64) of input/grad.
- Accumulates `[64, 64] fp32 = 16 KB` SMEM for g_K. Same for g_W_out, g_C.
- At end of block: single write of [64, 64] to global at owned slot. **No atomics.**
- 144 blocks > 132 SMs → mild over-subscription, fine.

**g_C special-case** (sparse — only 3 of L bins active per token):
- Grid: `(j_tile=12)` × `(b_tile=L)` = 12 × 22 = 264 blocks.
- Each block accumulates `[J_BLOCK=64, b_per_tile, r=32]` SMEM = 4-12 KB.
- For tokens whose `bin_idx ∉ b_tile`, contributes nothing to this block — kernel
  branches early.

**g_x is per-token** (not a parameter gradient): it goes back into the FFN's input. Same
M-tile parallelism as forward — token_tile grid, write own slice via TMA. No atomics.

**Workspace cost: ZERO** — no global partials, no replication. Pure block-local reduction +
single write per block.

**Tradeoff**: backward kernel is structurally different from forward kernel (different
grid shape, different work distribution). Two distinct kernels (forward megakernel +
backward megakernel). Reuse of TMA descriptor builders + warp specialization + pipeline
infrastructure stays.

**Reference**: this is the standard CUTLASS GEMM split-K pattern (data-parallel along
output, no atomics) — see [CUTLASS sm90_tile_scheduler.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp). Stream-K is only needed when the output tile count is too small to saturate
all SMs; ours has 144 ≥ 132 so plain data-parallel suffices.

### 10.5b Mbarrier specification (single source of truth)

PTX 8.5 §9.7.12 + ThunderKittens `sync.cuh` `init_semaphore` confirm: each mbarrier in
SMEM is initialized once with `mbarrier.init.shared::cta.b64 [bar], total_count`, where
`total_count` is the **expected number of arrivals per phase**. After `total_count`
arrivals, the parity bit flips and waiters are released.

We don't use the `wait_for_full / arrive_empty` symbolic API in actual code — those are
shorthand. Real PTX is `mbarrier.try_wait.parity.shared::cta.b64 P1, [bar], parity` in
a polling loop, plus `mbarrier.arrive.release.cta.shared::cta.b64 _, [bar]`.

Because pipelined ring buffers alternate full/empty across cycles, **each ring slot needs
TWO mbarriers** (one for full, one for empty) and the consumer must track parity (flips
each iteration).

**Forward megakernel mbarrier table**:

| mbarrier | count | producer | consumer | purpose |
|---|---|---|---|---|
| `x_full[3]`, `x_empty[3]` | 1 / 1 | producer WG (TMA) | consumer-1 | x tile ready / consumed |
| `K_full[3]`, `K_empty[3]` | 1 / 1 | producer WG | consumer-1 | K weight tile ready / consumed |
| `C_full[2]`, `C_empty[2]` | 1 / 1 | producer WG | consumer-2 | C tile ready / consumed |
| `Wout_full[2]`, `Wout_empty[2]` | 1 / 1 | producer WG | consumer-3 | W_out tile ready / consumed |
| `z_full[2]`, `z_empty[2]` | 128 / 128 | consumer-1 (writes z_handoff SMEM) | consumer-2 | per-h_block z handoff |
| `f_ready` | 128 | consumer-2 (writes f_smem) | consumer-3 | f fully populated, single-shot |
| `y_smem_ready[2]` | 128 / 1 | consumer-3 (writes y SMEM tile) | producer (TMA-store) | y tile ready for store / store complete |

Producer-WG arrivals use `expect_tx` form for TMA (`mbarrier.expect_tx + cp.async.bulk`
auto-arrives on completion). Consumer arrivals are plain `mbarrier.arrive`. Counts of
`128` mean all threads of a warpgroup must arrive (used when the writer's work spans the
whole WG, e.g. all 4 warps participate in writing z fragment).

**Init sequence** (run once at kernel entry, single thread per CTA):
```cpp
if (threadIdx.x == 0) {
    init_mbar(&x_full[0..2], 1);       init_mbar(&x_empty[0..2], 1);
    init_mbar(&K_full[0..2], 1);       init_mbar(&K_empty[0..2], 1);
    init_mbar(&C_full[0..1], 1);       init_mbar(&C_empty[0..1], 1);
    init_mbar(&Wout_full[0..1], 1);    init_mbar(&Wout_empty[0..1], 1);
    init_mbar(&z_full[0..1], 128);     init_mbar(&z_empty[0..1], 128);
    init_mbar(&f_ready, 128);
    init_mbar(&y_smem_ready[0..1], 128); init_mbar(&y_smem_done[0..1], 1);
}
asm("barrier.sync.aligned 0, 512;");   // all threads wait until init done
```

**Wait pattern** (per consumer-1 example):
```cpp
uint32_t parity = 0;
for (int k_blk = 0; k_blk < d / K_BLOCK; ++k_blk) {
    int stage = k_blk % 3;
    asm("{ .reg .pred P; \n"
        "L: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1; \n"
        "   @!P bra L; }\n"
        :: "r"(__cvta_generic_to_shared(&x_full[stage])), "r"(parity));
    // ... use x_buf[stage] ...
    asm("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :: "r"(__cvta_generic_to_shared(&x_empty[stage])));
    if (stage == 2) parity ^= 1;       // parity flips when ring wraps
}
```

References: TK `sync.cuh` lines 35-78 (init/arrive), 121-130 (try_wait.parity).

### 10.6 z register-→-SMEM handoff (consumer-1 → consumer-2)

**Problem**: wgmma m=64 fp32 acc lives in 4-warp distributed registers. Consumer-2 needs
to read z as a normal SMEM tile.

**Fix**: use `wgmma::store_matrix_sync` (or PTX equivalent) to write each warp's portion
of z fragment to SMEM, then named-barrier sync, then consumer-2 reads.

**Per consumer-1 step**:
```cpp
// After wgmma_wait_group<0>(), z_acc[16] has the m64n_block fragment.
// We want SMEM layout: smem.z_handoff[h_block_idx][m, n] row-major fp32.
// Each thread owns 16 elements of the fragment, distributed over (m=64, n=H_BLOCK=8).
// Layout: m64n8 → 4 elements per thread (mma m16n8 layout); we have m64n8 = 8/8 = 1
//   chunk of 4 elements. So each thread writes 4 fp32 to SMEM.

float* z_smem_base = &smem.z_handoff[ring_slot][0];
constexpr int M = 64, N = H_BLOCK; // 8
// Use __nv_bfloat162 vectorized store via `st.shared.v4.b32` for efficiency,
// but starting from explicit per-element store is fine for Phase 1:
int groupID = lane_id / 4;          // 0..7
int tigid    = lane_id % 4;          // 0..3
int row_warp = warp_id * 16;         // 0, 16, 32, 48 for the 4 warps
// Per-thread elements (mma m16n8 fp32 layout):
#pragma unroll
for (int e = 0; e < 4; ++e) {
    int row = row_warp + groupID + (e/2) * 8;
    int col = tigid * 2 + (e % 2);
    z_smem_base[row * N + col] = z_acc[e];
}
```

**Synchronization**:
- Consumer-1 does the stores above, then `fence.proxy.async.shared::cta`,
  then `cutlass::arch::NamedBarrier::arrive_and_wait(2*128, NB_Z_READY)`.
- Consumer-2 also calls `NamedBarrier::arrive_and_wait(2*128, NB_Z_READY)`, then reads.
- After consumer-2 is done with this h_block: `NamedBarrier::arrive_and_wait(2*128, NB_Z_DONE)`.

**SMEM cost**: `z_handoff[2_RING][M_TILE=64][H_BLOCK=8] fp32 = 2 × 2 KB = 4 KB`.

**Reference**: ThunderKittens `descriptor.cuh` shows the canonical layout; PTX 8.5
§9.7.14.5.1.2 fp32 acc fragment layout for m64nNk16.

### 10.7 f-buffer (replaces the earlier "a ring buffer" plan)

Per the §10.2 decision (option 1), we keep **all of f in SMEM persistently** across the
W_out GEMM, not in a streaming ring. Consumer-2 fills `f_smem[M_TILE][h+r]` incrementally
as it finishes each h_block; consumer-3 waits once for the whole thing to be ready, then
reads it 12 times (one per n_tile) without reloading.

**Layout**:
```cpp
__shared__ alignas(128) bf16 f_smem[M_TILE][h + r];   // 64 × 800 × 2 = 100 KB
__shared__ uint64_t f_ready;                           // single named-barrier-like mbarrier
```

**Convention**: `f_smem[m, k]` is laid out so that `f_smem[m, 0..h-1] = a[m, :]` and
`f_smem[m, h..h+r-1] = λ·δ[m, :]`. Consumer-2 writes a in H_BLOCK chunks (8 cols per
h_block); writes δ as one contiguous r=32 chunk at offset h.

**Producer (consumer-2) per h_block**:
```cpp
// (1) Compute a_local = ReLU²(z_local) in registers.
// (2) Write to SMEM at column offset h_blk*H_BLOCK:
for (int row = 0; row < M_TILE/4_warps_per_wg; ++row) {
    for (int col = 0; col < H_BLOCK; ++col) {
        f_smem[warp_id*M_TILE/4 + row][h_blk*H_BLOCK + col] = a_local[row][col];
    }
}
// No mbarrier yet — consumer-2 itself loops over h_blk before signaling.
```

After all h_blocks are done AND δ is accumulated, consumer-2 writes δ:

```cpp
for (int row = 0; row < M_TILE/4_warps_per_wg; ++row) {
    for (int col = 0; col < r; ++col) {
        f_smem[warp_id*M_TILE/4 + row][h + col] =
            (bf16)(lambda_scale * delta_local[row][col]);
    }
}
// Now f is complete. Issue cross-proxy fence and arrive on f_ready.
fence_proxy_async_shared_cta();
arrive_mbar(f_ready, /*count=*/128);   // all consumer-2 threads contribute
```

**Consumer (consumer-3)**:
```cpp
wait_mbar(f_ready);   // single wait at start; f stays valid for the whole y GEMM
fence_proxy_async_shared_cta();
// Now read f_smem N_TILES times via wgmma (see §10.8 inner loop)
```

**SMEM cost**: `f_smem` = 100 KB (large but unavoidable — N_TILES=12 reads of f mean
recompute is too costly). Single mbarrier `f_ready` initialized with arrival_count = 128
(all 128 threads of consumer-2 must arrive after all h_blocks + δ writes complete).

**`a` ring buffer is removed from the plan** — replaced by persistent `f_smem`. The
earlier "stages of a" pattern was based on streaming f through W_out's K dim, which we
ruled out in §10.2 because of the 12× recompute cost.

### 10.8 Inner W_out GEMM tile structure

**Goal**: `y[M_TILE=64, d=768] = f[:, h+r=800] · W_out[:, h+r=800]^T`. Per the §10.2 outer
loop: for each `n_tile ∈ [0, N_TILES=12)`, sweep K dim through K_BLOCK_WOUT=32 chunks,
TMA-store the resulting [M_TILE, N_BLOCK] tile, then move to the next n_tile.

**f stays in `f_smem`** (per §10.7) — no streaming, no h/r boundary issues, just slice it
by K offset.

```cpp
constexpr int K_BLOCK_WOUT = 32;
constexpr int N_BLOCK      = 64;     // wgmma m64n64
constexpr int N_TILES      = d / N_BLOCK;             // 12
constexpr int K_BLOCKS_F   = (h + r) / K_BLOCK_WOUT;   // 25 = 24 h-blocks + 1 r-block

// f is already complete in f_smem (waited on f_ready once before this loop).

for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
    float y_acc[16] = {0};   // 16 fp32/thread for one m64n64 fragment
    wgmma_fence();

    for (int k_blk = 0; k_blk < K_BLOCKS_F; ++k_blk) {
        int k_start = k_blk * K_BLOCK_WOUT;   // 0, 32, 64, ..., 768 (= h), 800

        // Wait for the W_out tile [:, k_start:+K_BLOCK_WOUT] for THIS n_tile.
        // Producer schedule: load (n_tile, k_blk) in the same outer/inner order.
        Wout_full[k_blk % WOUT_STAGES].wait_for_full();
        fence_proxy_async_shared_cta();

        for (int kk = 0; kk < K_BLOCK_WOUT / 16; ++kk) {  // 2 wgmma per K_BLOCK_WOUT
            const __nv_bfloat16* a_ptr = &f_smem[0][k_start + kk*16];   // f K offset
            const __nv_bfloat16* b_ptr = &Wout_buf[k_blk % WOUT_STAGES][kk*16];
            wgmma_m64n64k16(y_acc, a_ptr, b_ptr, /*scale_d=*/1);
        }

        // Release this Wout buffer slot for the producer to refill (next n_tile or wraparound).
        Wout_empty[k_blk % WOUT_STAGES].arrive();
    }

    wgmma_commit_group();
    wgmma_wait_group<0>();

    // Epilogue: y_acc fragment → SMEM tile → TMA store to global. See §10.8a.
    store_y_n_tile_to_global(y_acc, n_tile);
}
```

#### 10.8a Output epilogue (fp32 fragment → SMEM → TMA store)

Pattern (from FA3 `epilogue_fwd.hpp` lines 234-260): wgmma fp32 accumulator is in
register fragment; we cannot TMA-store registers. Two-step:

1. **R2S (register → SMEM)**: each thread writes its 16 fp32 elements to a SMEM tile,
   converting fp32 → bf16 in-flight. Use canonical wgmma m64n64 fp32 layout from §10.6:
   each thread owns 16 elements at known (row, col) positions. Plain `st.shared.b32`
   suffices (no need for stmatrix; stmatrix is bf16-only).

   ```cpp
   __shared__ alignas(128) bf16 sY[M_TILE][N_BLOCK];   // 64×64×2 = 8 KB (overwritten per n_tile)
   // After wgmma_wait_group<0>():
   #pragma unroll
   for (int chunk = 0; chunk < 4; ++chunk) {     // 4 chunks of 4 elements (m64n64 layout)
       #pragma unroll
       for (int e = 0; e < 4; ++e) {
           int frag_idx = chunk * 4 + e;
           int gid = lane_id / 4;
           int tig = lane_id % 4;
           int row_w = (e < 2) ? gid : gid + 8;
           int col_w = tig*2 + (e % 2);
           int row = warp_id * 16 + row_w;
           int col = chunk * 16 + col_w;       // 16 cols/chunk × 4 chunks = 64 cols
           sY[row][col] = (bf16)y_acc[frag_idx];
       }
   }
   ```

2. **fence + arrive**: `fence.proxy.async.shared::cta` then arrive on `y_smem_ready`.

3. **TMA store (issued by one thread of producer WG, or any thread we elect)**:
   ```cpp
   if (warp_id == 0 && lane_id == 0) {
       cp.async.bulk.tensor.2d.global.shared::cta.bulk_group
           [tm_y, {token_start, n_tile*N_BLOCK}], [sY];
       cp.async.bulk.commit_group;
       cp.async.bulk.wait_group 0;
   }
   ```
   `bulk_group` completion mechanism is the SMEM→global counterpart of mbarrier (PTX
   8.5 §9.7.8.24.10). `wait_group 0` ensures the previous n_tile's store finishes
   before we overwrite sY for the next n_tile.

**SMEM cost**: `sY` = 8 KB shared with all consumer-3 work; reused across n_tiles.

**z_saved store** uses the same pattern (R2S then TMA), but because z is fp32 wgmma acc
internally and we save bf16, we just downcast on R2S. SMEM tile `sZ[M_TILE][H_BLOCK] bf16
= 1 KB`.

**W_out producer schedule**: producer must walk `(n_tile, k_blk)` in the same nested order
as consumer-3, with 2-stage double-buffering on the inner k_blk axis. Producer loads:

```
for n_tile in 0..N_TILES:           # outer
  for k_blk in 0..K_BLOCKS_F:        # inner
    wait Wout_empty[k_blk%2]
    cp.async.bulk.tensor.2d Wout_buf[k_blk%2] ← W_out[n_tile*N_BLOCK : +N_BLOCK,
                                                       k_blk*K_BLOCK_WOUT : +K_BLOCK_WOUT]
    arrive Wout_full[k_blk%2]
```

**SMEM cost for Wout**: 2 stages × `[N_BLOCK=64, K_BLOCK_WOUT=32]` bf16 = **8 KB**.

### 10.9 Backward consumer-2 register/SMEM budget

**Operations consumer-2 does in backward**:
1. Load `z` saved from forward (TMA from `z_saved`)
2. Compute `bin, τ, B, dB` from z (recompute, cheap)
3. Receive `g_δ` from consumer-1 (via SMEM handoff)
4. Compute `g_z_spline[m, j] = scale * Σ_k dB_k(τ) · ⟨g_δ[m], C[j, bin+k, :]⟩` — **one r-dot per k per (m,j)**
5. Scatter `g_C[j, bin+k, :] += B_k(τ) · g_δ[m]` per (m,j) and k — **3 vector accumulations of size r**

**Register budget per consumer-2 thread (160 regs after §3.1 fix)**:

The naive "all loop vars live at once" tally (B0/B1/B2/dB×3/bin/τ/z/g_z_spline = ~190 regs)
**does not fit** in the 160 budget. Restructure the loop to keep only one M-row's
worth of intermediates live at a time, paying for an outer loop over M:

```cpp
// Outer loop: 16 M rows / thread (M_TILE=64, 4 warps/WG, 32 lanes/warp → 16 rows/lane?
// Actually: M_TILE=64, NCWG-2 has 128 threads, m64n8 fragment layout puts 4 elements
// per thread spanning rows {gid, gid+8} × cols {2*tig, 2*tig+1}. So each thread owns 4
// rows total, not 16.).
// Rows-per-thread = 4 (verified via wgmma fragment layout).

for (int m_local = 0; m_local < 4; ++m_local) {       // rows owned by this thread
    // Live-once registers:
    float z_m;       // 1 reg
    float B0, B1, B2, dB0, dB1, dB2;   // 6 regs
    int bin_idx;                       // 1 reg
    float tau;                         // 1 reg
    // Per-h_block live:
    for (int j_local = 0; j_local < H_BLOCK_BWD/4; ++j_local) {
        float g_z_spline_acc = 0.0f;   // 1 reg, fused into outer m loop
        // ...gather C[j, bin+k, :] from SMEM, dot with g_δ (in registers), mul dB_k...
    }
}
```

**Live register count after restructure**: ~50-60 regs (z_m, B/dB, bin/τ, g_z_spline_acc,
g_δ row in regs only briefly, plus loop counters + SMEM ptrs). Comfortably ≤ 160. ✓

**g_C accumulator stays in SMEM** (5.5 KB, atomic-add inside block):
- Per block, `[H_BLOCK=8, L=22→pad 24, r=32] fp32 = 6 KB SMEM`
- intra-block atomic-add (SMEM atomics on Hopper run at ~1 ns each — fast)
- final bulk-write to `g_C[h_block_owned_by_this_block, :, :]` slice in global (no
  inter-block atomics due to §10.5 grid-by-output-tile design)

If after Phase 4.2 NCU shows `smsp__sass_local_load.sum > 0` (= register spill),
restructure further: split consumer-2 work across two warpgroups (degrade to NCWG=2 +
240 regs each), with consumer-2A doing g_C scatter and consumer-2B doing g_z_spline
accumulation.

**g_C accumulator**:
- Per block, accumulator shape `[H_BLOCK=8, L=22, r=32] fp32 = 5.5 KB` SMEM (cheap).
- Use SMEM atomic-add inside block (we know SMEM atomicAdd works from the v1 kernel
  experience), then bulk write to `g_C_partial[sm_id]` at end.

**`g_z_spline` reduction order**:
- For each token m and key j: `g_z_spline[m, j] = scale * Σ_{k=0..2} dB_k(τ) · ⟨g_δ[m, :], C[j, bin+k, :]⟩`
- Fastest: keep g_δ in registers (r=32 fp32 = 32 regs); for each (m, j) gather C[j, bin+k, :] from SMEM, dot-with-g_δ, multiply by dB_k, sum into accumulator.
- Per (m, j): 3 dot products of length 32 = 96 fma + 3 fp32 multiply-accumulate. Fast.

### 10.10 Phase 1 sub-steps (more granular)

Phase 1 was previously "single-WG forward, no producer". Split into:

**Phase 1.1** — z = K x only:
- Single-WG kernel that does only the K-linear via wgmma.
- Output: z [N, h] bf16 to global.
- Verify: `||z - x @ K^T||_rms / ||z||_rms ≤ 1.7e-3`.

**Phase 1.2** — add spline forward δ:
- Same kernel; also computes B/τ/bin from z (using existing wgmma kernel logic).
- Loads C tile via cp.async, computes δ.
- Output: δ [N, r] bf16 to global (additional output).
- Verify: `δ` matches reference within 1.7e-3 rel rms.

**Phase 1.3** — add y = f @ W_out^T:
- Same kernel; builds f = [a; λδ] in SMEM, runs second wgmma against W_out.
- Output: y [N, d] bf16 (final output of FFN).
- No more intermediate δ output (it's consumed in-kernel now).
- Verify: full forward end-to-end vs reference within 2e-3 rel rms.

Each sub-step has its own micro-test that compares against the existing reference path
(`flash_spline_feature_reference` + manual K/W_out linear).

### 10.11 Python autograd API (full signature)

Already shown in §10.3. Restated for completeness:

```python
class RLSplineKVMega(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, K, C, W_out, grid_lo, grid_hi, G, lambda_scale, spline_order=2):
        """
        x: [B*T, d]      bf16
        K: [d, h]        bf16
        C: [h, L, r]     bf16
        W_out: [d, h+r]  bf16
        grid_lo, grid_hi: float
        G: int (grid intervals)
        lambda_scale: float
        spline_order: int (1 or 2; B1 / B2)
        Returns:
        y: [B*T, d]      bf16
        """
        ...

    @staticmethod
    def backward(ctx, g_y):
        """
        g_y: [B*T, d]    bf16
        Returns gradients in the order of forward inputs:
            (g_x, g_K, g_C, g_W_out, None, None, None, None, None)
        """
        ...
```

`spline_order` is constexpr-promoted via dispatch table on the C++ side
(`if constexpr (SPLINE_ORDER == 2) { ... } else { ... }`). Two compiled kernel variants.

### 10.12 Edge cases (M_TILE / H_BLOCK divisibility, tail handling)

- **N_tokens not multiple of M_TILE**: use TMA's OOB fill (`OOB_FILL_NONE` masks the
  read; in-kernel mask applies). For the last tile, set up a guard variable
  `tile_size = min(M_TILE, N_tokens - token_start)` and only write `tile_size` rows in
  the y store.
- **h not multiple of H_BLOCK**: H_BLOCK=8 and h=768 → 96 h_blocks exactly. For
  non-divisible cases (e.g., h=750), pad C/K/W_out to nearest H_BLOCK=8 multiple and
  zero-fill (init time cost only). Skip in v1 (production h=768 always divides).
- **L not power-of-2**: L_PAD = round_up(L, 8). Pad C tile in SMEM; W_out doesn't see L.

### 10.13 CUDA Graphs — static allocation, fixed addresses, capture-safe autograd

CUDA Graphs require **every kernel argument pointer to be the same on each replay**.
Three things commonly break this in PyTorch and must each be controlled.

**(a) Static input/target tensors** — preallocate once, copy into them every step:

```python
# One-time setup, outside any graph:
static_idx     = torch.empty((B, T), dtype=torch.long, device='cuda')
static_targets = torch.empty((B, T), dtype=torch.long, device='cuda')

def real_step(real_idx, real_targets):
    static_idx.copy_(real_idx, non_blocking=True)
    static_targets.copy_(real_targets, non_blocking=True)
    g.replay()
    return static_loss  # static tensor; .item() reads its current value
```

**(b) Static gradient buffers** — `optimizer.zero_grad(set_to_none=False)`:

`set_to_none=True` (the PyTorch default since 1.7) **deallocates `param.grad` every step**
and reallocates inside backward, giving each replay a different pointer for the optimizer
step. Force the buffers to stay alive:

```python
# After model init, before any forward pass:
for p in model.parameters():
    if p.grad is None:
        p.grad = torch.zeros_like(p)  # allocate once
optimizer = torch.optim.AdamW(model.parameters(), ...)
# Per-step:
optimizer.zero_grad(set_to_none=False)  # zero in place, keep address
```

**(c) Static megakernel-internal scratch** — our megakernel needs:
- `g_f [N, h+r]` (forward fwd→bwd handoff, ~6 MB / layer)
- `g_z_spline [N, h]` (bwd 4.2 → bwd 4.3 handoff, ~3 MB / layer)
- `bin_idx [N, h] uint8 + τ [N, h] bf16` (saved from fwd, used in bwd 4.2 / 4.3)

These must be **module-attribute tensors**, not freshly allocated inside fwd, or each
replay sees different pointers and re-encodes TMA descriptors:

```python
class RLKVCell(nn.Module):
    def __init__(self, ...):
        ...
        # Persistent scratch for CUDA Graph capture
        self.register_buffer('_scratch_g_f',
            torch.empty((B*T, h+r), dtype=torch.bfloat16, device='cuda'),
            persistent=False)
        self.register_buffer('_scratch_bin', ..., persistent=False)
        self.register_buffer('_scratch_tau', ..., persistent=False)

    def forward(self, x):
        return rl_kv_fused.apply(x, self.K, self.C, self.W_out,
                                 self._scratch_g_f, self._scratch_bin, self._scratch_tau)
```

The megakernel's TMA descriptors are encoded ONCE on module first-use (cached on
`self`) using these buffer addresses. CUDA Graph capture sees stable pointers.

**(d) Capture sequence** with PyTorch's recommended warmup pattern (avoids capturing
allocator side-effects from cuBLAS handles, autograd lazy state, etc.):

```python
# Step 1: 11 warmup steps in eager mode on a dedicated stream
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(11):  # PyTorch docs recommend ≥ 3; we use 11 for safety
        optimizer.zero_grad(set_to_none=False)
        static_loss = model(static_idx, targets=static_targets)
        static_loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# Step 2: capture
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=False)
with torch.cuda.graph(g):
    static_loss = model(static_idx, targets=static_targets)
    static_loss.backward()
    optimizer.step()

# Step 3: replay loop
for batch in loader:
    static_idx.copy_(batch.idx)
    static_targets.copy_(batch.targets)
    g.replay()
    log_loss(static_loss.item())
```

**(e) Capture-safe autograd test** — before claiming victory, verify the graph itself
gives identical outputs and grads as eager:

```python
def test_capture_safe():
    model_eager = build()
    model_graph = copy.deepcopy(model_eager)
    # ... build static tensors + capture graph for model_graph ...
    for trial in range(20):
        idx = torch.randint(0, V, (B, T), device='cuda')
        # eager
        loss_eager = model_eager(idx, targets=...)
        loss_eager.backward()
        ge = {n: p.grad.clone() for n,p in model_eager.named_parameters()}
        model_eager.zero_grad(set_to_none=False)
        # graph
        static_idx.copy_(idx); static_targets.copy_(...)
        g.replay()
        loss_graph = static_loss.item()
        gg = {n: p.grad.clone() for n,p in model_graph.named_parameters()}
        # Compare
        assert math.isclose(loss_eager.item(), loss_graph, rel_tol=1e-3)
        for n in ge:
            assert torch.allclose(ge[n], gg[n], rtol=2e-3, atol=1e-4), n
```

**Constraints summary**:
- All shapes static (B, T, d, h, L, r fixed at capture time).
- `optimizer.zero_grad(set_to_none=False)` always.
- `param.grad` preallocated before first `.backward()`.
- Internal scratch tensors live as `register_buffer(persistent=False)`.
- TMA descriptors re-encoded ONLY when buffer pointers change (cached on the module).
- ≥ 11 eager warmup steps on a dedicated stream before capture.
- `loss.item()` after `g.replay()` reads from the SAME static tensor every time.

**Net win**: ~30 kernel launches per layer × 12 layers = 360 launches/step → **1 graph
submit/step**. At ~3 μs Python+CUDA overhead per launch this saves ~1.1 ms/step that the
megakernel alone cannot recover.

**References**:
- [PyTorch CUDA Graph notes](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) — capture/replay constraints, warmup-on-side-stream pattern
- [PyTorch CUDA Graph Trees](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) — `torch.compile`'s automatic capture (alternative if manual capture proves brittle)

### 10.14 Phase 7 FP8 scaling strategy

**Save for after BF16 megakernel hits the wall target.** Per-tensor scaling is rejected
upfront — RL-KV's spline path concentrates most signal in a few high-activation channels
near the grid boundaries, so a single tensor-wide scale would either clip the bins that
matter or under-utilize the FP8 dynamic range across the rest. Use **fine-grained
scaling** end-to-end (DeepGEMM-style).

**Scaling granularity per tensor**:

| Tensor | Shape | Scale layout | Why |
|---|---|---|---|
| `K` (input proj) | `[d, h]` | per `[1, 128]` block along d-axis (row scale per 128-d chunk) | matches `x @ K` GEMM tile structure; spread across d |
| `W_out` | `[d, h+r]` | per `[1, 128]` block along d-axis | same rationale |
| `C` (spline coeffs) | `[h, L, r]` | per `[1, 1, r]` (one scale per (h, bin)) | each bin's coefficients have own dynamic range; rare-bin channels need own scale |
| `x`, `f`, `g_*` | activations | per-block `[128, ?]` dynamic | DeepGEMM-style dynamic activation scaling |

This is "1×128 weights + 128×128 activations" tiling matching DeepGEMM's
`m_grouped_gemm_fp8_fp8_bf16_nt` pattern.

**Quantize C tensor LAST** (critical ordering for our spline path):

The spline coefficient tensor `C` accumulates into a low-magnitude residual (recall
ρ_δ ≈ 0.20-0.45 from Phase B/v3.5 results). Quantizing C to FP8 too early can crush
small but important bins. Required ordering:

1. **First**: BF16 megakernel passes both quality (≤ 1.0% perplexity delta vs MLP) and
   wall-time targets across full 50K-step nanochat run. Keep C in BF16.
2. **Then**: convert K and W_out to FP8 (these are dense GEMM weights, similar to
   transformer Q/K/V/O — well-understood quantization). Verify quality + wall again.
3. **Finally**: convert C to FP8 with per-(h, bin) scales. This is the high-risk step;
   keep a runtime fallback to BF16 C for the first epoch.

**Why this order**: at each step, only one variable changes — if quality regresses we
know exactly which conversion broke it. Running all three FP8 conversions at once would
require a full bisection if anything regresses.

**Dynamic vs static scales**:
- Activations (`x`, `f`, `g_*`): **dynamic per-block** scales computed inline (cheap on
  Hopper, ~1-2% kernel time).
- Weights (`K`, `W_out`, `C`): **static scales recomputed every N=200 steps** from a
  rolling AbsMax — avoids per-step quantization cost but tracks weight drift during
  training.
- Reference: DeepGEMM ships the `(scale_a, scale_b)` API as separate input tensors;
  we mirror that.

**FP8 format choice**: E4M3 for weights and forward activations (more precision, narrower
range matches post-LN distributions); E5M2 for backward gradients (wider range matches
high-magnitude `g_y` near loss).

**Skip path**: if BF16 megakernel already beats MLP wall by ≥ 1.5×, **do not pursue FP8
in v1**. The quality risk on C is non-trivial and the paper's claim hinges on quality
parity.

**References**:
- [DeepGEMM (DeepSeek)](https://github.com/deepseek-ai/DeepGEMM) — per-block FP8 GEMM at >1 PFLOP/s on H100 with 1×128 weight scales + 128×128 activation scales
- [TransformerEngine](https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fp8_primer.ipynb) — E4M3 / E5M2 format primer + delayed scaling reference impl
- [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](https://arxiv.org/abs/2209.05433) — original E4M3 / E5M2 design rationale

### 10.15 Profiling targets (NCU metrics to watch)

- `sm__inst_executed_pipe_tensor.sum` — wgmma issue count
- `sm__throughput.avg.pct_of_peak_sustained_active` — overall throughput vs peak
- `dram__throughput.avg.pct_of_peak_sustained_active` — HBM saturation (target ≥ 80% during
  steady-state)
- `smsp__sass_local_load.sum` — register spill detector (target = 0)
- `l1tex__t_sectors_pipe_lsu_mem_global_op_atomic.sum` — global atomic count (target = 0 in
  Phase 4 — every parameter-grad write is uncontested per the 3-kernel split)
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy (target ≥ 50%)

### 10.16 Build & verification gating spec (single source of truth)

**Compile flags** — wgmma is sm_90 **architecture-specific**, not sm_90 generic:

```bash
nvcc -arch=sm_90a \                          # NOT sm_90 — wgmma.fence requires sm_90a
     -gencode arch=compute_90a,code=sm_90a \ # explicit per-SM gen
     -std=c++20 -O3 \
     --expt-relaxed-constexpr \
     --use_fast_math \                       # OK for forward/backward (we own tolerance gates)
     --ptxas-options=-v \                    # PRINT register usage per kernel — gating signal
     --resource-usage \                      # PRINT SMEM / regs / stack
     -DCUDA_FORCE_CDP1_IF_SUPPORTED=0 \
     -DTORCH_USE_CUDA_DSA \                  # device-side asserts in dev builds (strip for prod)
     spline_kv_megakernel.cu
```

If `nvcc` is invoked with `-arch=sm_90` (no `a`), `wgmma.fence` and the wgmma SMEM
descriptor's swizzle bits silently produce wrong PTX → kernel returns garbage. Already
hit on our prior wgmma backward kernel; do NOT repeat.

`torch.utils.cpp_extension.load(...)` users: pass via `extra_cuda_cflags=["-arch=sm_90a", ...]`
explicitly (PyTorch's default is `-arch=sm_XX` from the device, which on H100 is `sm_90`,
NOT `sm_90a`).

**Hard gates** — every Phase 0-7 milestone must pass ALL of these or block:

| Gate | Tool | Pass criterion | Failure action |
|---|---|---|---|
| **Compile-arch** | parse `nvcc -v` output | `arch=sm_90a` appears | fix build script |
| **No spill** | `ncu --metrics smsp__sass_local_load.sum,smsp__sass_local_store.sum` | both = 0 | reduce reg pressure: drop accum tile size, more inner-loop unroll, NCWG=2 fallback |
| **No global atomics** (Phase 4) | `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_atomic.sum` | = 0 across `bwd_y`, `bwd_spline`, `bwd_K` | grid-by-output-tile is wrong; check kernel grid and atomic-free invariant |
| **No deadlock / hang** | `cuda-gdb` + `info cuda kernels` (or just `cudaDeviceSynchronize` with 5 s timeout in test harness) | kernel returns within 5 s on B=2 T=1024 | mbarrier count or parity wrong; print mbarrier states with `cuda-gdb` and re-derive from §10.5b |
| **Mbarrier convergence** | added test (see below) | every named mbarrier reaches its expected arrival count exactly once per phase | fix arrival_count or producer/consumer count mismatch |
| **rel_err ≤ 2e-3 vs ref** | `pytest tests/test_rl_kv_megakernel.py` | bf16 numeric tolerance | bisect to first failing tile shape; check core-major SMEM layout + `fence.proxy.async` placement |
| **Boundary stress** (Phase 4 only) | added test on z near grid edges | rel_err on `g_C, g_z` ≤ 2e-3 | confirms saved-state design (§10.3) is correct |
| **Capture-safe** (Phase 6 only) | autograd-graph diff test (§10.13 (e)) | grad match between eager and graph-captured paths | static-buffer or reset-state issue |

**Mbarrier convergence test** (gates kernel before any rel_err run, because hangs and
silent rolling parity desyncs are expensive to debug):

```cpp
// In dev build: instrument every mbarrier with a u32 counter incremented on arrive.
// At kernel exit, single thread copies counters to a global scratch tensor.
// Host side asserts counters match expected_arrivals (per §10.5b table) for ALL phases run.
__device__ uint32_t mbar_arrive_counts[NUM_MBARS];
// arrive() macro: { mbarrier.arrive...; atomicAdd(&mbar_arrive_counts[id], 1); }
// at kernel epilogue: producer thread 0 copies counts to global scratch
```

Then `tests/test_megakernel_mbarrier_convergence.py` runs the kernel and checks each
mbarrier hit its expected count exactly once per phase, **before** running any
correctness test. A drift here (e.g. mbarrier arrived 127 times instead of 128) catches
the silent off-by-one bugs that otherwise show up as either intermittent garbage or
random hangs depending on warp scheduling.

**Strip the instrumentation in prod build** (`-DPROD_BUILD` defines the `arrive()`
macro to drop the atomic counter increment).

**NCU run integrated into CI** — every PR that touches kernel code runs
`ncu --section ComputeWorkloadAnalysis,SourceCounters,LaunchStats --target-processes all`
on the smoke test and parses the JSON for the gates above. CI fails on any gate
violation; no human inspection required.

**References**:
- [PTX 8.5 §9.7.14.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions) — wgmma sm_90a requirement, `wgmma.fence` semantics
- [NCU CLI metrics reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#nvidia-tools-extension-api-nvtx) — `smsp__sass_local_load.sum`, atomic counters, occupancy
- [PyTorch cpp_extension docs](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) — `extra_cuda_cflags` for setting `-arch=sm_90a`

---

## 11. Additional references (for §10 specs)

- [CUDA C++ Programming Guide §4.11 Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) — TMA + mbarrier semantics
- [Colfax CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) — persistent grid + Stream-K reduction patterns
- [Veitner blog: TMA introduction](https://veitner.bearblog.dev/tma-introduction/) — concrete code for `cuTensorMapEncodeTiled` + mbarrier arrive/wait
- [CUTLASS sm90_tile_scheduler.hpp](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_tile_scheduler.hpp) — reference persistent scheduler implementation
- [PyTorch CUDA Graph notes](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) — capture/replay constraints
- [DeepGEMM (DeepSeek)](https://github.com/deepseek-ai/DeepGEMM) — FP8 GEMM reference for Phase 7
