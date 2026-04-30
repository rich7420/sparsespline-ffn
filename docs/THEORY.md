# SparseSpline-FFN — Theoretical Foundation

**Branch:** `jhcg-redesign`
**Date:** 2026-04-30 (literature-grounded revision)
**Working name:** `SparseSpline-FFN` (Sparse + Spline + FFN-specialization).
This is a research label, not a frozen architecture name: whichever candidate
actually wins the validation matrix becomes the final `SparseSpline-FFN`.
**Mission:** Locally-supported B-spline FFN that beats MLP on
quality (match-or-better), VRAM (3-5× less), storage (5-10× less),
and speed (faster end-to-end training step; target 1.3-2.5×).

This document is the master theoretical reference for the redesign. It
starts from first principles (what an FFN computes), expands the full
JHCG lineage (K-A theorem → KAN → JHCG → SparseFuse-JHCG), exposes where
the locality property of B-splines is wasted, and lays out the design
space.

**Revision history:**
- v1 (2026-04-30 morning): Initial theory + 13 sections
- v2 (2026-04-30 afternoon): Part L0 added (drop KAN base branch for FFN)
- v3 (2026-04-30 evening): **Literature-grounded revision** — incorporates
  citations from KAN, PWLU, MoE locality, tensor-decomp LLM compression
  papers; tempers over-optimistic targets; CP → Tucker recommendation;
  realistic speedup numbers from MoE Triton precedent.
- v4 (2026-05-01): **FullMix-Tucker revision** — replaces the old
  `A + Tucker + D` aggressive design with a decoder-folded
  `A + Tucker-readout` design; adds explicit formula expansion and a
  self-review gate.
- v5 (2026-05-01): **Output-rank bottleneck exposure** — algebraic
  expansion of the Tucker readout shows the per-token FFN update is
  constrained to the column space of $U \in \mathbb{R}^{d \times R_o}$
  (i.e., dim $\leq R_o$). This bottleneck **was always present** in the
  Tucker formulation; the simplification exposed it. JHCG had an
  *input-side* bottleneck ($E: d \to d_z$); FullMix-Tucker has an
  *output-side* bottleneck ($U: R_o \to d$). The two are not the same
  pathology, and FullMix-Tucker's is mitigable. Adds Part F.4.b
  (algebraic exposure), Part F.4.c (four mitigation strategies),
  and asymmetric-rank ablation (later renamed `fm_b1_pa6_asym` in v6).
- v6 (2026-05-01): **Implementation strategy + placement axis** —
  formalizes the reference (Phase 1) vs production (Phase 2) split with
  a proof of bit-level equivalence (within bf16 1e-3 tolerance). Adds
  Part K.0 (implementation strategy with proof). Discovers that the
  *cumulative* output-rank coverage scales as $\min(K \cdot R_o, d)$
  across $K$ replaced layers; this makes full replacement (12 layers)
  rank-safer than Pattern A (3 layers) on coverage grounds. Promotes
  **Pattern A+ (6 layers, late-half)** to primary placement (matches the
  m7 reference and has $\min(576, 768) = 576$ cumulative coverage), adds
  **Pattern Full (12 layers)** as stretch goal (full coverage), and
  reclassifies Pattern A (3 layers) as the safety net. Phase 1 cells
  expanded from 6 to 8 with placement-axis structure (`pa3`, `pa6` primary,
  `full_r96` stretch, plus `pa6_r128`, `pa6_asym`, `pa6_direct` ablations,
  `pwlu_baseline` sanity, `mlp_baseline` reference); old single-placement
  cells renamed to placement-suffixed names (e.g. `direct_b1_r96` →
  `fm_b1_pa6_direct`, `fm_b1_asym_256_96` → `fm_b1_pa6_asym`).

---

## Notation

Throughout, $d$ is model dim (768), $T$ is sequence length, $B$ is batch.
$x \in \mathbb{R}^d$ is a single token's residual-stream activation
(post-RMSNorm in nanochat).

`MAC` = multiply-accumulate (1 mul + 1 add ≈ 2 FLOPs). All compute counts
are per-token unless stated.

---

# Part A — What the FFN block actually computes

In transformer, the FFN block per token:

$$x_{l+1} = x_l + \text{FFN}(\text{RMSNorm}(x_l))$$

The FFN is **a parameterized function** $f: \mathbb{R}^d \to \mathbb{R}^d$.
Its job is to add a learned correction to the residual stream that:

1. **Mixes information across channels** — channel $i$ of output may depend on all $d$ input channels (full-rank coupling)
2. **Applies pointwise nonlinearity** — to break linearity and allow non-affine transforms
3. **Has high-rank input/output sensitivity** — late-layer FFNs especially have $\|f(x)\|_\text{RMS} / \|x\|_\text{RMS} \gg 1$ (we measured 2.70 at L11 of nanochat)

The function class needs to be:
- **Universal** (or at least sufficiently expressive)
- **Differentiable** (for gradient training)
- **Computable** in O(d) or O(d²) time per token

Both MLP and KAN-based FFN satisfy these. The question is **inductive bias** and **compute/memory profile**.

---

# Part B — MLP analyzed

## B.1 Architecture

$$y = W_2 \cdot \phi(W_1 x), \quad W_1 \in \mathbb{R}^{4d \times d}, \quad W_2 \in \mathbb{R}^{d \times 4d}$$

Standard $\phi$ choices: ReLU, GELU, ReLU². Nanochat uses **ReLU²**:

$$\phi(z) = \max(z, 0)^2$$

## B.2 What ReLU² actually is

ReLU² is **piecewise quadratic with hard zero**:

$$\phi(z) = \begin{cases} z^2 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Properties:
- **C¹ continuous everywhere** (smooth at z=0: derivative is 2z|_{z=0} = 0)
- **NOT C²** (second derivative is 2 for z>0, 0 for z≤0 → discontinuous at z=0)
- Provides **sharp activation boundary** (hard zero on negative side)
- Provides **magnitude amplification** on positive side ($z^2$ for large $|z|$)

## B.3 MLP compute / storage

| metric | value (d=768) |
|---|---:|
| Storage | $W_1 + W_2 = 2 \cdot 4d \cdot d = 4.7M$ params |
| Active per token | 100% (all 4.7M MACs touched) |
| FLOPs/token | $\sim 9.4M$ FLOPs |
| Hidden activation | $4d \cdot T = 3072 T$ floats per layer |
| VRAM activation footprint (B=1, T=1024) | ~6.3 MB per layer in bf16 |

## B.4 MLP function class

MLP is **universal approximator**: any continuous $f: \mathbb{R}^d \to \mathbb{R}^d$ on a compact set can be approximated to arbitrary accuracy with sufficient hidden width (Cybenko 1989, Hornik 1991).

**Inductive bias of ReLU² MLP:**
- Compositional functions of half-spaces: each hidden unit defines a half-space ($w_i^\top x > 0$), output is a sum of polynomials over union of half-spaces
- Sharp decision boundaries (the "ReLU edge")
- Magnitude amplification (squared activation)

**Why MLP works for token prediction:**
- Token decisions are near-binary ("the" vs "The")
- Late-layer FFN must produce vocab-aligned features → high-magnitude output
- ReLU² provides both sharpness (hard-zero) and magnitude (squaring)

## B.5 ReLU **is** a spline (literature)

A key result from approximation theory:

> _"ReLU activation functions correspond to optimal first-order polynomial splines"_ — Williams 2019 (arxiv 1910.02333). ReLU is mathematically a piecewise-linear (B1) spline with one knot at $x=0$.

> _"For any function expressible in MARS [Multivariate Adaptive Regression Splines] with M parameters, there exists a multilayer neural network with comparable expressivity"_ — Eckle & Schmidt-Hieber 2018 (Neural Networks, ScienceDirect S0893608018303277).

> _"A Spline Theory of Deep Networks"_ — Balestriero & Baraniuk ICML 2018 (proceedings.mlr.press/v80/balestriero18b/) formally establishes: deep ReLU networks compute *piecewise linear* spline functions on input space.

**Consequence for SparseSpline-FFN:**
- ReLU MLP **already is** a spline approximator — KAN-with-B1 is a generalization (per-edge knots vs ReLU's single knot at 0)
- B1 KAN at $G$ knots per edge has *strictly more* knot positions than ReLU MLP (ReLU has 1, B1 has G+1)
- The "smoothness mismatch" we identified for cubic B-spline does NOT apply to B1
- **B1 has the same inductive bias class as ReLU**, plus learnable per-edge knot positions

This gives strong theoretical priors that **B1-basis KAN ≥ ReLU MLP in expressivity per parameter**, modulo the topology choice.

## B.6 MLP weaknesses (theoretically)

1. **No structured sparsity** — every weight contributes to every token
2. **Smoothness mismatch** — for tasks needing C² smooth functions (regression, PDEs), ReLU² is suboptimal
3. **High param count** — universal approximation requires $\Theta(\epsilon^{-d/2})$ params for $\epsilon$ accuracy in d-dim

---

# Part C — KAN (Liu et al. 2024)

## C.1 The Kolmogorov-Arnold Theorem (1957)

**Theorem (Arnold, 1957):** Any continuous $f: [0,1]^n \to \mathbb{R}$ can be represented as

$$f(x_1, \dots, x_n) = \sum_{q=0}^{2n} \Phi_q\!\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

where $\phi_{q,p}, \Phi_q$ are continuous univariate functions.

**Key insight:** A multivariate function decomposes into univariate functions composed with sums. This is structurally different from MLP's "linear combinations passed through nonlinearity."

**Caveats:**
- Existence theorem (does NOT give explicit form)
- $\phi_{q,p}$ in original proof are **non-smooth** (highly irregular)
- Requires exactly $2n+1$ outer functions (rigid)

## C.2 KAN: making K-A learnable

Liu et al. parameterize each $\phi$ as a B-spline:

$$\phi(x) = \sum_i c_i B_i(x) \quad \text{or} \quad \phi(x) = w_b \cdot \text{SiLU}(x) + w_s \cdot \sum_i c_i B_i(x)$$

(The second form is the "base + spline" branch in current SparseFuseJHCG's `kan_branch_mode = base_spline`.)

**KAN topology** for layer with $n_\text{in}$ inputs, $n_\text{out}$ outputs:

$$y_j = \sum_{i=1}^{n_\text{in}} \phi_{ij}(x_i)$$

where each $\phi_{ij}$ is an independent learnable spline.

**Storage:** $n_\text{in} \times n_\text{out} \times (G + k)$ params where $G$ is grid size, $k$ is spline order.

**For LM scale:** $d=768$, $G=20$, cubic ($k=3$): $768 \cdot 768 \cdot 23 = 13.5M$ per layer. **3× MLP.**

## C.3 Why pure KAN is unaffordable for transformers

13.5M per layer × 12 layers = 162M extra params just for FFN, almost 2× model size. Empirically also slow on dense GPU matmul (no obvious GEMM mapping).

→ KAN-based FFN needs **structural compression** to be viable.

---

# Part D — JHCG and SparseFuse-JHCG

## D.1 JHCG-Net (arxiv 2512.05049): hierarchical compression

JHCG-Net (described as encoder-decoder generalized KAN in the QKAN-LSTM/HQKAN
paper, arxiv 2512.05049 — **NOT** arxiv 2406.14026, which is unrelated)
adds linear projections around the KAN core:

$$y = D \cdot \text{KAN}(N(E x)), \quad E: \mathbb{R}^d \to \mathbb{R}^{d_z}, \quad D: \mathbb{R}^{d_o} \to \mathbb{R}^d$$

with $d_z = \rho \cdot d$ for $\rho < 1$ (e.g., $\rho = 1/4$).

**Effect on storage:**
- Original KAN: $d \cdot d \cdot (G+k)$
- JHCG: $E + \text{KAN}(d_z, d_o) + D = d \cdot d_z + d_z \cdot d_o \cdot (G+k) + d_o \cdot d$
- For $\rho=1/4$, $d_o=4d_z=d$: $E + D$ ≈ $1.18M$, $\text{KAN}$ ≈ $3.4M$ → total $\sim 4.6M$
- **Compares with MLP: $4.7M$** — comparable storage, **but 4× compressed in latent space**

## D.2 SparseFuse-JHCG: matrix-fused evaluation

The matrix-fused trick: precompute basis $B(u) \in \mathbb{R}^{d_z \times (G+k)}$, flatten to $\mathbb{R}^{d_z(G+k)}$, then one matmul with reshaped weight $W \in \mathbb{R}^{d_o \times d_z(G+k)}$:

$$y = W \cdot \text{flat}(B(u)), \quad y \in \mathbb{R}^{d_o}$$

This is **GEMM-friendly** but does **NOT exploit B-spline locality** — it computes the full $(G+k)$-wide basis vector even though most entries are 0.

## D.3 The LatentGridNorm

To ensure spline input is in the grid range, JHCG inserts a normalization:

$$z \to z_\text{normed} = z / \|z\|_\text{RMS}, \quad u = \tanh(z_\text{normed} / \tau), \quad z_\text{grid} = \frac{lo + hi}{2} + \frac{hi - lo}{2} u$$

This **destroys magnitude information** (only direction survives RMS-normalize) and **bounds spline input** to $[lo, hi] = [-1, +1]$ default.

## D.4 Current SparseFuse-JHCG profile

| metric | value (d=768, ρ=1/4, G=20) |
|---|---:|
| Storage | $\approx 4.1M$ params/layer (less than MLP's 4.7M) |
| Active per token | $\approx 590K$ (spline) + $590K$ (E+D) = $1.18M$ ≈ **30%** of dense (but dense matmul still computes all) |
| **Computed FLOPs/token** | $\sim 8.2M$ (close to MLP because dense kernel doesn't skip zeros) |
| Hidden activation | $d_z + d_o = 192 + 768 = 960$ floats |
| VRAM activation footprint | ~2 MB/layer (bf16) |

**Storage and VRAM advantage: real (small but real).**
**Speed advantage: NONE** (kernel doesn't exploit locality).
**Quality gap: structural (+0.07 nat at 100M tokens on nanochat).**

---

# Part E — B-spline Locality: the unused gold mine

## E.1 Cox-de Boor recursion

B-spline basis of degree $k$ on uniform knots $\{t_i = ih\}$:

$$B^0_i(x) = \mathbb{1}[t_i \leq x < t_{i+1}]$$

$$B^k_i(x) = \frac{x - t_i}{t_{i+k} - t_i} B^{k-1}_i(x) + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B^{k-1}_{i+1}(x)$$

## E.2 Fundamental locality property

**Theorem:** $B^k_i(x) \neq 0$ iff $x \in [t_i, t_{i+k+1})$.

**Corollary:** For any input $x$, **at most $k+1$ basis functions are non-zero**.

For our settings (G grid points, support on $[lo, hi]$):
- Number of basis functions: $G + k$ (we use $G+3$ for cubic)
- Active per input: $k + 1$
- **Active ratio:** $(k+1) / (G+k)$

| basis | order $k$ | active per input | active ratio (G=20) |
|---:|:---:|---:|---:|
| Step (B0) | 0 | 1 | 5.0% |
| **Linear (B1)** | 1 | 2 | **9.5%** |
| Quadratic (B2) | 2 | 3 | 13.6% |
| **Cubic (B3, current)** | 3 | 4 | **17.4%** |

## E.3 The waste: dense evaluation of sparse basis

For input $u \in \mathbb{R}^{d_z}$, computing $y \in \mathbb{R}^{d_o}$ via cubic spline:

**Dense (current MatrixFusedKANLinear):**
1. Compute basis: $\mathbf{B}(u) \in \mathbb{R}^{d_z \times (G+3)}$ (all 23 values per input dim)
2. Flatten: $\mathbf{b} \in \mathbb{R}^{d_z (G+3)}$
3. Matmul: $y = W \mathbf{b}$, $W \in \mathbb{R}^{d_o \times d_z(G+3)}$

FLOPs:
- Basis evaluation: $d_z \cdot (G+3) \cdot O(G) \approx d_z \cdot 23 \cdot 20 \approx 88K$
- Matmul: $d_o \cdot d_z \cdot (G+3) \approx 768 \cdot 192 \cdot 23 = 3.4M$ MACs $\approx 6.8M$ FLOPs

**Total: ~6.8M FLOPs/token (close to MLP's 9.4M).**

**Locality-aware (theoretical):**
1. Compute bin index: $b_j = \lfloor (u_j - lo) / \text{step} \rfloor$, $d_z$ floor ops
2. Evaluate $k+1=4$ active basis per input: $d_z \cdot 4 \cdot O(k) \approx d_z \cdot 4 \cdot 3 = 2.3K$ (40× less than dense)
3. Sparse matmul: gather 4 columns of $W$ per input dim, do reduced matmul: $d_o \cdot d_z \cdot 4 = 590K$ MACs $\approx 1.2M$ FLOPs

**Total: ~1.2M FLOPs/token (5.7× less than current dense, 7.8× less than MLP).**

→ **Locality is a 5-8× compute saving the current implementation throws away.**

## E.4 Active vs stored params

| design | total params | active per token | active ratio |
|---|---:|---:|---:|
| MLP (4d hidden) | 4.7M | 4.7M | 100% |
| Current JHCG (ρ=1/4, dense kernel) | 4.1M | 4.1M (computed) / 1.2M (active) | 100% / 30% |
| Locality-aware JHCG (ρ=1/4) | 4.1M | **1.2M** | **30%** |
| Locality-aware **direct-d KAN** (no mixer) | 13.5M | **2.4M** | **17.4%** |
| Locality-aware **T_mixer KAN** (m=d, A learned) | 13.5M + d² ≈ 14.1M | **2.4M + d² ≈ 3.0M** | **~21%** |
| **FullMix-Tucker (m=d, R₁=R₂=64, R₃=16)** | **754K** | **1.52M MACs** | **32% of MLP MACs** |
| **FullMix-Tucker (m=d, R₁=R₂=96, R₃=16)** | **885K** | **2.02M MACs** | **43% of MLP MACs** |

**T_mixer locality-aware KAN already has lower active compute than MLP, and restores MLP's learned-half-space inductive bias** (which T_direct gives up — see Part I.1).

---

# Part F — CP/Tucker decomposition: managing storage

## F.1 The 13.5M storage problem

If we drop encoder/decoder (to solve Defect 1), spline_weight becomes $W \in \mathbb{R}^{d \times d \times L}$ where $L = G + k$ is the number of B-spline basis functions (cubic: $L = G+3 = 23$; B1: $L = G+1 = 21$). For cubic at $G = 20$, $d = 768$: $W$ = 13.5M per layer.

Storage 3× MLP. While compute is still less, params bloat is unattractive.

## F.2 CP decomposition

$$W_{kji} = \sum_{r=1}^R u^{(r)}_k \cdot v^{(r)}_j \cdot w^{(r)}_i$$

Storage: $R \cdot (d + d + L) = R \cdot (2d + G+k) = R \cdot (1536 + 23) \approx R \cdot 1559$ for cubic, $R \cdot 1557$ for B1.
- $R = 64$: **100K per layer (47× less than MLP's 4.7M)**
- $R = 128$: **200K per layer (24× less)**
- $R = 256$: **400K per layer (12× less)**

## F.3 Active compute under CP + locality

For input $u$ with bin indices $b_j$ and active basis values $B_i(u_j)$ for $i \in \{b_j-k, ..., b_j\}$:

$$y_k = \sum_r u^{(r)}_k \cdot \underbrace{\sum_j v^{(r)}_j \cdot \underbrace{\sum_{i \in \text{active}} w^{(r)}_i B_i(u_j)}_{\text{per-(r, j) contraction over active basis}}}_{\text{contract over input dim}}$$

Step-by-step compute per token:
1. **Active basis eval:** $d \cdot (k+1) \cdot O(k) = d \cdot 4 \cdot 3 = 9.2K$ FLOPs
2. **Inner reduction (active basis × $w^{(r)}$):** $R \cdot d \cdot (k+1) = R \cdot d \cdot 4$ MACs
3. **Middle reduction (over input dim):** $R \cdot d$ MACs
4. **Outer (over rank, project to output):** $R \cdot d$ MACs

Total: $R \cdot d \cdot (k+3) = R \cdot d \cdot 6$ MACs (for cubic)

For R=64, d=768: $64 \cdot 768 \cdot 6 = 295K$ MACs $\approx 590K$ FLOPs.
**16× less compute than MLP.**

## F.4 Tucker decomposition

If CP rank-R limits expressivity, Tucker is more flexible:

$$W = \mathcal{C} \times_1 U \times_2 V \times_3 W_b$$

where $\mathcal{C} \in \mathbb{R}^{R_1 \times R_2 \times R_3}$ is core, $U: d \to R_1$, etc.

Storage: $R_1 R_2 R_3 + R_1 d + R_2 d + R_3 L$ where $L = G + k$.
For $R_1 = R_2 = 64, R_3 = 16$, cubic ($L=23$): $64^2 \cdot 16 + 64 d \cdot 2 + 16 \cdot 23 \approx 65K + 100K + 0.4K = 166K$ per layer.

Tucker has higher representational floor than CP at same rank, more flexible.

### F.4.a FullMix-Tucker: fold the decoder into the Tucker readout

The first aggressive `T_mixer` sketch used:

$$z = A x,\quad h = \text{KAN}_\text{Tucker}(z),\quad y = D h$$

This was conceptually clean but system-inefficient: the two dense matrices
$A: d \to m$ and $D: m \to d$ dominate both storage and MACs. The stronger
design folds the decoder into the Tucker output factor:

$$z = A x,\quad y_k = \gamma \sum_j \sum_i W_{kji} B_i(z_j)$$

with

$$W_{kji} = \sum_{a=1}^{R_o}\sum_{b=1}^{R_i}\sum_{c=1}^{R_b}
U_{ka}\, C_{abc}\, V_{jb}\, Q_{ic}.$$

There is no separate $D$. The output factor $U \in \mathbb{R}^{d \times R_o}$
is the readout.

For B1, $L=G+1$ and the active basis count is $s=2$. Per replaced layer:

$$P_\text{FM} = d m + dR_o + mR_i + LR_b + R_oR_iR_b$$

$$\text{MAC}_\text{FM} =
dm + smR_b + mR_iR_b + R_oR_iR_b + dR_o.$$

At $d=m=768$, $G=20$, $L=21$, $R_b=16$:

| rank | params | storage win vs MLP | MACs/token | MAC win vs MLP |
|---:|---:|---:|---:|---:|
| $(64,64,16)$ | 754K | 6.26× | 1.52M | 3.11× |
| **$(96,96,16)$** | **885K** | **5.33×** | **2.02M** | **2.34×** |
| $(128,128,16)$ | 1.05M | 4.50× | 2.55M | 1.85× |

**Recommendation:** use $(96,96,16)$ as the primary all-win candidate. It is
the smallest rank that still clears the storage target (>5× less than MLP)
while leaving more quality headroom than rank 64. Rank 64 is the speed/params
ablation; rank 128 is the quality rescue if rank 96 underfits.
**See F.4.b for an output-rank caveat that this naming hides.**

### F.4.b — Output-rank bottleneck (algebraic exposure)

The 5-stage simplification of F.4.a (mixer → spline lookup → $V$ contraction →
core contraction → $U$ projection) makes one structural property visible
that the index-soup form $W_{kji} = \sum_{abc} U_{ka} C_{abc} V_{jb} Q_{ic}$
hides. Note that the output index $k$ enters **only through** the factor $U$:

$$y_k = \sum_a U_{ka} \cdot \underbrace{\Big[\sum_{b,c,j,i} C_{abc} V_{jb} Q_{ic} B_i(z_j)\Big]}_{\eta_a(x)\, \in\, \mathbb{R}^{R_o}}$$

So the per-token FFN update factors as

$$\boxed{\; y \;=\; U\, \eta(x), \qquad U \in \mathbb{R}^{d \times R_o},\; \eta(x) \in \mathbb{R}^{R_o}\;}$$

Therefore, **for any single token, the FFN update is constrained to the
column space of $U$**:

$$\dim\big(\mathrm{span}\{y(x) : x \in \mathbb{R}^d\}\big) \;\leq\; R_o.$$

For our default rank choice $R_o = 96$ at $d = 768$, the per-layer update lives
in at most a 96-dimensional subspace of the 768-dimensional residual stream.
MLP has no such restriction: its $W_2 \in \mathbb{R}^{d \times 4d}$ has full
output rank $d$.

**This bottleneck was already present in F.4.a** — the simplification did not
introduce it. It is a direct algebraic consequence of Tucker decomposition
with $R_o < d$. The simplification only made it visible.

#### Comparison to JHCG's bottleneck

JHCG's information bottleneck (Defect 1) is on the **input side**: the
encoder $E: d \to d_z$ irrecoverably destroys input directions before the
spline ever sees them. FullMix-Tucker's bottleneck is on the **output side**:
the input is preserved (mixer $A$ is non-compressive, $m \geq d$) and the
spline operates on the full $m$-channel learned-coordinate representation,
but the readout $U: R_o \to d$ projects from a smaller subspace.

| pathology | location | reversibility |
|---|---|---|
| JHCG bottleneck | input-side ($E$ before spline) | **none** — destroyed information cannot be recovered |
| FullMix-Tucker bottleneck | output-side ($U$ after spline) | **per-layer rank limit only** — adjacent MLP layers can route what this layer cannot |

These are not the same pathology. Output-side rank limits are *recoverable*
across layers under the standard residual-stream assumption: even if layer
$\ell$'s update lives in a 96-dim subspace, layer $\ell+1$'s subspace can be
different, and Pattern A (alternating MLP layers between FullMix-Tucker
layers) guarantees full-rank coverage across the stack. JHCG's input-side
loss has no such recovery path.

#### What this means for the empirical question

The central paper-worthy question becomes:

> *Are FFN per-layer updates in a trained transformer naturally low-rank?
> If yes, FullMix-Tucker rank 96 should match MLP. If no, we need to lift $R_o$.*

There is real precedent for "yes": LLM intermediate activations are widely
observed to be approximately low-rank (ESPACE NeurIPS 2024 hits 50%
compression with 0.18 PPL increase; LoRA rank 8-64 works for adaptation;
attention head outputs concentrate in a few singular directions). Per-layer
*FFN-update* rank is less directly studied but expected to follow.

But this is no longer "FullMix-Tucker vs MLP on quality" — it is "what is
the empirical rank of the FFN update signal." That is a sharper, more
testable question, and the next subsection lists four mitigations if the
answer is "higher than 96."

### F.4.c — Mitigations for the output-rank bottleneck

Four strategies, ordered from least to most architectural change:

#### Strategy A — Asymmetric rank (boost $R_o$ relative to $R_i$)

The bottleneck is on the output side, so spend rank there. Instead of
$(R_o, R_i, R_b) = (96, 96, 16)$, try:

$$(R_o, R_i, R_b) = (256, 96, 16)$$

Storage cost (compared to symmetric rank-128):

| design | params | output rank |
|---|---:|---:|
| symmetric $(96, 96, 16)$ | 885K | 96 |
| symmetric $(128, 128, 16)$ | 1.05M | 128 |
| **asymmetric $(256, 96, 16)$** | **1.04M** | **256** |
| symmetric $(256, 256, 16)$ | 1.84M | 256 |

The asymmetric variant gives output rank 256 at the same storage cost as
symmetric rank 128, because $C \in \mathbb{R}^{R_o \times R_i \times R_b}$
costs $R_o \cdot R_i \cdot R_b$, and we are doubling only one factor.

**This should be the primary mitigation tested.** It costs almost nothing
extra and directly targets the diagnosed pathology.

#### Strategy B — Multi-head output

Split the output into $H$ heads, each with its own $U_h$ and $\eta_h$:

$$y = \sum_{h=1}^H U_h\, \eta_h(x), \qquad U_h \in \mathbb{R}^{d \times R_o^{(h)}}.$$

Total output rank $\leq \sum_h R_o^{(h)}$, but with structured sharing of
the inner factors $V$, $Q$. Implementation-wise this is a small refactor of
the einsum `eta = einsum("tbc, abc -> ta", xi, C)` to carry an extra head dim.

Cost: roughly proportional to $H$. For $H = 4$ with per-head $R_o^{(h)} = 64$,
total output rank = 256, total params slightly higher than asymmetric A.
B is preferred over A only if multi-head structure also improves
generalization (e.g., heads specialize on different tasks).

#### Strategy C — Full-rank cheap residual linear

Add a dense $W_0: d \to d$ in parallel:

$$y = W_0 x + U\, \eta(x).$$

$W_0$ is full-rank (output rank = $d$), so the bottleneck is gone. Storage
cost: $d^2 = 590K$ params per layer — significant but smaller than MLP's
$8 d^2 = 4.7\text{M}$.

This is conceptually clean: "sparse spline branch + dense linear escape
path." But it changes the paper story slightly — no longer "pure
SparseSpline-FFN" but "SparseSpline-FFN with linear residual." Some
reviewers will see this as honest; others as moving the goalposts.
$W_0$ can also be initialized to zero so that early training matches the
pure variant.

#### Strategy D — Small dense MLP escape hatch (Part M.4)

$$y = U\, \eta(x) + \alpha \cdot W_2'\,\mathrm{ReLU}^2(W_1' x), \quad W_1': d \to h,\; h \ll 4d.$$

This is heavier than C but adds back ReLU²'s squared-magnitude curvature
(which B1 lacks), so it covers two risks at once. $\alpha$ initialized
near zero. Use only if A/B/C fail.

#### Recommendation

Phase 1 sweep order:

1. Run `fm_b1_pa6` (primary, symmetric, K=6 late-half).
2. If it underfits: run `fm_b1_pa6_asym` (Strategy A) — same storage as
   `fm_b1_pa6_r128`, but with 2.7× the output rank.
3. If asymmetric also underfits: try Strategy C (linear residual) before D
   (MLP escape) because C is one extra dense layer; D is two.

If even C fails, the FullMix-Tucker topology likely cannot replace MLP
for this task and we should revisit the architecture (Part M.4).

### F.4.d — Ablation ladder revealed by the simplification

A side benefit of the 5-stage form (F.4.a/b) is that several FullMix-Tucker
hyperparameters degenerate FullMix-Tucker into other architectures that
already exist in the literature. This gives a **clean, interpretable
ablation ladder** where each rung corresponds to a published baseline:

| ablation | constraint | equivalent architecture | published baseline |
|---|---|---|---|
| **L0**: pure activation only | $R_b = 1$, $A = I$ (identity), $V$ identity, $C = I_{R_o}$, $U = I$ at $R_o = d$ | MLP with **scalar** learnable spline activation | **PWLU** (ICCV 2021) |
| **L1**: + learned mixer | $R_b = 1$, $A: d \to m$ learned, $V = I$, $C = I_{R_o}$, $U$ low-rank | MLP with mixer + scalar spline activation | (no direct equivalent) |
| **L2**: + low-rank $W_2$ | $R_b = 1$, $A$ learned, full Tucker on $V, C, U$ | LoRA-style **low-rank MLP** + scalar spline activation | **LoRA-MLP + PWLU** |
| **L3**: + vector activation | $R_b > 1$, full FullMix-Tucker | full FullMix-Tucker | **this paper** |
| **L3-asym**: + asymmetric rank | L3 with $R_o > R_i$ | output-rank-rescued FullMix-Tucker | **this paper, Strategy A** |

Why this is paper-relevant:

1. **Reviewer-friendly story.** Each rung adds exactly one architectural
   axis, and the relevant published baseline is named. A reviewer can
   trace the contribution from PWLU forward without ever needing to know
   the K-A theorem or the JHCG lineage.

2. **Honest contribution attribution.** L0 ≈ PWLU isolates the value of
   "spline activation alone". L1 → L2 adds the learned mixer ablation.
   L2 → L3 isolates the value of *vector* over scalar activation. L3 → L3-asym
   isolates the output-rank fix. Each step's $\Delta\text{val}$ is its own
   independent claim.

3. **Cheap diagnostic cells.** L0 (PWLU) and L2 (LoRA-MLP + scalar
   spline) are *literature-existing* baselines — they should match
   published numbers within noise. If they don't, it is a
   bug-detection signal (init wrong, grid wrong, or PyTorch op wrong)
   *before* the FullMix-Tucker numbers can be trusted.

We will include **at least one rung-degeneracy cell** in the Phase 1 sweep
(see K.1). The cheapest diagnostic is `pwlu_baseline` — set $R_b = 1$,
$A = I$, $V = I$, $C = I$, $R_o = d$, and verify that the remaining
1D-spline-activation MLP matches PWLU's published GPT-2-small numbers
(within $\pm 0.02$ nat). If yes, the spline pipeline is verified. If no,
we have a bug to fix before interpreting any FullMix-Tucker result.

## F.5 Risks of decomposition (literature-grounded)

> _"CP decomposition has not been efficient for compressing CNNs due to the **CP instability issue**, which often impairs fine-tuning after decomposition. Specifically, fitting the convolutional tensors by numerical optimization algorithms often encounters diverging components, i.e., extremely large rank-one tensors but canceling each other, which often causes non-interpretable results and **numerical instability for neural network fine-tuning**."_ — *Stable Low-rank Tensor Decomposition for Compression of CNN* (ECCV 2020, arxiv 2008.05441).

> _"Tucker decomposition is beneficial for latency and model size reduction... rank can be pruned down to 1 with minimal accuracy impact, **early layers should not be decomposed, and adjacent layers should not be decomposed**."_ — *Characterizing the Accuracy-Efficiency Trade-off of Low-Rank Decomposition LLMs* (arxiv 2405.06626, 2024).

> _"ESPACE achieves 50% compression of GPT3, Llama2, and Nemotron4 models with small accuracy degradation, **as low as 0.18 perplexity increase on GPT3-22B**."_ — *ESPACE: Dimensionality Reduction of Activations for Model Compression* (NeurIPS 2024).

**Implications for SparseSpline-FFN:**

1. **Use Tucker, not CP, as default decomposition.** CP is unstable for fine-tuning per ECCV 2020. Tucker has more degrees of freedom (per-mode rank) and known stable optimization.

2. **Don't decompose adjacent layers.** Our `late_half = L6-L11` is 6 adjacent layers. Per literature this violates best practice.

3. **Realistic compression target: 5-10×, not 20-50×.** ESPACE's 50% (≈ 2× compression) with 0.18 PPL is the established baseline for LLM tensor compression. Aiming for 23× compression with ≤ 0.02 nat (≈ 0.02 PPL) increase is **aggressive vs literature**.

**Recommended placement patterns** (revised v6, see F.5.1 below for the
cumulative output-rank argument that drives the revision):

| pattern | applied layers | K (KAN layers) | cumulative output rank $\min(K R_o, d)$ at $R_o=96$ | role |
|---|---|---:|---:|---|
| **A: Alternating** | L6, L8, L10 | 3 | 288 / 768 (38%) | safety net |
| **A+: Late-half** | L6–L11 | **6** | **576 / 768 (75%)** | **primary; matches v3.5 m7 placement** |
| **Full: All layers** | L0–L11 | 12 | 768 / 768 (100%) | stretch goal — paper headline target |
| B: Sparse | L8, L11 | 2 | 192 / 768 (25%) | extreme conservative |
| C: Last only | L11 | 1 | 96 / 768 (13%) | unit test |

**Default for v6 redesign:** Pattern A+ (late-half, K=6) as primary.
Rationale per F.5.1: A+ matches the v3.5 m7 6-layer placement so we can
directly claim "FullMix-Tucker breaks the +0.07 nat ceiling that capped m7";
its cumulative output-rank coverage of 576/768 leaves the early-half MLP
layers responsible for only the remaining 25% of the residual stream's
update directions, which keeps the literature warning about adjacent-layer
decomposition (arxiv 2405.06626) at a manageable 6-layer span.

Pattern A is reclassified from primary to **safety net**: its 38% cumulative
output-rank coverage relies heavily on MLP layers to route the missing
directions, which both weakens the paper claim and (counterintuitively)
makes per-layer output-rank the binding constraint.

Pattern Full is the **stretch goal**: at 12 layers, cumulative coverage
reaches the full $d$, so the residual stream's update directions are not
limited by any single $U_\ell$ — the bottleneck dissolves at the model
level even if it persists at the per-layer level. This is the
architecturally cleanest claim and the strongest paper headline if it works.

### F.5.1 Cumulative output-rank coverage (the placement argument)

Per F.4.b, each replaced layer contributes a per-layer FFN update inside
the column space of its $U_\ell \in \mathbb{R}^{d \times R_o}$. After
$K$ replaced layers, the union of their column spaces

$$\mathcal{S}_K \;:=\; \bigcup_{\ell\in\text{KAN layers}} \mathrm{Col}(U_\ell)$$

has dimension at most $K \cdot R_o$ but never more than $d$:

$$\dim(\mathcal{S}_K) \;\leq\; \min(K \cdot R_o,\; d).$$

Generically, with diverse training data and standard SGD, the inequality is
tight — different layers see different input distributions and learn
different output bases. The literature on attention-head subspace
specialization and on LoRA-rank diversity supports this.

**Implication for placement:**

| K | $\min(K R_o, d)$ at $R_o=96, d=768$ | meaning |
|---:|---:|---|
| 1 | 96 (13%) | one $U_\ell$ alone covers a tiny slice |
| 3 | 288 (38%) | Pattern A — MLP layers must route 62% |
| 6 | 576 (75%) | **Pattern A+** — MLP layers route 25% |
| 8 | 768 (100%) | minimum K to saturate $d$ |
| 12 | 768 (100%) | Pattern Full — saturated, no MLP needed for rank |

**This reverses one common intuition.** Naively one might think
"more KAN layers = more output-rank pressure on the model," and prefer
Pattern A. But the math shows the opposite: Pattern Full has fewer
output-rank concerns *at the model level* than Pattern A, because the union
of 12 different $U_\ell$ subspaces saturates the residual stream. Pattern A
is rank-safer *per layer* (because MLPs route what KANs can't) but
rank-tighter *cumulatively* (only 38% coverage from KAN layers themselves).

**Caveat — diversity of $U_\ell$.** The cumulative bound assumes the
$U_\ell$'s are linearly independent. Phase 1 must check this: log
$\sigma_{\min}(\,[\,U_1\ |\ U_2\ |\ \dots\ |\ U_K\,])$ at the end of training.
If it collapses (i.e., layers learn redundant subspaces), Pattern Full
loses its rank-safety advantage and degenerates toward Pattern A's per-layer
bound.

For LM at 100M tokens, FullMix-Tucker rank-(96, 96, 16) is the
literature-grounded starting point. Rank 64 is the fast/lean ablation; rank
128 is the symmetric quality rescue; asymmetric (256, 96, 16) is the
output-rank rescue (F.4.c). Cumulative-rank analysis above shows that
placement choice (K=3 vs 6 vs 12) and per-layer rank choice are **two
orthogonal axes** — both should be ablated.

---

# Part G — Full FLOP/VRAM/storage comparison table

All values per layer at d=768, T=1024, B=1.

| design | storage | active compute/token | FLOPs/token | activation VRAM (bf16) |
|---|---:|---:|---:|---:|
| **MLP** (4d hidden) | 4.7M | 4.7M | 9.4M | 6.3 MB |
| Current JHCG (ρ=1/4, dense kernel) | 4.1M | 4.1M (computed) | ~8.2M | 1.9 MB |
| **Locality-aware JHCG** (ρ=1/4, sparse kernel) | 4.1M | 1.2M | **2.4M** | 1.9 MB |
| **Direct-d KAN, locality-aware** (no mixer, B1) | 13.5M | 2.4M | 4.7M | ~1.5 MB |
| **T_mixer KAN, locality-aware** (m=d, B1) | ~14.1M | ~3.0M | ~5.9M | ~1.5 MB |
| **FullMix-Tucker (64,64,16) + B1** | **754K** | 1.52M MACs | **3.03M FLOPs** | ~1.5 MB |
| **FullMix-Tucker (96,96,16) + B1** | **885K** | 2.02M MACs | **4.03M FLOPs** | ~1.5 MB |
| FullMix-Tucker (128,128,16) + B1 | 1.05M | 2.55M MACs | 5.10M FLOPs | ~1.5 MB |
| Direct-d KAN + CP rank-64 (ablation, unstable per ECCV 2020) | 100K | 295K | 590K | ~1.5 MB |

**Targets:** $<$ MLP on every column.

**Achievable (theoretically):**
- Storage: ✓ FullMix-Tucker rank-96 wins by 5.3× while keeping a learned mixer
- Active compute: ✓ rank-96 uses 43% of MLP MACs
- FLOPs: ✓ rank-96 is ~2.3× lower than MLP at the per-layer math level
- VRAM: ✓ rank-96 saves the $4d$ MLP hidden activation and keeps only the
  $m=d$ mixed activation (about 4× smaller before recomputation tricks)

→ **The math says FullMix-Tucker can strictly beat MLP on storage, activation
VRAM, and arithmetic while retaining MLP's learned oblique coordinate system.**

The remaining question is **quality** — can it match MLP's quality at this
rank? That's the empirical risk, and per F.4.b the dominant axis of that
risk is the **output rank $R_o$**, not the inner rank or basis rank. This
motivates the asymmetric design point $(R_o, R_i, R_b) = (256, 96, 16)$ as
the first rescue (same storage as symmetric r128, 2.7× the output rank).

---

# Part H — Where current JHCG wastes potential

Cataloging every gap between current JHCG and the theoretical optimum:

## Gap 1: Dense kernel ignores locality (5-8× compute waste)
- Current `MatrixFusedKANLinear` does dense matmul over the full $L = G + k$ basis dim
- 83% of those entries are exact zeros for cubic
- **Fix:** Triton kernel with explicit bin-index gather

## Gap 2: Encoder/decoder bottleneck (Defect 1: information bottleneck)
- $E: d \to d_z$ destroys $d - d_z$ linearly independent directions
- Full-rank reconstruction by $D$ is impossible
- **Fix (theoretical):** drop the compressive bottleneck; use FullMix-Tucker
  so the learned mixer is non-compressive and the readout is low-rank

## Gap 3: Cubic basis overkill for sharpness (Defect 2: smoothness)
- Cubic B-spline is C² smooth; LM needs C¹ (or even C⁰) decisions
- Cubic uses 4 active basis; B1 uses 2 (2× cheaper)
- **Fix:** linear B1 basis (or hybrid with hard transitions)

## Gap 4: tanh-squash kills magnitude (Defect 3)
- $u = \tanh(z/\tau) \in (-1, +1)$ → $\|z\|$ information lost
- Late layers need high-magnitude output (L11 ratio 2.70)
- **Fix:** RMSNorm + adaptive grid range OR scale-and-shift (no saturation)

## Gap 5: Param count not actually compressed by KAN
- Currently $4.1M \approx 4.7M$ (MLP) — JHCG isn't even smaller in storage!
- The "compression" was supposed to come from $\rho$, but encoder/decoder eat it back
- **Fix:** Tucker decomposition gives 4.5-6.3× compression at the recommended
  FullMix ranks while retaining a learned mixer

## Gap 6: No exploitation of "active basis = MoE-light" property
- Different inputs activate different spline regions
- This is implicit conditional computation
- Could be made explicit: per-bin cached optimization, weight pruning per region
- **Future direction:** dynamic kernel dispatch per bin

## Gap 7: Output scale not learned
- `output_scale` is fixed at 1.0 (or hardcoded per layer in v4.0)
- A learnable per-layer (or per-channel) gain would auto-tune ffn/res ratio
- **Fix:** learnable scalar gain on JHCG output

---

# Part I — Design Space for Locality-Optimal JHCG

Each design is parameterized by a few key choices:

## I.1 Topology
| option | front-end | back-end | dim policy | rationale |
|---|---|---|---|---|
| **Current (T_jhcg)** | $d \to d_z$ linear ($d_z < d$) | $d_o \to d$ linear | compressive ($\rho < 1$) | original SparseFuse-JHCG |
| **Direct-d (T_direct)** | identity (use $x$ as $z$) | identity | $z = x$, no learned mixer | maximum bandwidth, **but no learned coordinate system** |
| **Mixer (T_mixer)** | $z = A x$, $A: d \to m$ with **$m \geq d$** | usually $D: m \to d$ linear | non-compressive learned mixer | preserves MLP's learned oblique coordinate system |
| **FullMix-Tucker** | $z = A x$, $A: d \to m$ with **$m \geq d$** | Tucker output factor $U$ (no separate $D$) | non-compressive mixer + low-rank readout | **primary all-win candidate** |
| **Hybrid (T_hybrid)** | shallow $d \to d_z$ with $d_z = d/2$ | shallow $d_o \to d$ | $\rho = 1/2$ | middle ground |

**Why T_mixer rather than T_direct as the aggressive design point:**

The MLP's $W_1: \mathbb{R}^d \to \mathbb{R}^{4d}$ does **two distinct jobs simultaneously**:

1. *Capacity expansion* ($4 \times$ width).
2. *Learned oblique coordinate system* — channels of the hidden activation are linear combinations of input channels, NOT the canonical basis. Each ReLU/ReLU² then fires on a *learned half-space* $\{x : w_i^\top x > 0\}$, not on a canonical input dim.

A pure-locality KAN with $T_\text{direct}$ skips step 2: knot-firing happens on **canonical input dims** $x_j$, not on learned linear combinations. This is a strict expressivity loss vs MLP — the spline basis can only carve along axis-aligned directions.

$T_\text{mixer}$ restores step 2 by inserting a learned linear mixer $A: d \to m$ ($m \geq d$, **non-compressive**, so no information bottleneck) **before** the per-coordinate splines. The splines then fire on $z = A x$, which is a learned-rotation/expansion of the input. This is the KAN analog of "learned half-space activation" and is necessary for matching MLP's inductive bias.

**Recommendation:** make FullMix-Tucker the primary aggressive design; reserve
$T_\text{direct}$ as an ablation to confirm the mixer matters.

## I.2 Basis order
| option | k | active per input | smoothness | sharpness ability |
|---|:---:|---:|---|---|
| Step (B0) | 0 | 1 | discontinuous | maximal sharp; no gradient |
| **Linear (B1)** | 1 | 2 | C⁰ | sharp at G+1 knots |
| Quadratic (B2) | 2 | 3 | C¹ | smoother, less sharp |
| **Cubic (B3, current)** | 3 | 4 | C² | smoothest, no sharp |

**B1 is the sweet spot:** sharp transitions (matches MLP's ReLU edge) + **C⁰ continuous, not strictly differentiable at the G+1 knots, but a.e. differentiable and subgradient-trainable in the same way ReLU is** + locality-2.

## I.3 Tensor decomposition
| option | storage | rank | risk |
|---|---|---|---|
| None (full storage) | $d^2 \cdot L$ where $L = G + k$ | — | high storage |
| **CP rank-R** | $R \cdot (2d + L)$ | $R$ | underexpressive at low $R$, **CP instability per ECCV 2020** |
| **Tucker $(R_1, R_2, R_3)$** | $R_1 R_2 R_3 + ...$ | flexible | hyperparameter tuning |
| Block-Tucker (per layer-l) | varies | varies | research level |

## I.4 Normalization
| option | preserves ‖z‖? | grid range | extrapolation? |
|---|:---:|---|---|
| Current LatentGridNorm (RMS+tanh) | ✗ | fixed $[-1, +1]$ | no |
| **RMSNorm only** | ✗ direction-only | bound by $\sigma$ of post-RMS | no needed |
| **RMSNorm + learnable scale** | ✓ scale | adaptive | optional extrapolation |
| Identity (no norm) | ✓ | unbounded | extrapolation required |

## I.5 Output gain
| option | flexibility | risk |
|---|---|---|
| Fixed 1.0 (current) | none | model down-regulates |
| **Per-layer learnable scalar** | layer-specific magnitude | mild |
| Per-channel learnable | channel-specific | overfits |
| Per-channel + softplus parameterization | always positive | safe |

---

# Part J — Theoretical Performance Limits

## J.1 Two design points (literature-tempered)

### J.1.a Conservative (`SparseSpline-FFN-CONS`) — high success rate

Aim: 3 of 4 axis win, with quality match-or-better.

```
Topology       Keep encoder/decoder (T_jhcg) at ρ=1/4
Basis          B1 linear (k=1, locality 2/(G+1) = 9.5%)
Decomposition  None (full storage on bottlenecked tensor, ~3.4M)
Normalization  RMSNorm + learnable scale (no tanh)
Grid           Adaptive [lo, hi] from EMA of z distribution
Output gain    Per-layer learnable scalar
Placement      Pattern A (L6, L8, L10 KAN; L7, L9, L11 MLP) [Conservative]
Kernel         Fused Triton: bin-index → 2-active gather (no CP)
```

### J.1.b Aggressive (`FullMix-Tucker`) — primary all-win candidate

Aim: 4 of 4 axis blowout win, with quality bet.

```
Topology       T_mixer: z = A x with A: d -> m, m >= d
                 (non-compressive learned mixer, restores MLP's
                  learned-half-space inductive bias).
                 No separate decoder D: the output readout is folded into
                 the Tucker output factor U.
                 Default m = d. m = 2d is a quality rescue, not the default.
Basis          B1 linear (locality 2 / (G+1) = 9.5%) on z
Decomposition  Tucker readout on W[k,j,i]
                 primary rank (R_o, R_i, R_b) = (96, 96, 16)
                 fast ablation (64, 64, 16)
                 symmetric rescue (128, 128, 16)
                 output-rank rescue (256, 96, 16) -- see F.4.b/c
                 — NOT CP (per ECCV 2020 instability finding)
Normalization  RMSNorm + learnable scale on z (no tanh squash)
Grid           Adaptive [lo, hi] from EMA on z
Output gain    Per-layer learnable scalar on y
Placement      Pattern A+ (late-half: L6-L11 all KAN, L0-L5 all MLP)
                 [primary; matches v3.5 m7 placement;
                  cumulative output-rank coverage 576/768 = 75%]
                 Pattern Full (all 12 layers) is the stretch goal
                 Pattern A (3 layers, L6/L8/L10) is the safety net
Kernel         Fused Triton: dense matmul A (BLAS-friendly)
                 -> B1-locality gather (k+1=2 active per dim)
                 -> Tucker readout contraction to y
Init           Variance-preserving spline-coef init (see L.4, KAT precedent)
```

**Why $m \geq d$ (non-compressive) and not $m < d$:** the whole point of the
mixer is to add a learned coordinate system *without* re-introducing Defect 1
(information bottleneck). $m = d$ is the lossless default. $m = 2d$ doubles
the mixer activation and mostly removes the speed win, so it should be used
only if $m=d$ underfits. $m < d$ is forbidden — that is exactly the
$T_\text{jhcg}$ regime this redesign is trying to escape.

**Why no separate $D$:** an independent $D: m \to d$ costs another $dm$
params and MACs. At $m=d$, this is 589K params and 589K MACs per token,
which materially weakens the all-win claim. Folding the decoder into the
Tucker output factor $U$ keeps the learned output projection but avoids the
extra dense path.

## J.2 Realistic vs theoretical metrics (with literature reality check)

| metric | MLP | CONS realistic | FullMix-Tucker rank-96 | old MAX with separate D |
|---|---:|---:|---:|---:|
| Storage (3 KAN layers × params) | 14.1M | ~11.5M (slight win) | **2.66M (5.3× less)** | ~4.0M (3.5× less) |
| FLOPs/token (3 KAN layers) | 28.3M | **~6.2M (4.6× less)** | **~12.1M (2.3× less)** | ~12.6M (2.2× less at rank 64) |
| Activation VRAM | 6.3 MB / layer | **~1.9 MB (3.3× less)** | **~1.5 MB (4.0× less)** | ~1.5 MB |
| Wall-clock speed | 1× | **1.5-2.5× faster** | **1.3-2.0× faster** | lower after A/D overhead |
| Quality vs MLP | (ref) | **±0.01 nat (high prob)** | **±0.02 nat (medium prob)** | similar quality, weaker systems win |

> **Speed precedent:** MoE locality-aware Triton kernels and sparse retrieval
> kernels show that locality-aware GPU code can deliver multi-x speedups on
> the sparse subproblem. FullMix-Tucker is harder because it retains the dense
> mixer $A$, so the realistic end-to-end training target is 1.3-2.5×, not the
> old 3-5× wall-clock headline.

> **VRAM precedent:** Activations dominate VRAM during training (input $u$ must be saved for backward). Per-token activation: $u \in \mathbb{R}^{B \times T \times d}$ = 1.5 MB at our scale. Cannot drop below this without recomputation. The "126× less" v1 claim was wrong.

> **Compression precedent:** ESPACE (NeurIPS 2024) achieves 50% LLM compression with 0.18 PPL increase. FullMix-Tucker rank-96 claims 5.3× FFN-block compression with ≤ 0.02 nat degradation — still aggressive vs literature, but much less implausible than v1's 20-50× claim.

## J.3 Quality estimate (probabilistic)

This is where empirical risk lives. Theoretical priors:

- **B1 has more sharp transitions than ReLU²** (G+1 knots vs 1 origin) → potentially better fit ability
- **No input-side information bottleneck** ($A: d \to m$ with $m \geq d$) → no Defect 1 cap
- **Adaptive grid + RMSNorm** → no Defect 3 magnitude loss
- ⚠ **Output-rank bottleneck (F.4.b)** — per-layer FFN update lives in a
  $\leq R_o$-dim subspace. For $R_o = 96$ vs MLP's $d = 768$, this is the
  dominant remaining risk on quality.

**Expected quality at 100M tokens, single seed, for `fm_b1_pa6` (primary):**
- 20%: $\Delta \leq -0.01$ nat (strict win — output-rank thesis holds and
  spline expressivity adds margin)
- 35%: $-0.01 \leq \Delta \leq +0.02$ nat (paper-worthy near-iso-quality —
  output rank is sufficient at $R_o = 96$)
- 30%: $+0.02 \leq \Delta \leq +0.05$ — output-rank bottleneck is real;
  asymmetric `fm_b1_pa6_asym` should rescue this band per F.4.c
- 15%: $\Delta > +0.05$ — neither symmetric nor asymmetric closes the gap;
  proceed to Strategy C (linear residual) in Phase 1.5

**For `fm_b1_pa6_asym` (the output-rank mitigation):** if the output-rank
bottleneck is the dominant cause of any gap at primary, asymmetric should
add ≈ 0.01-0.03 nat improvement at the same storage as `fm_b1_pa6_r128`.
If asymmetric delivers no detectable lift over `fm_b1_pa6`, the bottleneck
is elsewhere (likely curvature — see Part M.4 Step 3).

**For `fm_b1_full_r96` (the stretch headline candidate):** per F.5.1, full
replacement has 100% cumulative output-rank coverage and may match or beat
`fm_b1_pa6` despite per-layer rank being identical. Expected band:
similar to `fm_b1_pa6` in the central case ($\pm 0.02$ nat), with ~25%
chance of strict-winning where `fm_b1_pa6` only ties. Caveat: if $U_\ell$'s
collapse to redundant subspaces (L.1 Q2), `fm_b1_full_r96` underperforms
`fm_b1_pa6` and the headline downgrades to A+.

This estimate is **less optimistic than v4** because v5's output-rank
exposure shifts probability mass from the "near-iso" band to the "weak
medium" band, while increasing the total mass that requires asymmetric
rescue.

## J.4 Where this might fail

1. **Optimization stability**: Tucker decomposition is more stable than CP but still non-convex. Mitigation: warm-start dense tensor for a short phase, then SVD/HOSVD initialize Tucker factors.

2. **Sparse kernel slower than dense in practice**: cache miss patterns, register pressure. Mitigation: empirical benchmarking, fall back to dense if needed.

3. **B1 basis too sharp**: piecewise-linear with knots may oscillate or miss smooth structure. Mitigation: hybrid B1+B3 basis, or smooth regularization.

4. **Adaptive grid drift**: grid range can collapse or explode during training. Mitigation: grid_range clipping, EMA decay schedule.

5. **Output-rank bottleneck (the dominant risk per F.4.b).** The Tucker
   readout factors as $y = U\eta(x)$ with $U \in \mathbb{R}^{d \times R_o}$,
   so the per-layer FFN update is constrained to a subspace of dimension
   at most $R_o$. For $R_o = 96$, the layer can write into at most a 96-dim
   subspace of the 768-dim residual stream. MLP has full rank-$d$ output.
   This is *not* the same pathology as JHCG's input-side bottleneck
   (which is unrecoverable across layers); output-rank limits are
   recoverable via Pattern A (alternating MLP layers route what KAN
   layers cannot). Mitigation order: asymmetric rank ($R_o = 256$, same
   storage as symmetric r128), then linear residual, then MLP escape.
   See F.4.c.

---

# Part K — Path to Paper-Worthy Result

## K.0 Implementation strategy: reference vs production

Before any phase-specific code, we fix a global implementation principle.
FullMix-Tucker has three mathematically equivalent computational forms:

| form | description | use |
|---|---|---|
| **A — explicit Tucker tensor** | materialize $W_{kji} = \sum_{abc} U_{ka} C_{abc} V_{jb} Q_{ic}$ then evaluate | **never** — wastes Tucker's whole point |
| **B — 5-stage simplified** | $z \to \beta \to \xi \to \eta \to y$ as 5 independent PyTorch ops (F.4.b derivation) | **Phase 1 reference** + **permanent oracle** |
| **C — fused custom kernel** | fuse stages 1+2 (or 1+2+3) into a single Triton kernel; $\beta$ never materialized | **Phase 2 production** + **paper wall-clock numbers** |

### K.0.1 Mathematical equivalence proof (B ⇔ C ⇔ original)

Starting from the original Tucker form,

$$y_k \;=\; \gamma \sum_{j,i} W_{kji}\, B_i(z_j)
\;=\; \gamma \sum_{j,i,a,b,c} U_{ka} C_{abc} V_{jb} Q_{ic}\, B_i(z_j),$$

apply associativity to expose the 5-stage form:

$$y_k = \gamma \sum_a U_{ka} \underbrace{\sum_{b,c} C_{abc} \underbrace{\sum_j V_{jb} \underbrace{\sum_i Q_{ic} B_i(z_j)}_{\beta_{j,c}}}_{\xi_{b,c}}}_{\eta_a}.$$

The 5 stages are exactly the inner-to-outer reductions over $i$, $j$,
$(b,c)$, $a$ respectively. **No approximation, no factor reordering, no
basis truncation** — only associativity of finite sums.

The fused form C is the same expression with stages 1+2 (and optionally 3)
collapsed into a single loop that accumulates into $\xi$ (or $\eta$)
without materializing $\beta$. This is again pure associativity — the
mathematical function is identical.

**Equivalence at floating-point precision:**

| numerical layer | equivalence |
|---|---|
| infinite-precision real arithmetic | exact |
| fp32 | $\sim 10^{-6}$ relative error (different summation order) |
| bf16 | $\sim 10^{-3}$ relative error (different summation order) |
| autograd gradients | identical to fwd (tracked through the chosen ops) |

Any production kernel must match the reference output within bf16's
$\approx 10^{-3}$ relative tolerance. This is the unit-test contract
between Phase 1 and Phase 2.

### K.0.2 Why reference (B) lives forever, not just in Phase 1

The reference implementation is *permanent* infrastructure, not a temporary
crutch:

- **Phase 2 unit-test oracle**: kernel correctness check requires a
  trusted comparison output. The reference is that.
- **CPU / no-Triton fallback**: when the user runs on a machine without
  the kernel (e.g., a 3080 dev box, a CI runner, an academic reviewer's
  machine), the reference still produces a correct (slow) answer.
- **Supplementary code reproducibility**: the paper supplementary will
  ship the ~120-line reference module. Reviewers can read and run it
  without engaging the Triton kernel.
- **Numerical-stability investigations**: when training does something
  weird, swapping the kernel for the reference is the first triage step.

Treat the reference and the kernel as **a contract between two different
physical implementations of the same mathematical function**. If they
diverge by more than $10^{-3}$ in bf16, the kernel is buggy. Always.

### K.0.3 VRAM cost of B and how to manage it

The 5-stage form materializes $\beta \in \mathbb{R}^{B T \times m \times R_b}$
during the forward, which autograd retains for backward. At nanochat scale
($B=8$, $T=1024$, $m=d=768$, $R_b=16$):

$$\text{VRAM}(\beta) = 8 \cdot 1024 \cdot 768 \cdot 16 \cdot 2\,\text{B (bf16)} \approx 200\,\text{MB / replaced layer}.$$

For Pattern Full (12 replaced layers) this is 2.4 GB — manageable on H100
80 GB but worth handling explicitly. The standard fix:

```python
ffn_out = torch.utils.checkpoint.checkpoint(
    fullmix_tucker_module, x, use_reentrant=False
)
```

Recompute $\beta$ on backward at the cost of 1 extra forward pass per layer
($\sim 1.3-1.5\times$ slowdown on Phase 1, fully removed in Phase 2 where
the kernel never materializes $\beta$ at all).

### K.0.4 The non-negotiable phase order

```
Phase 1 (B reference, ~1 week)  ──passed quality gate──→  Phase 2 (C kernel, ~2 weeks)
                                                                ↓
                                                           Phase 3 validation
```

Do **not** start Phase 2 before Phase 1's quality gate passes. Reasoning:

1. C inherits B's quality exactly (per K.0.1 proof). If B fails, C is
   wasted engineering.
2. Phase 1's 6 cells include 4 ablations whose only purpose is diagnostic
   (`pa3` safety net, `direct_b1_pa6` mixer ablation, `pwlu_baseline`
   sanity check, `fm_b1_pa6_asym` output-rank rescue). Writing a
   custom kernel for them is over-engineering.
3. C only changes wall-clock; quality and rank arguments are decided in
   Phase 1 and are independent of kernel.

**Exception**: skeleton C (function signature, autograd.Function class,
test harness) can be drafted in parallel with Phase 1 — it is dead code
until Phase 1 passes, but having it ready halves Phase 2's calendar time.

## K.1 Phase 1 — Pure-PyTorch dense MVP (1 week)

**Goal:** validate quality at the FullMix-Tucker design WITHOUT relying on
the Triton kernel for correctness.

```python
# Topology: FullMix  (z = A x with A: d -> m, m = d non-compressive)
# Basis: B1 (compute 2 active per input, but in dense PyTorch)
# Tensor: Tucker readout W[k,j,i], primary rank (96,96,16)
#         Optional warm-start: train full m x d x L tensor briefly, then HOSVD-init Tucker
# Norm: RMSNorm on z + learnable scale (no tanh squash)
# Grid: adaptive [lo, hi] via EMA on z
# Init: variance-preserving spline-coef init, sigma_c approx sqrt(3/m)
# Diag: log post-FFN-update sigma per layer every 100 steps (Part L.4)
```

**Phase 1 cells — two-axis sweep (placement × rank), 8 cells total:**

The previous v5 cell list was rank-only. v6 adds the placement axis per
F.5.1 (cumulative output-rank coverage). The sweep is structured as a
2D grid where the diagonal is the primary path:

```
                     Pattern A    Pattern A+        Pattern Full
                     (K=3)        (K=6)             (K=12)
                     288/768      576/768           768/768
                     ─────────    ─────────────     ───────────────
sym  (96,96,16)      pa3          pa6 [PRIMARY]     full_r96 [STRETCH]
sym  (128,128,16)    --           pa6_r128          --
asym (256,96,16)     --           pa6_asym          --
direct  (no mixer)   --           pa6_direct        --
PWLU L0 baseline (R_b=1, A=I, R_o=d) ─── pwlu_baseline (sanity)
MLP                                  ─── mlp_baseline (reference)
```

| cell | placement (K) | rank $(R_o, R_i, R_b)$ | role |
|---|---:|---|---|
| `mlp_baseline` | – | – | reference |
| `pwlu_baseline` | 6 | $R_b=1, A=I, R_o=d$ | F.4.d L0 sanity — must match published PWLU within ±0.02 nat |
| `fm_b1_pa3` | 3 | (96, 96, 16) | safety net |
| **`fm_b1_pa6`** | **6** | **(96, 96, 16)** | **primary — matches v3.5 m7 placement, 75% output coverage** |
| `fm_b1_pa6_r128` | 6 | (128, 128, 16) | symmetric rank rescue |
| `fm_b1_pa6_asym` | 6 | (256, 96, 16) | output-rank rescue (F.4.c Strategy A) |
| `fm_b1_pa6_direct` | 6 | (96, 96, 16), $A=I$ | mixer ablation |
| `fm_b1_full_r96` | 12 | (96, 96, 16) | **stretch — paper headline target, 100% output coverage** |

**Why this set:**

1. **Diagonal-primary path** (`pa6` at sym r96) directly compares against
   v3.5 m7's +0.07 nat ceiling. Any improvement is the headline.
2. **Placement axis** (pa3 / pa6 / full at sym r96) tests F.5.1's
   cumulative-rank prediction. If `full` ≥ `pa6` ≥ `pa3` in quality,
   the cumulative-rank argument is empirically confirmed.
3. **Rank axis at the primary placement** (pa6 r96 / r128 / asym) tests
   F.4.b's output-rank thesis. If asym beats r128 at same storage,
   confirmed.
4. **`pa6_direct`** isolates the mixer's contribution at the primary
   placement.
5. **`pwlu_baseline`** is the F.4.d L0 rung — its result must match
   published PWLU GPT-2 numbers, otherwise we have a bug to fix before
   any FullMix-Tucker number can be trusted.

**Decision tree after Phase 1:**

```
if fm_b1_pa6 beats MLP by >= 0 nat at 100M:
    if fm_b1_full_r96 also beats MLP:
        -> Phase 2 with placement=Full (paper headline)
    else:
        -> Phase 2 with placement=A+ (paper still strong, m7-frame story)
elif fm_b1_pa6_asym beats MLP:
    -> output-rank thesis confirmed; Phase 2 default = asymmetric, A+
elif fm_b1_pa6_r128 beats MLP:
    -> general rank issue; Phase 2 with r128, A+; revisit asymmetric ablation
elif pwlu_baseline matches published PWLU:
    -> spline pipeline OK; failure is FullMix-Tucker specific
       -> Phase 1.5: add Strategy C (linear residual) cell
else:
    -> spline pipeline has a bug; do not interpret other cells; debug first
```

**Cost:** 8 cells × 100M tokens ≈ ~$30-35. (The two-extra cells over the
v5 plan — `fm_b1_full_r96` and `pwlu_baseline` — buy the placement-axis
diagnostic and the bug-detection oracle, both worth substantially more
than the marginal $10.)

**VRAM check before launch:** Pattern Full at $\beta$ materialization
costs $\approx 200\text{ MB} \times 12 \approx 2.4\text{ GB}$ extra
activation memory. Use `torch.utils.checkpoint.checkpoint` on each
FullMix-Tucker block to keep this in check (see K.0.3). On H100 80 GB this
is comfortable; on smaller cards it is mandatory.

## K.2 Phase 2 — Triton kernel (2 weeks, forward + backward)

**Goal:** make the locality + Tucker-decomposed compute path fast on
**both** forward and backward (per FlashKAT, Part L.5, the backward is the
real bottleneck — atomic adds on the coefficient table serialize on H100).

Forward sketch (T_mixer + B1 + Tucker):
```python
@triton.jit
def sparsespline_b1_tucker_forward(
    x_ptr,                       # (B*T, d) input pre-mixer
    A_ptr,                       # (m, d) mixer weights
    U_ptr, V_ptr, C_ptr,         # Tucker factors: U(d_out, R_o), V(m, R_i), C(R_o, R_i, R_b)
    W_b_ptr,                     # (R3, L=G+1) spline-mode factor
    grid_lo, grid_hi,            # adaptive grid
    out_ptr,                     # (B*T, d) output
    R_o: tl.constexpr, R_i: tl.constexpr, R_b: tl.constexpr,
    m: tl.constexpr, d: tl.constexpr, G: tl.constexpr,
):
    # Stage 1: dense matmul z = A x  (BLAS-friendly)
    # Stage 2: B1 locality on z  (2 active basis per input dim)
    #   bin_idx, b0 = 1-t, b1 = t
    # Stage 3: Tucker readout contraction over (R_o, R_i, R_b)
    #   collect active basis values into a length-2 tile per dim
    ...
```

Backward design (FlashKAT-style):
- **Tile over coefficient dims**, not tokens — each program owns a slice of
  the spline-mode factor and accumulates token contributions locally,
  avoiding cross-program atomics on $C$ and $W_b$.
- **Double-buffer active basis** in SMEM; B1 only needs 2 floats per dim.
- Use FlashAttention-2-style 2-pass scheme where pass-1 computes per-tile
  partial sums and pass-2 reduces — no atomics anywhere.

**Expected end-to-end train-step speedup:** 1.4-2.5× vs MLP baseline (this
is the **literature-grounded** target after FlashKAT's lesson; the
"5-10× forward speedup" phrasing was insufficient).

**Cost:** engineering time + benchmark cells ~$10 (heavier than v1
estimate because of backward).

**Pass criterion:**
```
end-to-end train step wall clock  <=  0.7 x MLP baseline
forward-only wall clock           <=  0.4 x MLP baseline
all_finite over 1k steps          ==  True
```

## K.3 Phase 3 — Full validation (1 week)

- Multi-seed × 100M (3 seeds for σ_seed measurement)
- 400M validation (single seed) at best config
- Comparison: FullMix-Tucker vs MLP baseline vs current JHCG vs hybrid (refiner) reference

**Cost:** ~$30.

## K.4 Phase 4 — Paper-grade artifacts

- Plot all axes (quality, storage, VRAM, speed) on Pareto chart
- Open-source kernel implementation
- Theoretical contribution writeup: "exploiting B-spline locality in transformer FFN"

---

# Part L0 — FFN Specialization: Drop the Base Branch

## L0.1 The original KAN edge formula

Liu et al's per-edge activation:

$$\phi(x) = \underbrace{w_b \cdot \text{SiLU}(x)}_{\text{base branch}} + \underbrace{w_s \cdot \sum_{i=0}^{G+k} c_i B_i(x)}_{\text{spline branch}}$$

The base branch has 3 motivations in the original paper:
1. Default working function (when spline coefficients init to small values)
2. Training stability (spline params high-noise; base provides low-noise baseline)
3. Generic robustness (KAN designed for arbitrary tasks: PDE, regression, ...)

## L0.2 Why FFN-in-LLM doesn't need it

In transformer FFN:

$$x_{l+1} = x_l + \text{FFN}(\text{RMSNorm}(x_l))$$

The **residual stream itself is the default path**. If FFN's output is small/zero, $x_{l+1} = x_l$. Internal base-SiLU is *architectural redundancy* with the existing residual.

KAN's original tasks (PDE, regression) **don't have external residual** → internal base is needed. **FFN does** → drop it.

## L0.3 Smooth-on-smooth makes things worse for LM

base branch = SiLU = $C^\infty$ smooth. Spline branch = $C^k$ smooth. Sum = at most $\min(C^\infty, C^k) = C^k$ smooth.

Adding SiLU **does NOT add sharp transitions** — it's an even smoother component being summed in. For token prediction (which needs sharp decisions), this is the **wrong direction**.

## L0.4 Empirical confirmation (v3.9)

| cell | branch_mode | Δval at 100M | spline_W |
|---|---|---:|---:|
| m7_lh_ref | spline_only_active | **+0.0667** | 22.04 |
| m7_lh_base_spline | base_spline | +0.0727 (worse 0.006) | 27.16 |

Adding base branch **measurably hurts** quality on nanochat. Empirical agrees with theory.

## L0.5 What we save by dropping base branch

Per edge:
- **Storage:** $-2$ params ($w_b$ removed, plus the base SiLU has no own params but the base gating stays; in current SparseFuseJHCG `spline_only_active` mode this is already 0)
- **Compute:** $-1$ SiLU eval, $-1$ multiply, $-1$ add per token-edge
- For full-d KAN ($d=768$, $d_o=768$ edges): saves ~$3M$ FLOPs/token (additive to spline savings)

**Crucially for our redesign:** with no base branch, the **entire forward path is locality-aware**. Triton kernel doesn't have to fuse two parallel paths — only the locality-friendly spline. Kernel is cleaner and faster.

## L0.6 Inductive bias becomes purer

| component | bias |
|---|---|
| current (with base branch) | "smooth spline + smoother SiLU" — strictly smoother than spline alone |
| **redesign (no base)** | "pure piecewise polynomial via spline" — **bias matches grid resolution exactly** |

For B1 basis specifically: pure piecewise linear, with G+1 sharp knots. **G+1 sharp transitions per dim available** (vs MLP's 1 sharp transition at ReLU edge). This is structurally MORE sharp-capable than MLP.

## L0.7 Naming

This redesign is no longer "vanilla KAN/JHCG":
- Drops base branch (FFN specialization)
- Locality-exploiting kernel (not standard)
- No encoder/decoder bottleneck (architectural)
- CP/Tucker decomposition (compression layer)

Working name: **LL-FFN-KAN** (Locally-Local FFN-specialized KAN) or **SparseFuse-FFN-KAN**. Final naming TBD by results.

---

# Part L — Risks and Open Questions

## L.1 Open theoretical questions

1. **Are FFN per-layer updates in trained transformers naturally low-rank?**
   This is the central empirical question that F.4.b's algebraic exposure
   surfaces. ESPACE NeurIPS 2024 establishes that LLM intermediate
   activations admit ~50% rank reduction at low PPL cost; LoRA shows
   rank-8-to-64 adapters work for fine-tuning; attention-head outputs
   concentrate in few singular directions. But *FFN-update rank* (the
   per-layer correction signal, not the activation) is less directly
   measured. If it is naturally $\leq R_o = 96$, FullMix-Tucker matches MLP.
   If it requires $R_o = 256-512$, asymmetric rescue (F.4.c) is needed.
2. **Are output-readout subspaces $\mathrm{Col}(U_\ell)$ across layers
   diverse?** F.5.1's cumulative-coverage bound is tight only if the
   $U_\ell$ for $\ell = 1, ..., K$ are linearly independent (so that
   $\dim(\bigcup \mathrm{Col}(U_\ell)) = \min(K \cdot R_o, d)$). If layers
   redundantly learn similar output bases, the cumulative bound collapses
   and Pattern Full does not actually saturate $d$. Phase 1 must verify
   this by logging $\sigma_{\min}$ of the stacked $U$ matrix.
3. **What is the empirical relationship between Tucker rank and LM quality?**
   The asymmetric ablation `fm_b1_pa6_asym` is the first experiment to
   answer this for output-rank specifically.
4. **Does B1 basis have hidden disadvantages we haven't seen?** ReLU has been
   studied for decades; B1 splines as activation function less so. The
   `pwlu_baseline` cell isolates this from other variables.
5. **How does adaptive grid interact with cosine LR schedule?** Both adapt;
   could oscillate.

## L.2 Engineering risks

1. **Triton kernel correctness vs PyTorch reference** (must verify numerics)
2. **Memory bandwidth bottleneck** when token batch is large (sparse gather can thrash cache)
3. **Backward pass complexity** — autograd through sparse gather is non-trivial

## L.3 Paper risk

1. **If quality strict-win (Δval ≤ −0.01) requires >100M tokens** → expensive validation
2. **If kernel speedup falls short of theoretical** → "theoretical 16× → empirical 3×" is less compelling
3. **If hybrid (refiner) achieves same systems wins** the user-rejected hybrid path wouldn't be a real failure

## L.4 Initialization sensitivity (KAT precedent)

> *"The initialization of weights in KANs is particularly challenging due to
> their learnable activation functions. Standard initialization schemes used
> for MLPs (e.g., Kaiming, Xavier) do not directly translate."* —
> *Kolmogorov–Arnold Transformer* (KAT, Yang & Wang, ICLR 2025, arxiv
> 2409.10594).

KAT identifies three sources of pain when scaling KAN-style activations to
transformer-scale models:
(i) base/spline branch contributions are not variance-balanced under standard
init, causing early-training instability;
(ii) B-spline basis functions are not CUDA-native, so naive forward kernels
are slow;
(iii) per-edge learnable activations make the *effective* fan-in/fan-out
input-dependent, so a fixed-σ init is not variance-preserving across the
network.

KAT's own fix is to swap B-spline for *rational* basis functions (Padé form,
trivially CUDA-friendly) plus a "variance-preserving" init derived for
rationals.

**For SparseSpline-FFN we keep B-spline** (locality is the central point) but
**inherit the init lesson**:

- **Variance-preserving spline-coef init (corrected v6.1).** The full
  pipeline (mixer → spline lookup → V → C → U → γ) shrinks variance through
  the Tucker readout, not just at the spline edge. Tracing through with
  orthogonal-column $V, U$ inits and $C \sim \mathcal{N}(0, 1/(R_i R_b))$
  and unit-variance input $z$:

  $$\mathrm{Var}[\beta] = \sigma_c^2 \cdot \mathbb{E}[B_0^2 + B_1^2] = \tfrac{2}{3}\sigma_c^2$$
  $$\mathrm{Var}[\xi] = \mathrm{Var}[\beta] \quad (V \text{ orthogonal})$$
  $$\mathrm{Var}[\eta] = \mathrm{Var}[\xi] \quad (R_i R_b \text{ terms in } C \text{ contraction})$$
  $$\mathrm{Var}[y] = \gamma^2 \cdot \tfrac{R_o}{d} \cdot \mathrm{Var}[\eta] \quad (U \text{ orthogonal cols})$$

  Solving $\mathrm{Var}[y] = 1$ at $\gamma = 1$:

  $$\boxed{\;\sigma_c \;=\; \sqrt{\tfrac{3 d}{2 R_o}}\;}$$

  At $d = 768, R_o = 96$: $\sigma_c \approx 3.46$.

  **Verified empirically** in `tests/test_fullmix_tucker.py::test_init_output_variance_close_to_unit`
  across $d \in \{64, 128, 256\}$ and $R_o \in \{32, 64, 128\}$: layer output
  $\sigma_y$ lands in $[0.92, 0.99]$, very close to the target.

  **Earlier draft formula** $\sigma_c = \sqrt{3/m}$ was incomplete — it
  treated the layer as bare spline edges and forgot the Tucker readout
  variance shrinkage. At production scale that formula gives output
  $\sigma_y \approx 0.018$ (under-shoots target by $\sim 50\times$). Do not
  use it; use $\sqrt{3 d / (2 R_o)}$ instead.

  Cubic B3 has different active-basis variance ($\mathbb{E}[B^2]$ for cubic
  averages roughly half of B1's, so the constant prefactor differs) and
  needs a separate calibration; we will measure empirically when B3
  ablations are added.

- **No base branch (per Part L0)** removes the base/spline variance-balancing
  problem entirely — there is only one branch, so no cross-branch divergence.

- **Mixer $A$ init.** $A: d \to m$ uses standard Kaiming-uniform; only the
  spline coefficients need the special init.

- **Tucker factor init.** Per ECCV 2020 (CP instability) and the LLM
  decomposition literature, factor matrices should be initialized by SVD of a
  *target* dense $W$ trained for a few hundred steps in PyTorch dense mode.
  This avoids cold-start divergence of factors. Concretely:
    1. Train FullMix with full $m \times d \times L$ spline tensor (no Tucker)
       for ~500 steps.
    2. SVD-initialize Tucker factors to match.
    3. Switch to fused Tucker kernel for the rest of training.
  This adds ~1% wall-clock overhead at start but materially de-risks the
  factorized run.

**Diagnostic to add to the FullMix driver:** log $\sigma$ of post-FFN residual-stream
update at every layer at step 0 and every 100 steps. If late-layer σ collapses
(< 0.5 of step-0) or explodes (> 2× of step-0) within first 200 steps, init
is wrong — abort and recalibrate $\sigma_c$.

## L.5 Backward kernel atomics (FlashKAT precedent)

> *"We identified the bottleneck during the backward pass of [efficient KAN]
> kernels: the backpropagation requires significant slow memory traffic and
> the use of atomic adds, leading to a slowdown … FlashKAT, built on a
> restructured kernel, achieves training speedups of up to 86.5× compared
> with the state-of-the-art KAN."* — *FlashKAT* (Lin et al., arxiv
> 2505.13813, 2025).

This directly modifies our Phase 2 plan. The naive Triton kernel sketched in
Part K.2 *will* be fast on the **forward** pass (sparse gather, small
matmul), but will be **slow on the backward** pass if implemented in the
straightforward way:

- Each output token gradient must be scattered back to the $k+1$ active basis
  *coefficients* of every input dim. With one Triton program per output
  token, multiple programs concurrently update the same $c_i$ slot →
  `atomic_add` is required → slow on H100 (atomics serialize on the same
  cache line).
- Additionally, the gradient w.r.t. $u$ (the spline input) requires reading
  the same coefficient table that the forward read, plus the basis
  *derivative* values; if those traverse HBM rather than SMEM, bandwidth
  bottlenecks.

**Implications for the SparseSpline-FFN kernel design:**

1. **Tile the backward over coefficient dims, not over tokens** — invert
   the parallelization axis so each Triton program owns a slice of $c_i$
   and accumulates contributions from all tokens locally (no cross-program
   atomics). FlashKAT's published recipe.

2. **Double-buffer the active-basis table in SMEM** for backward. With B1,
   only 2 basis values per input — small enough to keep in registers.
   With cubic, 4 — still small. Tucker rank $R_3 = 16$ means
   $L \times R_3 \approx 23 \cdot 16 = 368$ floats — fits trivially in SMEM
   per program.

3. **Forward speedup is *not* the right metric**. Report end-to-end
   train-step wall clock (forward + backward + optimizer), and compare to
   MLP under the same conditions. A 10× forward win that is offset by a
   2× backward loss is a 1.7× train-step win — still useful but
   materially below the 3-5× headline.

4. **CONS variant first.** If the FlashKAT-style backward turns out to be
   too engineering-heavy for the paper timeline, the CONS design point in
   J.1.a (no Tucker decomposition, dense-on-bottlenecked tensor with B1
   locality only) has a much simpler backward (standard PyTorch autograd
   on a fused gather) and is the safe fallback for "wins on storage and
   FLOPs, ties on speed."

We update Phase 2's success criterion accordingly:

```
Phase 2 PASS if  end-to-end train step wall clock <= 0.7 x MLP baseline
                 (i.e., >= 1.4x train speedup, not just forward)
```

This is **stricter than v1's "5-10x forward speedup"** and is the metric
reviewers will actually look at.

---

# Part M — Self-review of FullMix-Tucker

This section is deliberately adversarial: it evaluates the proposed
FullMix-Tucker method as if we were reviewing it for rejection.

## M.1 What is genuinely strong

1. **It keeps MLP's most important inductive bias.** The learned mixer
   $A: d \to m$ makes spline knots fire on learned oblique directions
   $z_j = a_j^\top x$, not on raw coordinate axes. This fixes the main
   weakness of direct-d KAN.

2. **It removes a redundant dense path.** Folding the decoder into Tucker's
   output factor $U$ preserves a learned output projection but avoids the
   extra $dm$ params and MACs of a separate $D$.

3. **It exposes true conditional computation.** For B1, each mixed channel
   activates only two local basis functions. MLP touches all hidden weights
   for every token; FullMix-Tucker touches only a tiny basis slice and then
   contracts low-rank factors.

4. **The system margins are real at rank 96.** At $d=m=768$,
   $R=(96,96,16)$ gives 885K params and 2.02M MACs per token, versus MLP's
   4.72M params/MACs. That clears storage, arithmetic, and activation-VRAM
   targets without relying on CP.

## M.2 Weak points reviewers can attack

1. **Output-rank bottleneck (the sharpest weak point, per F.4.b).** Algebraic
   expansion of the Tucker readout exposes $y = U\eta(x)$, $U \in
   \mathbb{R}^{d \times R_o}$. The per-layer FFN update lives in a subspace
   of dimension $\leq R_o = 96$, vs MLP's $d = 768$. A reviewer who reads
   F.4.b (or who simply does the algebra) will see this immediately and
   challenge the rank choice. Our defense rests on (a) Pattern A
   alternation (output-rank limits are recoverable across layers, unlike
   input-side bottlenecks), (b) the empirical observation that LLM FFN
   updates appear approximately low-rank (ESPACE NeurIPS 2024 precedent),
   and (c) the asymmetric-rank mitigation in F.4.c (boost $R_o$ to 256 at
   the same storage as symmetric r128). The asymmetric ablation
   `fm_b1_pa6_asym` in Phase 1 is specifically designed to give
   reviewers a satisfying answer to this attack.

2. **The mixer is dense.** We cannot claim MoE-like wall-clock speedups unless
   the Tucker/spline part is sufficiently dominant or the kernel fuses well.
   The dense $A x$ cost is unavoidable if we want learned oblique directions.

3. **Adaptive grids can drift.** If the EMA range follows training noise, knots
   move under the optimizer and create non-stationary targets. This can make
   the layer look worse than MLP even when the function class is adequate.

4. **Backward remains the hard part.** A fast forward kernel is not enough.
   If coefficient gradients require atomics or large HBM traffic, the training
   step can lose despite lower theoretical MACs.

5. **B1 may be too sharp or too low-order.** ReLU-like sharpness is the reason
   to try B1, but token modeling may benefit from ReLU²'s magnitude curvature.
   A B1 spline plus output gain may not perfectly substitute ReLU².

## M.3 Kill criteria

FullMix-Tucker should be demoted if any of these happen in controlled runs:

| gate | fail condition | interpretation |
|---|---|---|
| Pipeline sanity | `pwlu_baseline` differs from published PWLU GPT-2 numbers by >0.02 nat | spline pipeline has a bug — fix before interpreting any other cell |
| Quality (symmetric) | `fm_b1_pa6_r128` still worse than MLP by >0.04 nat at 100M | symmetric scaling does not close the gap |
| Output-rank diagnostic | `fm_b1_pa6_asym` ≈ `fm_b1_pa6` (asymmetric does not help over baseline) | bottleneck is NOT the output rank — go to Strategy C (linear residual) |
| Output-rank confirmation | `fm_b1_pa6_asym` $\gg$ `fm_b1_pa6_r128` at same storage | output-rank thesis confirmed — make asymmetric default downstream |
| Placement-rank confirmation | `fm_b1_full_r96` $>$ `fm_b1_pa6` $>$ `fm_b1_pa3` | F.5.1 cumulative-rank prediction confirmed — favor Pattern Full |
| Placement-rank disconfirmation | `fm_b1_pa3` ≈ `fm_b1_pa6` ≈ `fm_b1_full_r96` | placement axis does not matter; cumulative-rank argument was wrong |
| Subspace diversity | $\sigma_{\min}([U_1\,\|\,...\,\|\,U_K]) < 0.05 \cdot \sigma_{\max}$ | $U_\ell$ collapsed to redundant subspaces; F.5.1 bound is loose |
| Mixer ablation | `fm_b1_pa6_direct` ≈ `fm_b1_pa6` | learned mixer is not buying quality; simplify |
| Speed | fused training step >0.9× MLP wall clock | locality does not overcome kernel overhead |
| Stability | output update RMS collapses/explodes within first 200 steps | init/grid/gain not controlled |
| Combined rank ceiling | r128, asym, full, AND linear residual all fail | bottleneck is topology, not rank or readout |

## M.4 If FullMix-Tucker fails, the better next direction

Do not immediately increase rank indefinitely. The escalation order
follows F.4.c:

**Step 1 — Asymmetric rank (Strategy A).** If the kill-criterion for
`fm_b1_pa6_asym` shows it materially helps over symmetric, default to
asymmetric for all downstream phases. This is *not* a fail; it is a
re-tuning under the diagnosed pathology.

**Step 2 — Linear residual (Strategy C).** If asymmetric also fails, add a
full-rank linear escape:

$$y = W_0 x + U\, \eta(x), \qquad W_0 \in \mathbb{R}^{d \times d}.$$

This eliminates the output-rank bottleneck at the cost of $d^2 = 590\text{K}$
extra params. Initialize $W_0$ near zero so early training matches the pure
variant. Paper story becomes "SparseSpline-FFN with linear residual" — still
clean, and the linear residual is a $\sim 8\times$ smaller dense path than
the MLP we are replacing.

**Step 3 — Gated MLP escape (Strategy D).** Only after Steps 1 and 2 fail:

$$y = W_0 x + U\, \eta(x) + \alpha \cdot W_2'\,\mathrm{ReLU}^2(W_1' x)$$

with $W_1': d \to h$, $h \ll 4d$, $\alpha \approx 0$ at init. This adds back
ReLU²'s squared-magnitude curvature, which B1 lacks and which may be the
true MLP advantage. Paper story becomes "SparseSpline-FFN with dense
curvature corrector" — still publishable, but the cleanest story is
weakened.

**Step 4 — Acknowledge limit.** If even Step 3 fails, the MLP advantage on
LM tasks is likely neither rank nor curvature but something more
fundamental about $W_2$'s full-rank dense mixing. At that point we publish
the negative result honestly: locality + Tucker is a clean compression
recipe but cannot fully replace MLP at this scale, and the hybrid refiner
mode (rejected for this paper) becomes the practical answer.

## M.5 Current recommendation

Proceed with the 8-cell Phase 1 from K.1 (placement × rank sweep):

```
reference         mlp_baseline                   pure MLP
sanity            pwlu_baseline                  R_b=1, A=I, R_o=d (PWLU equivalent)
safety net        fm_b1_pa3                      K=3, sym (96, 96, 16)
PRIMARY           fm_b1_pa6                      K=6, sym (96, 96, 16) — m7 placement
sym rescue        fm_b1_pa6_r128                 K=6, sym (128, 128, 16)
output-rank fix   fm_b1_pa6_asym                 K=6, asym (256, 96, 16)
mixer ablation    fm_b1_pa6_direct               K=6, sym (96, 96, 16), A=I
STRETCH           fm_b1_full_r96                 K=12, sym (96, 96, 16) — paper headline
```

Three diagnostic axes:

1. **`pwlu_baseline`** validates the spline pipeline. If it does not match
   published PWLU GPT-2 numbers, every other cell is suspect.
2. **Placement axis** (`pa3` / `pa6` / `full_r96`) tests F.5.1's
   cumulative-coverage prediction. The expected ordering is
   `full_r96` ≥ `pa6` ≥ `pa3`.
3. **Rank axis at primary placement** (`pa6` / `pa6_r128` / `pa6_asym`)
   tests F.4.b's output-rank thesis. If asym beats r128 at same storage,
   the thesis is confirmed.

Only call the final method `SparseSpline-FFN` after one of these candidates
clears the quality + storage + VRAM + speed gates. The paper headline is
chosen by which placement actually wins:

| best cell | paper headline |
|---|---|
| `fm_b1_full_r96` wins | "FullMix-Tucker fully replaces MLP in transformer FFN" |
| `fm_b1_pa6` wins, full does not | "FullMix-Tucker replaces late-half FFNs and breaks the rank bottleneck" |
| only asym/r128 wins | "Output-rank-aware FullMix-Tucker replaces 6-of-12 FFNs" |
| only pa3 wins | "Pattern A FullMix-Tucker as MLP complement" (weakest claim, last resort) |

---

# Part N — Decision Points

## N.1 Phase 1 success criteria

Read on the **best of {`fm_b1_pa6`, `fm_b1_pa6_asym`, `fm_b1_full_r96`}**
(primary, output-rank rescue, and stretch). Pre-flight sanity required:
`pwlu_baseline` must be within $\pm 0.02$ nat of published PWLU GPT-2.

```
Pure-PyTorch FullMix-Tucker at 100M tokens:

  STRICT WIN     best Δval ≤ -0.01 nat            → proceed to Phase 2 (kernel)
  IS-QUALITY     -0.01 < best Δval ≤ +0.02 nat     → proceed (paper-worthy)
  WEAK MEDIUM    +0.02 < best Δval ≤ +0.04 nat     → Phase 1.5: Strategy C (linear residual)
  FAIL           best Δval > +0.04 nat             → Phase 1.5 Strategy C, then D, then negative-result writeup
```

**Diagnostic reads (run regardless of pass/fail):**

```
Pipeline OK                       if |pwlu_baseline - published_pwlu| ≤ 0.02 nat
                                    (otherwise debug spline path before interpreting anything)

Output-rank thesis confirmed      if Δval(pa6_asym) - Δval(pa6) ≤ -0.015 nat
                                    (asymmetric materially better than symmetric pa6)
Output-rank thesis disconfirmed   if Δval(pa6_asym) ≈ Δval(pa6)
                                    → bottleneck is curvature or topology, not output rank

Cumulative-rank thesis confirmed  if Δval(full_r96) ≤ Δval(pa6) ≤ Δval(pa3)
                                    (more layers = better, per F.5.1)
Cumulative-rank thesis fails      if Δval(pa3) ≤ Δval(pa6) ≤ Δval(full_r96)
                                    (more layers = worse — likely U_ℓ collapsed; check σ_min diagnostic)

Mixer-matters confirmed           if Δval(pa6_direct) - Δval(pa6) ≥ +0.02 nat
                                    (mixer is doing real work)
Mixer-redundant                   if Δval(pa6_direct) ≈ Δval(pa6)
                                    (drop the mixer for free in Phase 2)
```

**Paper-headline branching:** the strict-win / is-quality bucket above is
sufficient for paper acceptance. Within those buckets, the *headline*
depends on which placement won (see M.5 table). The cleanest paper story
is `fm_b1_full_r96` strict-win, but `fm_b1_pa6` strict-win is also a
strong claim because it directly compares to v3.5 m7's +0.07 nat ceiling.

## N.2 Should we pre-write the kernel or wait for Phase 1?

**Pre-write argument:** Triton kernel is slow to develop. Starting in parallel saves time.
**Wait argument:** If Phase 1 fails, kernel is wasted.

**Recommendation:** parallel both — Phase 1 in PyTorch validates quality (fast), kernel engineering happens in parallel (sunk cost if quality fails, but exhilaratingly fast if quality works).

---

# Appendix: Files / Code Layout

```
sparsefuse/
  jhcg.py                    # current implementation, do not break
  fullmix_tucker.py          # NEW: Phase 1 reference (5-stage PyTorch, K.0 form B)
                             #      - permanent oracle for Phase 2 numerical match
                             #      - ~120 lines, autograd-only
  fullmix_tucker_kernel.py   # NEW: Phase 2 production (fused Triton, K.0 form C)
                             #      - autograd.Function wrapping Triton fwd/bwd
                             #      - must match fullmix_tucker.py within bf16 1e-3
  tucker_init.py             # NEW: variance-preserving init + HOSVD warm-start (L.4)

nanochat_integration/        # mirrored adapters (existing)
  nanochat_v41_redesign.py   # NEW: nanochat driver wrapping fullmix_tucker
                             #      - replace_mlp_with_fullmix_tucker(model, pattern, ...)
                             #      - Pattern A / A+ / Full layer selection

scripts/
  run_nanochat_v41_phase1_launch.py    # NEW: 8-cell parallel launcher (K.1)
  run_nanochat_v41_phase2_launch.py    # NEW: kernel-validated launcher (K.2)
  modal_phase3d_h100.py                # MODIFY: add run_nanochat_v41_phase{1,2}

JHCG_REDESIGN_THEORY.md          # this doc
EXPLAINATION_FOR_REDESIGN.md     # plain-language companion (existing)
PHASE3_NANOCHAT_V41_*.md         # closeout receipts (created as we go)
```

**Naming conventions** (v6):
- `fullmix_tucker.py` is the *reference*, slow and correct. Never deleted.
- `fullmix_tucker_kernel.py` is the *production* path, fast. Must pass
  `pytest sparsefuse/test_kernel_match.py` (compares to reference within
  bf16 tolerance) before any training run.
- All `nanochat_v41_*` files import from both — selection via a
  `use_kernel: bool` flag. Phase 1 sets it False; Phase 2 sets it True.
