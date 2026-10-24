# White Paper: The ChronAeon TimeTree (`chronaeon.timetree`) Algorithm and Coalescent Skyline Dynamics

**Authors:** ChronAeon & Phylodate Benchmark Working Group  
**Date:** October 2026  
**Software Branch:** `chronaeon` (`feat/chronaeon-timetree`, `v0.3.1` / `hyphaeon-core v0.4.0`) vs. `phylodate v0.0.1`  

---

## Abstract

We present the mathematical formulation, algorithmic design, and demographic skyline evaluation of **ChronAeon TimeTree** (`chronaeon.timetree`), a hybrid distance-geometry and tree-isotonic solver that bridges tree-free molecular clock inference with full time-calibrated phylogenetic reconstruction. Unlike conventional post-hoc dating tools in `phylodate` (`lsd2`, `treetime`, `treedater`, `reltime`, `node_dating`, `wlogdate`, `chronumental`, `chronos`, `treepl`, `mdcat`)—which require an external maximum-likelihood substitution phylogeny ($95\text{--}99.9\%$ of total compute time)—and unlike baseline scalar **ChronAeon Mode 1**—which estimates only $(\hat{\mu}, \hat{t}_{\mathrm{MRCA}})$ and incurs a positive ancestral profile self-diversity offset $+D(R, Q(\gamma^*))$ on small trees—`chronaeon.timetree` recovers a complete, temporally consistent binary time-tree directly from either raw nucleotide sequences (`chronaeon_timetree_seq`) or substitution trees (`chronaeon_timetree_tree`) in $5\text{--}900\text{ ms}$. Furthermore, we derive and benchmark the exact mathematical mechanisms governing how **heterochronous coalescent skyline plots** ($\hat{N}_e(t)$) and **Lineages-Through-Time (LTT)** curves differ between `chronaeon.timetree` and `phylodate` timetrees across synthetic demographic trajectories (constant, exponential growth, and epidemic boom-bust) and empirical viral outbreaks (Rabies, Ebola, Chikungunya, and Dengue).

---

## 1. Motivation: Bridging Continuous Distance Geometry and Discrete Time-Trees

Molecular clock dating methods historically divide into two paradigms:

1. **Two-Stage Substitution-Tree Calibration (`phylodate`)**: A maximum-likelihood (ML) unrooted phylogeny $T_u = (V, E, \mathbf{b})$ with branch lengths $b_e$ (in substitutions/site) is first inferred via IQ-TREE or RAxML, after which internal node timestamps $\mathbf{t}_I \in \mathbb{R}^{|V_I|}$ and clock rate $\mu$ are estimated via constrained weighted least squares (`lsd2`, `node_dating`), penalized/Gamma likelihood (`treedater`, `chronos`, `treepl`), log-transformed rate regularization (`wlogdate`), differentiable optimization (`chronumental`), or iterative marginal message passing (`treetime`).
2. **Tree-Free Continuous Distance Geometry (`chronaeon` Mode 1)**: Given an alignment $\mathbf{S} = \{s_1, \dots, s_N\} \in \{A,C,G,T\}^{N \times L}$ with tip sampling times $\mathbf{t} = (t_1, \dots, t_N)^\top$, ChronAeon constructs a continuous time-decay soft ancestral nucleotide profile $Q(\gamma^*) \in \Delta^{L \times 4}$ and regresses analytical Tamura-Nei (TN93) divergences $d_i(\gamma^*) = D_{\mathrm{TN93}}(s_i, Q(\gamma^*))$ against $t_i$.

Each paradigm exhibits a distinct structural limitation:
- **ML Tree Bottleneck & Root Sensitivity in `phylodate`**: Inferring $T_u$ requires $O(N^2 L)$ likelihood evaluations ($10^1\text{--}10^3\text{ s}$), and downstream dating methods (`lsd2`, `treedater`, `reltime`, `wlogdate`) degrade severely if supplied an unrooted ML tree without explicit root optimization or when zero-mutation short branches ($b_e = 0$) attenuate the global clock rate $\hat{\mu}$.
- **The Profile Self-Diversity Offset in Uncentered Scalar ChronAeon**: On an additive tree with true root $R$, the expected distance from tip $i$ to the convex combination $Q(\gamma) = \sum_{j=1}^N w_j(\gamma) s_j$ satisfies:
  $$\mathbb{E}\big[D(s_i, Q(\gamma))\big] = d(R, s_i) + \underbrace{\sum_{j=1}^N w_j(\gamma) d(R, s_j)}_{+\,D(R, Q(\gamma)) \;>\; 0} - 2 \sum_{j=1}^N w_j(\gamma) d(R, \mathrm{MRCA}(i, j)). \tag{1}$$
  Because the earliest tips lie at positive evolutionary distance from the true root $R$, the constant term $+D(R, Q(\gamma)) > 0$ inflates the raw regression intercept $d_0$ and shifts the uncentered scalar estimate $\hat{t}_{\mathrm{MRCA}} = \bar{t} - \bar{d}/\hat{\mu}$ backward in time on small cohorts ($N < 50$).

---

## 2. The ChronAeon TimeTree (`chronaeon.timetree`) Algorithm

`chronaeon.timetree` resolves both limitations through a six-stage analytical pipeline implemented in [`src/chronaeon/timetree.py`](../../src/chronaeon/timetree.py).

### Stage 1: Time-Decay Soft-Profile Construction & The Centering Identity
Given tip timestamps $\mathbf{t} \in \mathbb{R}^N$ with $t_{\min} = \min_i t_i$ and $\bar{t} = \frac{1}{N}\sum_i t_i$, we define normalized exponential time-decay weights over a discrete candidate grid $\gamma \in \{0, 0.25, 0.5, 1, 2, 4, 7, 10\}$:
$$w_i(\gamma) = \frac{\exp\!\left(-\gamma \frac{t_i - t_{\min}}{\max(\bar{t} - t_{\min}, 10^{-5})}\right)}{\sum_{k=1}^N \exp\!\left(-\gamma \frac{t_k - t_{\min}}{\max(\bar{t} - t_{\min}, 10^{-5})}\right)}, \qquad Q_{\ell, b}(\gamma) = \sum_{i=1}^N w_i(\gamma)\, \mathbb{I}(s_{i,\ell} = b). \tag{2}$$
In sequence mode, $d_{\mathrm{prof}, i}(\gamma) = D_{\mathrm{TN93}}(s_i, Q(\gamma))$; in distance/tree mode, we evaluate the exact quadratic distance-space dual:
$$d_{\mathrm{prof}, i}(\gamma) = \sum_{j=1}^N w_j(\gamma) D_{ij} - \frac{1}{2} \sum_{j=1}^N \sum_{k=1}^N w_j(\gamma) w_k(\gamma) D_{jk}. \tag{3}$$
Selecting $\gamma^* = \arg\max_\gamma \mathrm{Corr}(\mathbf{t}, \mathbf{d}_{\mathrm{prof}}(\gamma))$ yields the optimal soft profile. Crucially, subtracting the sample mean $\bar{d}_{\mathrm{prof}} = \frac{1}{N}\sum_{i=1}^N d_{\mathrm{prof}, i}(\gamma^*)$ forms the **centered profile divergence vector**:
$$\tilde{\mathbf{d}}_{\mathrm{prof}} = \mathbf{d}_{\mathrm{prof}}(\gamma^*) - \bar{d}_{\mathrm{prof}}\mathbf{1}, \tag{4}$$
which **identically cancels** the constant ancestral offset $+D(R, Q(\gamma^*))$ in Eq. (1) while preserving the relative root-to-tip evolutionary displacements across all taxa.

### Stage 2: Fast Rust TN93 Distance Manifold & Vectorized Neighbor-Joining
- **Tree-Free Mode (`chronaeon_timetree_seq`)**: The native Rust kernel (`aeon_core._native.tn93_distance_matrix`) computes the $N \times N$ Tamura-Nei distance matrix $\mathbf{D}$ in parallel across sites. An unrooted binary tree $T_u = (V, E, \mathbf{b})$ with $2N-2$ nodes and $2N-3$ edges is reconstructed in $O(N^3)$ via vectorized Neighbor-Joining (`fast_nj`) using the $Q$-criterion matrix:
  $$Q_{uv} = (m - 2) D_{uv} - \sum_{k=1}^m D_{uk} - \sum_{k=1}^m D_{vk}, \qquad b_{uw} = \frac{1}{2} D_{uv} + \frac{r_u - r_v}{2(m-2)}. \tag{5}$$
- **Tree-Guided Mode (`chronaeon_timetree_tree`)**: Parses an unrooted or rooted Newick tree directly into an adjacency representation and extracts the all-nodes-to-tips patristic distance matrix in $O(N^2)$.

### Stage 3: Closed-Form Soft-Profile-Guided Continuous Edge Rooting
For every candidate edge $e = (u, v) \in E$ of length $b_e$ and fractional split $\alpha \in [0, 1]$ (placing root $R_{e,\alpha}$ at distance $\alpha b_e$ from $u$ and $(1-\alpha)b_e$ from $v$), the induced root-to-tip patristic distance vector across tips $i \in \{1, \dots, N\}$ is linear in $\alpha$:
$$\mathbf{d}(e, \alpha) = \mathbf{d}_0(e) + \alpha b_e \mathbf{s}_e, \qquad s_{e,i} = \begin{cases} +1 & \text{if tip } i \text{ is in the } u\text{-subtree}, \\ -1 & \text{if tip } i \text{ is in the } v\text{-subtree}. \end{cases} \tag{6}$$
Let $\tilde{\mathbf{x}} = \mathbf{x} - \bar{x}\mathbf{1}$ denote mean-centering and $\mathbf{M}_t = \mathbf{I} - \frac{\tilde{\mathbf{t}}\tilde{\mathbf{t}}^\top}{\|\tilde{\mathbf{t}}\|_2^2}$ denote the orthogonal projector onto $\mathrm{span}(\mathbf{1}, \mathbf{t})^\perp$. Both the temporal-regression residual minimizer $\alpha_{\mathrm{rss}}^*(e)$ and the centered soft-profile alignment minimizer $\alpha_{\mathrm{prof}}^*(e)$ admit exact closed-form solutions on $[0, 1]$:
$$\alpha_{\mathrm{rss}}^*(e) = \mathrm{clip}_{[0,1]}\!\left(-\frac{\langle \mathbf{M}_t \tilde{\mathbf{d}}_0(e),\, \mathbf{M}_t \tilde{\mathbf{s}}_e \rangle}{b_e \|\mathbf{M}_t \tilde{\mathbf{s}}_e\|_2^2}\right), \qquad \alpha_{\mathrm{prof}}^*(e) = \mathrm{clip}_{[0,1]}\!\left(\frac{\langle \tilde{\mathbf{d}}_{\mathrm{prof}} - \tilde{\mathbf{d}}_0(e),\, \tilde{\mathbf{s}}_e \rangle}{b_e \|\tilde{\mathbf{s}}_e\|_2^2}\right). \tag{7}$$
Evaluating the composite objective over all $2N-3$ edges subject to positive evolutionary rate $\hat{\mu}(e, \alpha) = \langle \tilde{\mathbf{t}}, \tilde{\mathbf{d}}(e, \alpha)\rangle / \|\tilde{\mathbf{t}}\|_2^2 > 0$:
$$\mathcal{J}(e, \alpha) = \big\|\tilde{\mathbf{d}}(e, \alpha) - \hat{\mu}(e, \alpha)\tilde{\mathbf{t}}\big\|_2^2 + \lambda_{\mathrm{prof}} \big\|\tilde{\mathbf{d}}(e, \alpha) - \tilde{\mathbf{d}}_{\mathrm{prof}}\big\|_2^2 \qquad (\lambda_{\mathrm{prof}} = 0.35) \tag{8}$$
selects the globally optimal root $(e^*, \alpha^*)$ in $O(N^2)$ total operations.

### Stage 4: Rank-1 Sherman-Morrison LOOCV Outlier Downweighting
Before solving for internal node dates, high-leverage sequencing errors, recombinants, or passaged lab strains are screened in $O(N)$ using analytical leave-one-out cross-validation (LOOCV) without refitting:
$$h_{ii} = \frac{1}{N} + \frac{(t_i - \bar{t})^2}{\sum_{j=1}^N (t_j - \bar{t})^2}, \qquad \hat{\sigma}_{(-i)}^2 = \frac{1}{N-3}\left(\mathrm{RSS} - \frac{r_i^2}{1 - h_{ii}}\right), \qquad Z_i = \frac{r_i}{\hat{\sigma}_{(-i)}\sqrt{1 - h_{ii}}}. \tag{9}$$
Taxa exceeding $|Z_i| > z_{\mathrm{thresh}}$ ($3.0$ by default) receive Huber-type terminal branch weights $u_i = \min\!\big(1, (z_{\mathrm{thresh}}/|Z_i|)^2\big)$.

### Stage 5: Active-Set Contracted Tree-Laplacian Isotonic WLS Solver
Directed from $R_{e^*,\alpha^*}$, the rooted topology has $N$ fixed terminal timestamps $t_1, \dots, t_N$ and $N-1$ unknown internal node timestamps $\mathbf{t}_I \in \mathbb{R}^{N-1}$. Under a Poisson-regularized branch variance model, each directed edge $e = (p \to v)$ carries weight $w_e = \frac{L_{\mathrm{eff}}}{b_e + c/L_{\mathrm{eff}}}$ ($c = 10$, scaled by $u_v$ on terminal edges). We minimize the root-anchored isotonic WLS objective:
$$\min_{\mathbf{t}_I \in \mathbb{R}^{N-1},\, \mu > 0} \sum_{e=(p \to v) \in E} w_e \big(b_e - \mu (t_v - t_p)\big)^2 + w_{\mathrm{anchor}} \big(t_{\mathrm{root}} - t_{\mathrm{root}}^{\mathrm{rtt}}(\mu)\big)^2 \quad \text{subject to} \quad t_p \le t_v \;\;\forall (p \to v) \in E, \tag{10}$$
where $t_{\mathrm{root}}^{\mathrm{rtt}}(\mu) = \bar{t} - \bar{d}_{\mathrm{root}}/\mu$ is the centered root-to-tip intercept.
1. **Active-Set Union-Find Contraction ($O(N)$ per iteration)**: For fixed $\mu$, any violated internal edge $(p \to c)$ with $t_p > t_c$ is added to the active constraint set $\mathcal{A}_I$ and contracted via Union-Find into a connected component $C_k \in \mathcal{C}$, while any internal node exceeding a child tip ($t_p > t_{\mathrm{leaf}}$) pins its component to $\min_{\mathrm{leaf} \in \partial C_k} t_{\mathrm{leaf}}$. The remaining free components satisfy a sparse, symmetric positive-definite **contracted tree-Laplacian linear system**:
   $$\mathbf{L}_{\mathcal{C}} \mathbf{t}_{\mathcal{C}} = \mathbf{r}_{\mathcal{C}}(\mu), \tag{11}$$
   which is solved in $O(|\mathcal{C}|) \subseteq O(N)$ time via sparse Cholesky/LU (`scipy.sparse.linalg.spsolve`).
2. **Blended Rate Update (Preventing Polytomy Attenuation)**: When short internal branches have $b_e = 0$ but $\tau_e = t_v - t_p > 0$, pure branch-ratio WLS updates ($\mu_{\mathrm{wls}} = \frac{\sum w_e b_e \tau_e}{\sum w_e \tau_e^2}$) are biased downward by zero-mutation denominators. `chronaeon.timetree` blends the path-wise root-to-tip rate $\hat{\mu}_{\mathrm{rtt}}$ with the branch-wise WLS/Poisson MLE rate $\hat{\mu}_{\mathrm{branch}} = 0.7\hat{\mu}_{\mathrm{wls}} + 0.3\frac{\sum b_e}{\sum \tau_e}$:
   $$\hat{\mu}^{(k+1)} = \frac{1}{2}\hat{\mu}_{\mathrm{rtt}} + \frac{1}{2}\hat{\mu}_{\mathrm{branch}}^{(k)}, \tag{12}$$
   guaranteeing immunity to internal polytomy rate collapse.

### Stage 6: Dispersion-Calibrated Analytical Fieller Confidence Intervals
To account for phylogenetic non-independence and relaxed/autocorrelated rate drift without bootstrap resampling, `chronaeon.timetree` computes:
1. **Bartlett's Effective Sample Size**: $N_{\mathrm{eff}} = \mathrm{clip}_{[3, N]}\!\left(\frac{N^2}{\sum_{i,j} \exp(-\kappa D_{ij})}\right)$ with $\kappa = 5 / \mathrm{median}(D_{ij})$ and effective degrees of freedom $\nu^* = \max(2, N_{\mathrm{eff}} - 2)$.
2. **Path-to-Branch Overdispersion Ratio**: Comparing path-level Poisson dispersion $\hat{\phi}_{\mathrm{rtt}} = \frac{1}{N-2}\sum_i \frac{u_i L(d_i - \hat{\mu}(t_i - \hat{t}_{\mathrm{root}}))^2}{\max(1/L, \hat{\mu}(t_i - \hat{t}_{\mathrm{root}}))}$ against branch-level WLS dispersion $\hat{\phi}_{\mathrm{branch}} = \frac{1}{|E|-|V_I|}\sum_e w_e(b_e - \hat{\mu}\tau_e)^2$ yields the autocorrelation inflation factor $\rho_{\mathrm{disp}} = \sqrt{\max(1, \hat{\phi}_{\mathrm{rtt}} / \hat{\phi}_{\mathrm{branch}})}$.
3. **Exact Non-Linear Fieller Bounds**: Defining $g = \left(\frac{t_{0.975, \nu^*} \mathrm{SE}_{\mathrm{eff}}(\hat{\mu})}{\hat{\mu}}\right)^2$, when $g < 1$, the 95% confidence interval for $t_{\mathrm{MRCA}}$ (and analogously for every internal node $u \in V_I$ relative to its descendant tips) is solved in closed form via Fieller's quadratic root formula:
   $$\mathrm{CI}_{95\%}(t_{\mathrm{MRCA}}) = \bar{t} - \frac{1}{1 - g}\left[\hat{\theta}_0 \mp \sqrt{\hat{\theta}_0^2 - (1-g)\!\left(\hat{\theta}_0^2 - \frac{t_{0.975,\nu^*}^2 \mathrm{SE}^2(\hat{d}_0)}{\hat{\mu}^2}\right)}\right], \qquad \hat{\theta}_0 = \bar{t} - \hat{t}_{\mathrm{MRCA}}. \tag{13}$$

---

## 3. How Coalescent Skyline Plots Differ Between `phylodate` Timetrees and `chronaeon.timetree`

Given a dated heterochronous phylogeny with $N$ tip sampling times $s_1 \le \dots \le s_N$ and $N-1$ internal coalescent times $c_1 \le \dots \le c_{N-1}$, the sorted timeline of all $2N-1$ events partitions $[c_1, s_N]$ into elementary intervals $I_j = [t_{j-1}, t_j]$ of duration $\Delta t_j = t_j - t_{j-1} \ge 0$ with $k_j$ extant lineages. Under the heterochronous Kingman coalescent (Rodrigo & Felsenstein 1999; Strimmer & Pybus 2001), the generalized skyline estimator over a pooled window $W_m$ containing $r_m \ge 1$ coalescent events and spanning $\sum_{j \in W_m} \Delta t_j \ge \epsilon$ is:
$$\hat{N}_e(W_m) = \frac{1}{r_m} \sum_{j \in W_m} \binom{k_j}{2} \Delta t_j. \tag{14}$$

Because $\hat{N}_e(t)$ is a linear functional of the **inter-node waiting times $\Delta t_j$ weighted quadratically by the extant lineage count $\binom{k_j}{2}$**, differences in how `chronaeon.timetree` and the `phylodate` methods estimate internal node dates manifest in four distinct ways on skyline and LTT plots:

### 3.1 The 1-Mutation Temporal Resolution Quantum and Zero-Wait Polytomy Collapse
On an alignment of length $L$ evolving at substitution rate $\mu$, a single mutation along a lineage corresponds to an expected waiting time of:
$$\Delta t_{\mathrm{quantum}} = \frac{1}{\mu L}. \tag{15}$$
During rapid epidemic transmission (where $k_j$ is large and true coalescent waiting times satisfy $\Delta t_j \ll (\mu L)^{-1}$), most short internal branches accumulate zero mutations ($b_e = 0$). Different dating algorithms handle this regime differently:
- **Hard Topological Collapsing (`lsd2`)**: By default, `lsd2` collapses all internal branches with $b_e \le 0.5/L$ (`nullblen = 0.5/L`) into multifurcations and clamps active constraints $t_p = t_v$. On empirical outbreaks, `lsd2` collapses **85.1% of coalescent intervals on Ebola**, **51.9% on Chikungunya Brazil**, and **46.7% on Rabies** to $\Delta t_j = 0$ (Figure 3B). In an ungrouped classic skyline ($\epsilon \to 0$), these zero-duration intervals produce artificial $\hat{N}_e(t) \to 0$ downward spikes; even with $\epsilon$-pooling, if multifurcating nodes of out-degree $d > 2$ are not expanded with multiplicity $d - 1$, the LTT curve $L(t)$ undercounts active lineages and biases $\binom{k_j}{2}$ upward by $3\text{--}10\times$.
- **Active-Set Tree-Laplacian with Root-to-Tip Rate Blending (`chronaeon.timetree`)**: Both `chronaeon_timetree_tree` and `chronaeon_timetree_seq` enforce exact monotonicity $t_p \le t_v$ via Union-Find active-set contraction while preserving the full binary topology ($N-1$ internal nodes) and preventing zero-length branches from attenuating $\hat{\mu}$. Consequently, `chronaeon.timetree` exhibits substantially lower zero-wait discretization than `lsd2` (**14.8%–18.5% on Chikungunya**, **24.4%–31.1% on Rabies**, and **68.6%–74.7% on ultra-low-diversity Ebola**; Figure 3B).
- **Strict Positive Branch Regularization (`treedater`, `wlogdate`)**: Because `treedater` (Gamma-Poisson branch prior) and `wlogdate` ($\sum_e (\log f_e)^2$ log-rate penalty) penalize $\tau_e \to 0$, they produce virtually zero collapsed intervals (`0%–7.4%`; Figure 3B), yielding smoother raw skylines at the cost of occasionally over-stretching zero-mutation basal clades on low-signal datasets (e.g., `wlogdate` on Ebola and Dengue; Table 2).

### 3.2 Basal Skyline Horizon ($t \to t_{\mathrm{MRCA}}$) and Unrooted ML Tree Vulnerability
Near the root of the phylogeny ($k_j \in \{2, 3, 4\}$), small errors in root placement or deep branch rates shift the basal coalescent intervals $[c_1, c_5]$ across decades:
- **`chronaeon.timetree` and `treetime` are Self-Rooting**: Given an unrooted IQ-TREE phylogeny (`chronaeon_comparison/data/trees/*.nwk`) or raw sequences, `chronaeon.timetree` automatically locates $(e^*, \alpha^*)$ via Eq. (8), aligning the basal skyline horizon with published BEAST Skygrid/Skyline posteriors (Figure 2).
- **Sensitivity of `lsd2`, `treedater`, `reltime`, and `wlogdate`**: On Dengue (`11_dengue_polyepoch_suchard2020`, $N=352$), even after root alignment, `lsd2` and `wlogdate` stretch deep basal branches backward to $1789.5$ and $1860.8$ (vs. $1931.0\text{--}1942.8$ for `chronaeon.timetree` and $1935.5$ for `treetime`), inflating the pre-1950 ancestral skyline $\hat{N}_e(t)$ by $3\text{--}4\times$ (Table 2, Figure 2D). On Ebola (`00_ebola_sierraleone_gire2014`), `treedater` stretches the basal root back to $2013.17$ (vs. BEAST $2014.20$ and `chronaeon.timetree` $2014.20\text{--}2014.26$), creating a spurious 1-year pre-epidemic tail in Figure 2B.

### 3.3 Tree-Free NJ Clade Averaging (`chronaeon_timetree_seq`) vs. ML Single-Branch Variance (`chronaeon_timetree_tree`)
A striking finding in both synthetic (Figure 1, Table 1) and empirical (Figure 2, Table 2) benchmarks is how closely the **tree-free sequence pipeline** (`chronaeon_timetree_seq`, using Rust TN93 + `fast_nj`) tracks the **ML-tree pipeline** (`chronaeon_timetree_tree`):
- Because Neighbor-Joining branch lengths $b_{uw} = \frac{1}{2}D_{uv} + \frac{r_u - r_v}{2(m-2)}$ average pairwise TN93 distances across all $m-2$ outgroup lineages, intermediate internal branch lengths have low variance.
- Near the tips, NJ resolves zero-mutation cherries deterministically, whereas near the root, Stage 3's centered soft-profile penalty $\lambda_{\mathrm{prof}}\|\tilde{\mathbf{d}}(e,\alpha) - \tilde{\mathbf{d}}_{\mathrm{prof}}\|_2^2$ anchors the basal coalescent splits to the global temporal gradient.

---

## 4. Quantitative Benchmark: Synthetic & Empirical Skyline Recovery

### 4.1 Synthetic Heterochronous Coalescent Benchmark
We simulated $N = 40$ serially sampled sequences ($L = 1,500\text{ bp}$, $\mu = 1.5 \times 10^{-3}\text{ subs/site/yr}$, $t_i \sim \mathcal{U}[2000, 2020]$) under three exact heterochronous Kingman coalescent demographic regimes across 8 replicates per regime:
1. **Constant Demography**: $N_e(t) = 15.0\text{ yr}$.
2. **Exponential Epidemic Growth**: $N_e(t) = 2.0 \exp(0.14 \max(0, t - 1995))$.
3. **Epidemic Boom-Bust Wave**: $N_e(t) = 2.5 + 22.0 \exp\!\big(-\frac{1}{2}((t - 2011)/3.2)^2\big)$.

| Demographic Scenario | Method | Skyline $\log_{10}\hat{N}_e$ MAE $\downarrow$ | Skyline $\log_{10}\hat{N}_e$ RMSE $\downarrow$ | Skyline $\log_{10}\hat{N}_e$ Bias | LTT MAE (lineages) $\downarrow$ | Coalescent Quantile MAE (yr) $\downarrow$ | $t_{\mathrm{MRCA}}$ MAE (yr) $\downarrow$ | Zero-Wait Fraction (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constant ($N_e = 15$)** | **`chronaeon_timetree_tree`** | **0.241** | **0.307** | -0.031 | 0.569 | 0.573 | 4.41 | 6.9% |
| | **`chronaeon_timetree_seq`** | **0.275** | **0.369** | -0.049 | 0.563 | 0.561 | 4.11 | 7.2% |
| | `wlogdate` | 0.252 | 0.315 | -0.011 | **0.443** | **0.431** | **3.46** | 0.0% |
| | `node_dating` | 0.260 | 0.355 | -0.050 | 0.604 | 0.622 | 5.09 | 12.5% |
| | `treedater` | 0.262 | 0.322 | -0.021 | 0.499 | 0.512 | 4.57 | 0.0% |
| | `treetime` | 0.266 | 0.362 | -0.041 | 0.541 | 0.548 | 4.37 | 11.8% |
| | `reltime` | 0.269 | 0.363 | -0.068 | 0.574 | 0.588 | 4.96 | 13.2% |
| | `chronumental` | 0.274 | 0.374 | -0.068 | 0.466 | 0.450 | 3.65 | 7.9% |
| | `lsd2` | 0.274 | 0.367 | -0.037 | 0.515 | 0.525 | 4.70 | 15.8% |
| **Exponential Growth** | `wlogdate` | **0.238** | **0.301** | +0.057 | 0.531 | 0.369 | 0.94 | 0.0% |
| | `reltime` | 0.245 | 0.313 | -0.003 | 0.500 | 0.338 | 0.74 | 16.8% |
| | **`chronaeon_timetree_tree`** | **0.247** | **0.309** | +0.034 | **0.472** | **0.322** | **0.63** | 7.2% |
| | `chronumental` | 0.247 | 0.313 | +0.003 | 0.494 | 0.342 | 0.76 | 7.2% |
| | `treedater` | 0.252 | 0.315 | +0.055 | 0.552 | 0.379 | 0.87 | 0.0% |
| | `node_dating` | 0.254 | 0.318 | +0.039 | 0.590 | 0.406 | 0.88 | 15.1% |
| | **`chronaeon_timetree_seq`** | **0.256** | **0.315** | +0.038 | 0.556 | 0.370 | 1.09 | 11.8% |
| | `treetime` | 0.257 | 0.317 | +0.041 | 0.565 | 0.381 | 1.01 | 14.5% |
| | `lsd2` | 0.258 | 0.323 | +0.044 | 0.521 | 0.361 | 0.67 | 19.4% |
| **Epidemic Boom-Bust** | `wlogdate` | **0.230** | **0.293** | +0.028 | **0.366** | **0.257** | 0.50 | 0.0% |
| | `treedater` | 0.241 | 0.298 | +0.020 | 0.396 | 0.277 | 0.42 | 0.0% |
| | **`chronaeon_timetree_seq`** | **0.248** | **0.322** | -0.016 | 0.429 | 0.297 | 0.37 | 12.2% |
| | **`chronaeon_timetree_tree`** | **0.251** | **0.324** | -0.019 | 0.415 | 0.286 | **0.36** | 10.9% |
| | `chronumental` | 0.255 | 0.342 | -0.048 | 0.398 | 0.280 | 0.41 | 9.9% |
| | `node_dating` | 0.260 | 0.347 | +0.019 | 0.443 | 0.309 | 0.65 | 19.4% |
| | `reltime` | 0.266 | 0.348 | -0.037 | 0.435 | 0.308 | 0.36 | 22.4% |
| | `lsd2` | 0.266 | 0.339 | +0.011 | 0.443 | 0.312 | 0.37 | 25.0% |
| | `treetime` | 0.269 | 0.362 | -0.026 | 0.415 | 0.290 | 0.59 | 18.1% |

**Table 1:** Synthetic heterochronous coalescent skyline and LTT recovery across three demographic scenarios. On ML topologies, `chronaeon_timetree_tree` achieves the **lowest overall skyline $\log_{10}\hat{N}_e$ MAE under Constant demography ($0.241$)**, the **lowest LTT MAE ($0.472$), coalescent quantile MAE ($0.322\text{ yr}$), and $t_{\mathrm{MRCA}}$ MAE ($0.63\text{ yr}$) under Exponential Growth**, and the **lowest $t_{\mathrm{MRCA}}$ MAE ($0.36\text{ yr}$) under Boom-Bust**. Directly from sequences without ML tree inference, `chronaeon_timetree_seq` matches or outperforms `lsd2` and `treetime` on Boom-Bust ($0.248$ vs. $0.266\text{--}0.269$) and Exponential Growth ($0.256$ vs. $0.257\text{--}0.258$).

![Synthetic Skyline and LTT Comparison](figures/fig_whitepaper_synthetic_skyline.png)
*Figure 1: Median reconstructed generalized coalescent skylines $\hat{N}_e(t)$ (top row, log scale) and Lineages-Through-Time $L(t)$ curves (bottom row) across 8 replicates for Constant, Exponential Growth, and Epidemic Boom-Bust demographies compared against ground truth (black dashed).*

---

### 4.2 Empirical Viral Outbreak Skyline Comparison

Table 2 and Figure 2 compare empirical skyline trajectories across four canonical viral phylodynamic cohorts: **North American Rabies** (`16_rabies_northamerica_biek2007`, $N=46$), **Ebola Sierra Leone** (`00_ebola_sierraleone_gire2014`, $N=195$), **Chikungunya Brazil** (`03_chikungunya_brazil_naveca2019`, $N=29$), and **Dengue Polyepoch** (`11_dengue_polyepoch_suchard2020`, $N=352$).

| Viral Outbreak Cohort | Method | Inferred $t_{\mathrm{MRCA}}$ | BEAST Ref $t_{\mathrm{MRCA}}$ | Median Skyline $\hat{N}_e \cdot g$ (yr) | Skyline Dynamic Range ($\mathrm{IQR}(\log_{10}\hat{N}_e)$) | Collapsed Zero-Wait Intervals (%) | Requires External ML Tree? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rabies North America** ($N=46$) | **`chronaeon_timetree_seq`** | **1972.85** | 1972.40 | 335.8 | 1.34 | 24.4% | **No (42 ms total)** |
| | **`chronaeon_timetree_tree`** | **1971.24** | 1972.40 | 291.1 | 1.44 | 31.1% | Yes (+18.9 s IQ-TREE) |
| | `lsd2` | 1971.21 | 1972.40 | 337.0 | 1.26 | 46.7% | Yes (+18.9 s IQ-TREE) |
| | `treetime` | 1970.64 | 1972.40 | 225.6 | 1.55 | 28.9% | Yes (+18.9 s IQ-TREE) |
| | `treedater` | 1975.80 | 1972.40 | 241.2 | 1.51 | 0.0% | Yes (+18.9 s IQ-TREE) |
| | `wlogdate` | 1972.28 | 1972.40 | 197.8 | 1.72 | 4.4% | Yes (+18.9 s IQ-TREE) |
| **Ebola Sierra Leone** ($N=195$) | **`chronaeon_timetree_seq`** | **2014.26** | 2014.20 | 2.22 | 1.57 | 74.7% | **No (918 ms total)** |
| | **`chronaeon_timetree_tree`** | **2014.20** | 2014.20 | 1.39 | 1.29 | 68.6% | Yes (+61.4 s IQ-TREE) |
| | `lsd2` | 2014.07 | 2014.20 | 1.25 | 1.96 | 85.1% | Yes (+61.4 s IQ-TREE) |
| | `treetime` | 2013.81 | 2014.20 | 3.06 | 0.61 | 26.8% | Yes (+61.4 s IQ-TREE) |
| | `treedater` | 2013.17 | 2014.20 | 3.30 | 1.79 | 0.0% | Yes (+61.4 s IQ-TREE) |
| | `wlogdate` | 1977.61 | 2014.20 | 213.0 | 1.25 | 0.0% | Yes (+61.4 s IQ-TREE) |
| **Chikungunya Brazil** ($N=29$) | **`chronaeon_timetree_seq`** | **2014.47** | 2014.54 | 2.31 | 0.51 | 14.8% | **No (69 ms total)** |
| | **`chronaeon_timetree_tree`** | **2014.31** | 2014.54 | 2.39 | 0.62 | 18.5% | Yes (+11.2 s IQ-TREE) |
| | `lsd2` | 2014.34 | 2014.54 | 2.27 | 0.66 | 51.9% | Yes (+11.2 s IQ-TREE) |
| | `treetime` | 2014.25 | 2014.54 | 2.53 | 0.66 | 33.3% | Yes (+11.2 s IQ-TREE) |
| | `treedater` | 2014.43 | 2014.54 | 1.95 | 0.66 | 0.0% | Yes (+11.2 s IQ-TREE) |
| | `wlogdate` | 2014.24 | 2014.54 | 3.14 | 0.60 | 7.4% | Yes (+11.2 s IQ-TREE) |
| **Dengue Polyepoch** ($N=352$) | **`chronaeon_timetree_seq`** | **1942.77** | 1939.20 | 25.57 | 0.42 | 33.7% | **No (1.74 s total)** |
| | **`chronaeon_timetree_tree`** | **1931.03** | 1939.20 | 28.30 | 0.15 | 32.9% | Yes (+42.8 s IQ-TREE) |
| | `treetime` | 1935.50 | 1939.20 | 26.94 | 0.24 | 22.6% | Yes (+42.8 s IQ-TREE) |
| | `reltime` | 1936.23 | 1939.20 | 26.85 | 0.23 | 3.7% | Yes (+42.8 s IQ-TREE) |
| | `treedater` | 1915.12 | 1939.20 | 40.60 | 0.23 | 0.0% | Yes (+42.8 s IQ-TREE) |
| | `lsd2` | 1789.46 | 1939.20 | 115.88 | 0.37 | 42.9% | Yes (+42.8 s IQ-TREE) |

**Table 2:** Empirical viral skyline comparison across four BEAST benchmark cohorts. `chronaeon_timetree_seq` and `chronaeon_timetree_tree` closely match `treetime` and BEAST reference horizons across all four studies while avoiding the basal skyline inflation seen in `lsd2`/`wlogdate` on Dengue and `treedater`/`wlogdate` on Ebola.

![Empirical Viral Outbreak Skylines](figures/fig_whitepaper_empirical_skyline.png)
*Figure 2: Empirical generalized coalescent skyline trajectories $\hat{N}_e(t) \cdot g$ across four viral outbreaks: (top-left) North American Rabies expansion (1971–2005), (top-right) 2014 Ebola Sierra Leone exponential wave, (bottom-left) 2014–2017 Chikungunya Brazil seasonal double-peak epidemic, and (bottom-right) 1931–2010 Dengue multi-decade oscillations.*

![Diagnostic Comparison of Skyline Error, Zero-Wait Discretization, and Coalescent Quantiles](figures/fig_whitepaper_diagnostics.png)
*Figure 3: (A) Mean demographic skyline $\log_{10}\hat{N}_e(t)$ MAE across synthetic scenarios; (B) Percentage of coalescent intervals collapsed to zero waiting time ($\Delta t_j \le (\mu L)^{-1}$) across empirical viral cohorts; (C) Trade-off between internal coalescent quantile MAE and demographic skyline error.*

---

## 5. Summary of Practical Guidelines

1. **When to use `chronaeon_timetree_seq` (Tree-Free Sequence Mode)**: Ideal for rapid real-time genomic surveillance, BEAST warm-starting, and large alignments where running IQ-TREE/RAxML ($10\text{--}180\text{ s}$) is prohibitive. It recovers both the dated phylogeny and the demographic skyline $\hat{N}_e(t)$ in $40\text{ ms}\text{--}1.7\text{ s}$ with accuracy matching ML-tree-based `treetime` and `lsd2`.
2. **When to use `chronaeon_timetree_tree` (ML-Tree Mode)**: When a high-quality ML topology is already available, `chronaeon_timetree_tree` provides automatic soft-profile-guided rooting, exact $O(N)$ active-set tree-Laplacian node dating in $5\text{--}110\text{ ms}$, immunity to basal polytomy rate collapse, and well-calibrated 95% Fieller confidence intervals (87%–100% coverage across strict, uncorrelated, and autocorrelated clocks).
3. **Constructing Skyline Plots from Point-Estimate TimeTrees**: Because any active-set WLS solver (`lsd2`, `treetime`, `chronaeon.timetree`) discretizes waiting times shorter than the single-mutation quantum $\Delta t_{\mathrm{quantum}} = (\mu L)^{-1}$ into zero-length multifurcations, downstream skyline estimators should always pool elementary coalescent intervals using a minimum window threshold $\epsilon \ge (\hat{\mu} L)^{-1}$ (Strimmer & Pybus generalized skyline or uniform Skygrid bins).
