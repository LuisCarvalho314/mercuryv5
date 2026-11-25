# Gradient-based CSCG: derivation and controlled comparison with EM

Companion to `docs/online_jepa_cscg_formulation.md` (notation of Sections 1–3 is reused).
Tests: `tests/src/mercury_runs/test_cscg_gradient.py`. Benchmark: `scripts/run_cscg_em_vs_gradient.sh`,
report `scripts/analysis/cscg_em_vs_gradient_report.py`.

| `ExperimentConfig.optimizer` | Role | Implementation |
| --- | --- | --- |
| `em` | baseline (unchanged NumPy EM) | `distillation.py` |
| `gradcscg` | **primary gradient arm**: reference-style gradCSCG (Section 3) | `torch_gradient.GradCSCG` |
| `projected_adam` | secondary: Adam directly on $T$ + simplex projection (Section 4) | `torch_gradient.ProjectedAdamCSCG` |
| `projected_gradient` | secondary diagnostic: NumPy analytic gradient + projection + Armijo (Section 4) | `distillation.projected_gradient_step` |

The primary arm follows the reference implementation, `tristanstoeber/gradCSCG`
(`models/gradient_hmm.py`, `models/vqvae_cscg.py`). It uses trainable transition logits, a row-wise
softmax, the batched log-space forward recursion `GradientHMM._batch_neg_log_likelihood`,
autodiff, and standard Adam. TensorFlow is replaced by PyTorch; the deviations are listed in
Section 3.

## 1. The shared per-arrival objective

The comparison uses the windowed ("long") online EM learner. At arrival $t$ let
$s = \max(e(t), t-W+1)$, $n = t-s+1$ observations, $n-1$ transitions, and let $h = f_s$ be the
**cached historical filter belief** at the window start (computed once at arrival $s$ under
$T_s$ and never revised). The window evidence is

$$
Z_t(T) = p_T(x_{s+1:t} \mid h, a_{s:t-1})
       = \sum_{z_{s:t}} h(z_s) \prod_{u=s}^{t-1} T_{a_u, z_u z_{u+1}} e_{u+1}(z_{u+1}),
$$

and the per-arrival objective (the long-window analogue of $G^{\mathrm{EM}}_t$, Section 3.2 of the
companion document) is

$$
G_t(T) = -\big\langle \lambda S_t + \kappa \mathbf 1, \log T \big\rangle
         \;+\; \frac{\gamma_\lambda}{n-1}\big(-\log Z_t(T)\big).
$$

* The **window term** is an exact observation-level NLL over a *finite* window, conditional on
  the carried boundary belief.
* The **history term** is the accumulated sufficient statistics: $S_t$ is the sum of the
  window-mean E-step counts committed at earlier arrivals (each computed once, under the tensor
  in force then), plus the Dirichlet pseudocount $\kappa$.

The two are kept distinct on purpose. Optimising $-\log Z_t$ alone, the finite-window objective,
would discard every transition older than $W$. The gradient arm instead uses **exactly** $G_t$,
with the same $S_t$ recursion, $\kappa$, $\lambda$ and $\gamma_\lambda$ as EM, so neither arm loses
historical evidence. Exact full-history NLL (re-running forward–backward over $x_{0:t}$ under the
current $T$ at every arrival) is a third possibility, but it costs $O(t)$ per arrival and neither
arm uses it.

The history term is first-order exact for the past likelihood at the iterate where each count
was computed (Fisher's identity, $\nabla \log p = \nabla Q$ at the E-step point). It is not
the past NLL at the current $T$.

## 2. Gradient of the window NLL and its relation to the E-step counts

Let $\alpha_u$ and $\beta_u$ be the normalised forward and backward messages of Definition 5
(companion document), restricted to the clones $C(x_u)$. The unnormalised versions satisfy
$Z = \tilde\alpha_u^\top T_{a_u} \tilde\beta_{u+1}$ for every $u$, and $Z$ is multilinear in the
transition entries. Differentiating each factor:

$$
\frac{\partial Z}{\partial T_{a,ij}} = \sum_{u:\,a_u=a} \tilde\alpha_u(i)\, \tilde\beta_{u+1}(j),
\qquad
\frac{\partial(-\log Z)}{\partial T_{a,ij}}
 = -\sum_{u:\,a_u = a} \frac{\alpha_u(i)\, \beta_{u+1}(j)}{\alpha_u^\top T_{a_u} \beta_{u+1}} .
$$

The ratio is invariant to the per-step scalings, so the normalised messages already computed by
the existing forward–backward routine give the exact gradient. The two-slice posterior of the
E-step is

$$
\xi_u(i,j) = \frac{\alpha_u(i)\, T_{a_u,ij}\, \beta_{u+1}(j)}{\alpha_u^\top T_{a_u} \beta_{u+1}},
\qquad X = \frac{1}{n-1}\sum_u \xi_u \quad (\text{unit mass}),
$$

hence the identity

$$
\boxed{\; T \odot \nabla_T\Big(\tfrac{-\log Z}{n-1}\Big) = -X \;}
$$

The gradient is the E-step count divided entrywise by $T$. Computed directly from the messages,
it stays **finite at $T_{a,ij} = 0$**, where $X_{a,ij} = 0$ but the gradient is generally nonzero.
That is what lets a gradient method move mass back onto a zero entry, which $X/T$ could not express.

The gradient of the full objective is therefore

$$
\nabla G_t(T) = -\frac{\lambda S_t + \kappa}{T} + \gamma_\lambda \nabla\Big(\tfrac{-\log Z_t}{n-1}\Big)
\;\overset{T>0}{=}\; -\frac{B(T)}{T},
\qquad B(T) = \lambda S_t + \gamma_\lambda X(T) + \kappa,
$$

where $B$ is exactly EM's count tensor. EM's M-step $\operatorname{rowNorm}(B(T^{(k)}))$ is the
exact minimiser of the majoriser $-\langle B(T^{(k)}), \log T\rangle$, which touches $G_t$ at
$T^{(k)}$ (Jensen). The gradient arms take first-order steps on the same $G_t$. All are descent
methods on one objective. $G_t$ is non-convex through $Z_t$, so they may reach different points.

## 3. Primary gradient arm: reference-style gradCSCG (`optimizer = "gradcscg"`)

**Parameterisation.** Trainable logits $W \in \mathbb R^{A\times N\times N}$ with
$T = \operatorname{softmax}_{\text{row}}(W)$, as in the reference (`GradientHMM.transition`). The
logits are initialised to $W = \log T_0$, where $T_0$ is EM's strictly positive initial tensor, so
both arms start from the same $T$ exactly. There is no projection, no gradient centring and no
step halving: softmax rows are always strictly positive and normalised.

**Inference.** The forward recursion is the reference's batched log-space recursion
(`_batch_neg_log_likelihood`): for padded $(O, A, L)$ of shape $[B, T]$,

$$
\log\alpha_{t+1}(j) = \operatorname{logsumexp}_i\big(\log\alpha_t(i) + \log T_{A_t, ij}\big) + \log B_{O_{t+1}}(j),
\qquad \mathrm{NLL}_b = -\operatorname{logsumexp}_j \log\alpha_{L_b-1}(j),
$$

with deterministic clone emissions, where $\log B$ is $0$ on $C(o)$ and $-10^{10}$ elsewhere, and
length masking by $L$. This is not the legacy single-sequence path, which takes `final_alpha * 10`.
It equals $-\log Z_t$ of Section 1. Inside $G_t$ it is divided by $n-1$, and it is used with
$B = 1$ (one window per arrival).

**Objective.** The shared $\bar G_t = G_t / M_t$ (Section 4.1), written in $\log T = \operatorname{log\_softmax}(W)$:
$\bar G_t = [-\langle \lambda S_t + \kappa, \log T\rangle + \gamma_\lambda\,\mathrm{NLL}/(n-1)] / M_t$. One
backward pass gives $\nabla_W \bar G_t$. Since $\partial \bar G_t / \partial \log T = -(\lambda S_t + \kappa + \gamma_\lambda X)/M_t$
(the log-space form of $T \odot \nabla = -X$), the E-step counts committed to $S$ are read off the
gradient retained on $\log T$, with no separate NumPy E-step.

**Optimiser.** `torch.optim.Adam` with the Keras defaults the reference uses: $\beta = (0.9, 0.999)$
and $\epsilon = 10^{-7}$. There is no clipping and no schedule, since the reference's optional
linear decay is off by default. The Adam state persists across local iterations and arrivals.
The reference uses learning rates $10^{-3}$ (`fit_hmm`), $10^{-2}$ (`finalize_hmm` and the
notebook's `hmm_lr`) and $3\times10^{-4}$ (joint training); the sweep spans
$\{3\times10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$.

**Deviations from the reference, all deliberate:**
* **Initial belief and sink state.** The initial log-belief is the carried boundary belief
  $\log h$ (restricted to $C(x_s)$), not a trainable $\log\operatorname{softmax}(\pi)$. There is no
  ancillary sink state.
* **No log epsilon.** $\log T$ is the exact `log_softmax`, not $\log(\operatorname{softmax} + 10^{-9})$,
  so the objective is exactly the shared $G_t$.
* **Loss.** The loss is the shared online objective: history counts, pseudocount and the current
  window. It is not minibatches of random sequence chunks. The per-arrival schedule (five Adam
  steps) is identical to EM's.
* **Precision.** float64 instead of float32, for comparability with the NumPy EM.

## 4. Secondary variants: optimisation directly over the simplex

### 4.1 Mass normalisation and the projection

All gradient arms work on the mass-normalised objective $\bar G_t = G_t / M_t$, with
$M_t = \langle \lambda S_t + \kappa, \mathbf 1\rangle + \gamma_\lambda$, the total mass of $B$, which is
independent of $T$. With $\lambda = 1$, $\bar G_t$ is a per-transition average, so the learning
rate is dimensionless and comparable across arrivals. $\Pi_\Delta$ is the row-wise Euclidean
projection onto the probability simplex, $\Pi_\Delta(v) = \max(v - \theta, 0)$, with $\theta$ chosen so
the row sums to one (exact, sort-based). It preserves non-negativity and row normalisation, and
it can set entries **exactly** to zero.

### 4.2 Projected gradient descent (`projected_gradient`, NumPy, diagnostic)

Each local iteration $k$:

1. Run one forward–backward pass at $T^{(k)}$. It gives $-\log Z_t$, $X^{(k)}$ and
   $g^{(k)} = \nabla \bar G_t(T^{(k)})$ (Section 2).
2. For $m = 0, 1, \dots, m_{\max}$, set $T^{+} = \Pi_\Delta(T^{(k)} - \eta 2^{-m} g^{(k)})$. Accept the
   first $T^+$ that satisfies the Armijo condition
   $\bar G_t(T^{+}) \le \bar G_t(T^{(k)}) + \sigma \langle g^{(k)}, T^{+} - T^{(k)}\rangle$, with $\sigma = 10^{-4}$.
   If none is accepted, keep $T^{(k)}$ (a rejected step). Each trial costs one forward-only window
   pass. $m_{\max} = 0$ is the plain fixed-step method.

### 4.3 Projected Adam (`projected_adam`, PyTorch)

This variant applies `torch.optim.Adam` to $T$ itself, followed by $\Pi_\Delta$ under `torch.no_grad()`.
It is an adaptation of the reference optimiser, not its parameterisation. It needs two additions
that the logit arm does not:
* **Tangent gradient.** Adam receives $g - \operatorname{mean}_{\text{row}}(g)$. Every entry of
  $\nabla \bar G_t$ is negative (raising any single probability lowers $G_t$; the row constraint is
  what balances it), so Adam's per-coordinate normalisation would turn the raw gradient into an
  almost uniform step, which the projection then cancels. Removing the row constant, the
  multiplier of $\sum_j T_{ij} = 1$, leaves $\Pi_\Delta(T - \eta g)$ unchanged, but it gives Adam the
  informative part.
* **Feasibility halving.** If a row's projected proposal sets an entry to exactly zero, $G_t$ is
  $+\infty$ (Section 4.4). Only that row's Adam step is then halved, at most 20 times.

### 4.4 Zeros, the pseudocount and conditioning

Because $\kappa > 0$ multiplies $\log T$ in every entry, $G_t(T) = +\infty$ whenever any
$T_{a,ij} = 0$. The projection may *propose* exact zeros, but this objective excludes them.
Exact zeros only become attainable minimisers under objectives with no log-barrier on that
entry, such as a squared latent energy or a window NLL without a pseudocount. The barrier also
makes Euclidean steps badly conditioned. An entry's curvature is
$(\lambda S + \kappa)_{a,ij} / (M_t T_{a,ij}^2)$, which explodes as the entry approaches zero, and a
single global step size is limited by the stiffest entry of the whole tensor. EM avoids this
because its update is closed-form and scale-free; the logit arm avoids it because softmax
steps are multiplicative in $T$.

### 4.5 Extension points

Every arm takes an objective value and gradient that are agnostic to the objective. For
`gradcscg`, a future differentiable term $R(T)$ is added to the torch objective and autodiff
supplies its gradient. Candidate terms include an EMA-teacher latent prediction energy (KL or
squared Euclidean), transition sparsity, clone uniqueness and occupancy terms. For the NumPy
arms the analytic value and gradient are added to `projected_gradient_step`. The window
likelihood is reused unchanged, and it is simply dropped for objectives without observation NLL.

## 5. Validation

`tests/src/mercury_runs/test_cscg_gradient.py` checks:

* **NumPy analytic machinery**
  * the window NLL against brute-force path enumeration;
  * the identity $T \odot \nabla = -X$, and that $X$ equals the existing `smooth_window` counts;
  * central finite differences of the window NLL and of $\bar G_t$ along random feasible simplex
    directions (row sums zero), to relative error $10^{-6}$;
  * one-sided differences at points with exact zeros, where the gradient is finite and nonzero
    although $X = 0$.
* **Projection and PGD**
  * exactness, idempotence and the KKT conditions of the projection;
  * Armijo monotone descent in training;
  * detection of the plain fixed step's infinite objective (`NumericalInstabilityError`).
* **PyTorch against NumPy**
  * the reference batched log-space NLL over a padded batch of unequal lengths equals the NumPy
    window NLL per sequence ($10^{-12}$);
  * `GradCSCG` objective value, gradient-derived $X$ (against `smooth_window`, $10^{-12}$) and the
    autodiff logit gradient (against the analytic $T\odot(g - \langle T, g\rangle_{\text{row}})$, $10^{-9}$);
  * $\operatorname{softmax}(W_0) = T_0$;
  * one Adam step equal to the Keras-default Adam formula;
  * for projected Adam: autodiff equal to the analytic gradient, the projection equal to NumPy,
    and the halving logic.
* **EM unchanged:** the explicit EM recursion is reproduced bit-for-bit, and the existing
  distillation golden-fixture tests pass.

## 6. Remaining differences between the arms

* **Same:** stream, clone allocation, initial tensor, window and context, cached boundary
  beliefs, initial-state prior (uniform over clones), $S$ recursion, $\kappa$, $\lambda$, commit rule,
  number of local iterations per arrival, decoder and evaluation.
* **Different by construction:** the optimiser, and for `gradcscg` the parameterisation of the
  same $T$. After the first arrival the tensors differ, so the realised beliefs $f_s$, the counts
  $S_t$ and hence the realised $G_t$ differ too; the *functional form* of $G_t$ is identical.
* **Cost:** the costs per iteration are not equivalent, so the report gives wall-clock time and
  pass counts separately.
  * EM: one NumPy forward–backward pass plus a row normalisation.
  * `gradcscg`: one dense log-space torch forward pass over $N\times N$ with autodiff, plus one
    Adam step.
  * `projected_gradient`: additionally $1+m$ forward-only trial passes.
