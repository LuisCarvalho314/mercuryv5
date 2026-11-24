# Online Predictive JEPA–CSCG: Mathematical Formulation

This document formulates the online predictive JEPA–CSCG learner: a clone-structured
hidden Markov model whose transition tensor is learned online. It interpolates between an
online EM objective and a latent-space predictive term. In that term the student must
predict an exponential-moving-average teacher's smoothed posterior over the next latent
state. A coefficient $\rho \in [0,1]$ (`jepa_mix`) selects pure EM ($\rho = 0$), a hybrid,
or pure predictive self-supervision ($\rho = 1$) (Sections 4.6 and 5.8–5.11). The earlier
penalty parameterisation is retained as a legacy configuration (Sections 4.5 and 5.10).

Scope: the *predictive* JEPA objective, with smoothed student source beliefs. This is the
configuration evaluated in the reported runs. Other auxiliary objectives and baselines are
mentioned only where they clarify the relationship to standard CSCG learning. Section 9
maps symbols to code, and Section 10 collects the remarks and flagged issues.

---

## 1. Setting and notation

### 1.1 Data stream

The learner receives a single stream of observations and actions,

$$
x_0, a_0, x_1, a_1, \dots, a_{n-2}, x_{n-1},
\qquad x_t \in \{1,\dots,O\},\quad a_t \in \{1,\dots,A\}.
$$

Actions are *source-aligned*: $a_t$ is the action that moves the agent from $x_t$ to
$x_{t+1}$. The stream may be partitioned into episodes. Let $\mathcal E \ni 0$ be the set of
episode-start times, and let

$$
e(t) = \max\{\, e \in \mathcal E : e \le t \,\}
$$

be the start of the episode containing $t$. A *transition* $x_{t-1} \to x_t$ exists exactly
when $t \notin \mathcal E$. Slices are written $x_{s:t} = (x_s,\dots,x_t)$, inclusive.

### 1.2 The clone-structured HMM

Each observation symbol $o$ owns a disjoint set of latent *clones* $C(o)$ with
$n_o = \lvert C(o) \rvert$ elements. The latent state space has $N = \sum_o n_o$ states.
A clone emits its own symbol deterministically:

$$
p(x_t = o \mid z_t = i) = \mathbb 1[\, i \in C(o) \,].
$$

We write $e_o \in \{0,1\}^N$ for the indicator vector of $C(o)$ and use the shorthand
$e_t := e_{x_t}$.

Dynamics are given by an action-conditioned, row-stochastic transition tensor,

$$
\begin{aligned}
T_{a,ij} &= p(z_{t+1} = j \mid z_t = i,\ a_t = a), \\
\mathcal T &= \Big\{\, T \in \mathbb R_{\ge 0}^{A\times N\times N} :
\textstyle\sum_j T_{a,ij} = 1 \ \ \forall a,i \,\Big\}.
\end{aligned}
$$

$T_a \in \mathbb R^{N\times N}$ denotes the matrix for action $a$. The initial-state
distribution is uniform, $\pi_i = 1/N$, and the model restarts from $\pi$ at every
$e \in \mathcal E$.

### 1.3 Operators

Belief vectors are column vectors in $\mathbb R^N_{\ge 0}$.

$$
\begin{aligned}
\mathcal N(v) &= v / (\mathbf 1^\top v)
  &&\text{normalisation,} \\
(u \odot v)_i &= u_i v_i
  &&\text{elementwise product,} \\
\operatorname{rowNorm}(M)_{a,ij} &= M_{a,ij} \Big/ \textstyle\sum_{k} M_{a,ik}
  &&\text{row normalisation,} \\
\langle M, L \rangle &= \textstyle\sum_{a,i,j} M_{a,ij} L_{a,ij}
  &&\text{tensor inner product.}
\end{aligned}
$$

$\log T$ is taken elementwise, and $\mathbf 1$ also denotes the all-ones tensor.

---

## 2. Inference primitives

The learner uses five inference operations. Each is defined once here for a generic
tensor $M \in \mathcal T$ and reused throughout. Let $f$ be a belief over $z_{t-1}$,
$a$ the action $a_{t-1}$, and $o$ the observation $x_t$.

**Definition 1 (one-step prediction and evidence).**

$$
\hat p_M(f,a) = M_a^\top f,
\qquad
Z_M(f,a,o) = e_o^\top M_a^\top f .
$$

$\hat p_M$ is the predictive distribution over $z_t$. $Z_M$ is the predictive probability
of observing $o$.

**Definition 2 (forward step).**

$$
\Phi_M(f,a,o) = \mathcal N\big(\, e_o \odot M_a^\top f \,\big).
$$

This is Bayes' rule for one step: predict, then condition on $o$.

**Definition 3 (two-slice posterior).**

$$
\Xi_M(f,a,o)_{a',ij}
= \mathbb 1[a'=a]\,
\frac{f(i)\, M_{a,ij}\, e_o(j)}{Z_M(f,a,o)} .
$$

This is the joint posterior of $(z_{t-1}, z_t)$ given the prior $f$ on $z_{t-1}$ and the
observation $o$ at $t$. It has unit total mass, $\langle \Xi_M, \mathbf 1\rangle = 1$, and
its two marginals are

$$
\begin{aligned}
\textstyle\sum_{i} \Xi_M(f,a,o)_{a,ij} &= \Phi_M(f,a,o)_j, \\
\textstyle\sum_{j} \Xi_M(f,a,o)_{a,ij} &= \Psi_M(f,a,o)_i ,
\end{aligned}
$$

where $\Psi_M$ is defined next.

**Definition 4 (one-step retrospective smoothing).**

$$
\Psi_M(f,a,o) = \mathcal N\big(\, f \odot M_a e_o \,\big).
$$

$M_a e_o$ is the backward message of the single observation $o$. $\Psi_M$ updates the
belief about $z_{t-1}$ after also seeing $x_t$. It looks exactly one step into the future
of $z_{t-1}$, and no further.

**Definition 5 (window smoothing from a boundary belief).** Given a boundary belief $h$ for
$z_s$, an observation window $x_{s:t}$, and actions $a_{s:t-1}$, define the forward and
backward messages

$$
\begin{aligned}
\alpha_s &= \mathcal N(h \odot e_s), &
\alpha_{u+1} &= \Phi_M(\alpha_u, a_u, x_{u+1}), \\
\beta_t &= e_t, &
\beta_u &= \mathcal N\big( M_{a_u} \beta_{u+1} \odot e_u \big),
\end{aligned}
$$

and the smoothed marginals

$$
q_u = \mathcal N(\alpha_u \odot \beta_u), \qquad u = s,\dots,t.
$$

We write $(q_s,\dots,q_t) = \mathcal S_M(h;\, x_{s:t}, a_{s:t-1})$. By the standard
forward–backward identity, $q_u$ is the exact posterior $p(z_u \mid x_{s:t})$ of the HMM
that starts at time $s$ from the prior $h$ (instead of $\pi$) and evolves under $M$. The
normalisation of $\beta_u$ only rescales it and does not change $q_u$. Two special cases
connect the definitions:

$$
\begin{aligned}
q_t &= \alpha_t, \\
\mathcal S_M(h;\, x_{t-1:t}, a_{t-1})_{t-1} &= \Psi_M(h, a_{t-1}, x_t)
\quad \text{if } h \text{ is supported on } C(x_{t-1}).
\end{aligned}
$$

That is, smoothing the last node of a window is filtering, and smoothing a two-node window
is one-step retrospective smoothing.

---

## 3. From batch CSCG learning to online EM

### 3.1 Batch CSCG (reference point)

Standard CSCG learning maximises the likelihood of a fixed sequence by EM:

$$
\begin{aligned}
P_T(x_{0:n-1} \mid a_{0:n-2})
&= \sum_{z_{0:n-1}} \pi(z_0)\, e_0(z_0)
   \prod_{t=1}^{n-1} T_{a_{t-1}, z_{t-1} z_t}\, e_t(z_t).
\end{aligned}
$$

The E-step runs full forward–backward over the whole sequence. This yields the two-slice
posteriors $p(z_{t-1}, z_t \mid x_{0:n-1})$ for all $t$, so every transition's posterior
uses *all* past and future observations. The M-step re-estimates $T$ from these expected
counts, with a small pseudocount $\kappa > 0$:

$$
T \leftarrow \operatorname{rowNorm}\Big(
\textstyle\sum_t p(z_{t-1}, z_t \mid x_{0:n-1}) + \kappa \mathbf 1 \Big).
$$

Each posterior is placed at its action $a_{t-1}$. This learner is batch and non-causal;
the learner below is neither.

### 3.2 Online EM with a one-transition E-step

The online learner sees $x_t$ at *arrival* $t$. It keeps three quantities:

* a student tensor $T_t \in \mathcal T$, the model in force when $x_t$ arrives;
* sufficient statistics $S_t \in \mathbb R^{A\times N\times N}_{\ge 0}$, the expected
  transition counts accumulated over arrivals $< t$;
* the causal student filter $f_t$, computed once at arrival $t$ and never revised:

$$
f_t =
\begin{cases}
\mathcal N(\pi \odot e_t), & t \in \mathcal E, \\
\Phi_{T_t}(f_{t-1}, a_{t-1}, x_t), & t \notin \mathcal E.
\end{cases}
$$

Because each step uses the tensor in force at that time, $f_t$ is a *carried* filter. It
equals the exact filter $p(z_t \mid x_{e(t):t})$ under a single model only if $T$ did not
change over $[e(t), t]$.

The E-step at arrival $t$ uses only the newest transition, through its evidence and
two-slice posterior:

$$
\begin{aligned}
Z_t(T) &= Z_T(f_{t-1}, a_{t-1}, x_t), \\
\xi_t(T) &= \Xi_T(f_{t-1}, a_{t-1}, x_t).
\end{aligned}
$$

With forgetting factor $\lambda \in (0,1]$, define the increment weight

$$
\gamma_\lambda =
\begin{cases}
1, & \lambda = 1 \quad \text{(cumulative counts)}, \\
1-\lambda, & \lambda < 1 \quad \text{(stochastic approximation)},
\end{cases}
$$

and the per-arrival EM objective

$$
G^{\mathrm{EM}}_t(T)
= -\big\langle \lambda S_t + \kappa \mathbf 1,\ \log T \big\rangle
  \;-\; \gamma_\lambda \log Z_t(T).
$$

The first term is the expected complete-data log-likelihood of the past, summarised by
$S_t$ and held fixed, together with a Dirichlet prior of concentration $1+\kappa$ on every
row. The second is the log-evidence of the newest transition. Minimising $G^{\mathrm{EM}}_t$
by EM gives the iteration

$$
\begin{aligned}
\tilde S^{(k)} &= \lambda S_t + \gamma_\lambda\, \xi_t(T^{(k)}), \\
B^{(k)} &= \tilde S^{(k)} + \kappa \mathbf 1, \\
T^{(k+1)} &= \operatorname{rowNorm}\big(B^{(k)}\big),
\end{aligned}
$$

starting from $T^{(0)} = T_t$. After the last iteration, only the data counts are
committed: $S_{t+1} = \tilde S^{(K-1)}$.

**Relation to batch EM.** With $\lambda = 1$, $S_t$ is a sum of one two-slice posterior per
past transition. Each was computed once, from the *filtered* source $f_{u-1}$ and the
tensor current at its arrival, and never revised. Batch EM instead recomputes every
posterior under the current $T$, using the whole sequence. The online learner therefore
gives up two things: retrospective revision of past posteriors, and access to future
context. The predictive JEPA term reintroduces a bounded amount of retrospective context,
without reading beyond $x_t$.

This learner, with $K$ local iterations, is the "short EM" baseline.

---

## 4. The predictive JEPA objective

The JEPA reading is as follows.

* **Context encoder:** the student's beliefs about the source state.
* **Target encoder:** a slowly moving teacher's smoothed posterior over the destination
  state.
* **Predictor:** the student's own transition tensor.

The student is trained to predict, in latent space, the teacher's representation of the
*next* state. It is not trained to reconstruct observations. Three distinct kinds of
inference enter, and they are kept separate below:

| Quantity | Model | Conditioning | Role |
|---|---|---|---|
| $f_t$, $h_t$ | student, teacher | $x_{e(t):t}$, causal | carried filters |
| $b_u$ | student $T_{u+1}$ | $x_{e(u):u+1}$, lag 1 | context (source) belief |
| $q^{t}_{u+1}$ | teacher $\bar T_t$ | $x_{s_t:t}$, lag $\le W-2$ | distillation target |

### 4.1 Teacher

The teacher tensor is an exponential moving average of past student tensors:

$$
\begin{aligned}
\bar T_0 &= T_0, \\
\bar T_{t+1} &= \tau\, \bar T_t + (1-\tau)\, T_{t+1},
\qquad \tau = 2^{-1/H}.
\end{aligned}
$$

Here $H$ is the half-life, counted in transitions. A convex combination of row-stochastic
tensors is row-stochastic, so $\bar T_t \in \mathcal T$. The teacher has no objective of
its own. It is a temporally smoothed copy of the student. This gives a target that changes
more slowly than the student it supervises, which is the role of the target encoder in
JEPA and BYOL-style methods. The teacher carries its own causal filter:

$$
h_t =
\begin{cases}
\mathcal N(\pi \odot e_t), & t \in \mathcal E, \\
\Phi_{\bar T_t}(h_{t-1}, a_{t-1}, x_t), & t \notin \mathcal E.
\end{cases}
$$

Like $f_t$, $h_t$ is cached and never revised.

### 4.2 Context window and teacher targets

Let $W \ge 2$ be the context window. At arrival $t \notin \mathcal E$, define

$$
\begin{aligned}
s_t &= \max\big(e(t),\ t - W + 1\big), \\
\mathcal U_t &= \{ s_t, \dots, t-1 \}, \\
n_t &= t - s_t \ \ge 1 .
\end{aligned}
$$

The window covers the observations $x_{s_t:t}$, the transitions $u \to u+1$ for
$u \in \mathcal U_t$, and never crosses an episode boundary. The teacher smooths the window
under its current tensor, starting from its cached boundary belief:

$$
(q^t_{s_t}, \dots, q^t_t)
= \mathcal S_{\bar T_t}\big(h_{s_t};\ x_{s_t:t},\ a_{s_t:t-1}\big).
$$

The target for transition $u \to u+1$ is the destination marginal $q^t_{u+1}$. It is
supported on $C(x_{u+1})$ and conditions on $t-u-1 \le W-2$ observations after $x_{u+1}$,
all inside the window.

Two properties matter.

1. **Retrospective revision.** The targets are recomputed at every arrival. The target for
   a fixed transition $u \to u+1$ is re-smoothed with more future context as $t$
   increases, until $u$ leaves the window.
2. **Frozen within an arrival.** $q^t$ is computed once, before any student update at
   arrival $t$. It is a constant throughout the local optimisation.

By Definition 5, $q^t_u$ is the exact posterior of the teacher HMM initialised at $s_t$
with prior $h_{s_t}$. Here $h_{s_t}$ summarises $x_{e(t):s_t}$ under the teacher tensors of
those earlier times.

*Optional teacher-only window.* Given $W_T \ge W$ and
$s'_t = \max(e(t), t - W_T + 1)$, the targets are
$\mathcal S_{\bar T_t}(h_{s'_t};\, x_{s'_t:t}, a_{s'_t:t-1})$ restricted to
$u = s_t+1,\dots,t$. With $W_T = W$ this is the definition above. See Remark R5.

### 4.3 Student context beliefs

The source belief for transition $u \to u+1$ is the student's one-step retrospective
smoothing of $z_u$. It is computed once, at arrival $u+1$, under the tensor then in force:

$$
b_u = \Psi_{T_{u+1}}(f_u,\, a_u,\, x_{u+1})
\;\propto\; f_u \odot T_{u+1,a_u}\, e_{u+1}.
$$

So $b_u$ is the student's estimate of $p(z_u \mid x_{e(u):u+1})$, cached and never
revised. By Definition 3 it is the source marginal of $\xi_{u+1}(T_{u+1})$. The student's
context is therefore deliberately short: one step of lookahead. Only the teacher has the
full window. Distillation transfers the window's retrospective information from the
teacher into the student's one-step predictor.

*(Filtered-student variant, not reported here: $b_u = f_u$.)*

### 4.4 Predictive loss

For a candidate tensor $T$, the student's predicted mass on destination clone $j$ is

$$
p_u(j; T) = \hat p_T(b_u, a_u)_j = \sum_i b_u(i)\, T_{a_u,ij}.
$$

The predictive loss is the window average of an *unnormalised* KL divergence from teacher
target to student prediction, over the destination clones:

$$
\mathcal L^{t}(T)
= \frac{1}{n_t} \sum_{u \in \mathcal U_t}\
  \sum_{j \in C(x_{u+1})}
  q^t_{u+1}(j)\, \log \frac{q^t_{u+1}(j)}{p_u(j;T)} .
$$

Only $T$ is optimised. The quantities $b_u$, $a_u$ and $q^t_{u+1}$ are constants.

**Proposition 1 (decomposition).** Let

$$
\begin{aligned}
Z_u(T) &= e_{u+1}^\top\, \hat p_T(b_u, a_u), \\
\tilde p_u(T) &= \mathcal N\big( e_{u+1} \odot \hat p_T(b_u, a_u) \big).
\end{aligned}
$$

$Z_u(T)$ is the probability of $x_{u+1}$ under source $b_u$ and $T$; $\tilde p_u(T)$ is the
student's normalised posterior over $z_{u+1}$. Then

$$
\mathcal L^t(T)
= \underbrace{\frac{1}{n_t}\sum_{u \in \mathcal U_t}
   \mathrm{KL}\big( q^t_{u+1} \,\big\|\, \tilde p_u(T) \big)}_{\text{latent matching}}
\;+\;
\underbrace{\frac{1}{n_t}\sum_{u \in \mathcal U_t} -\log Z_u(T)}_{\text{replayed evidence}} .
$$

*Proof.* On $C(x_{u+1})$, $p_u(j;T) = Z_u(T)\, \tilde p_u(T)_j$. Since
$\sum_{j \in C(x_{u+1})} q^t_{u+1}(j) = 1$,

$$
\sum_j q^t_{u+1}(j) \log \frac{q^t_{u+1}(j)}{p_u(j;T)}
= \mathrm{KL}\big(q^t_{u+1} \,\|\, \tilde p_u(T)\big) - \log Z_u(T).
$$

$\square$

Both terms are non-negative, because $Z_u \le 1$. The predictive objective therefore asks
the student to do two things: place the right amount of mass on the clones of the observed
successor, and distribute that mass among the clones as the teacher's smoothed posterior
does. The first term is what the teacher contributes beyond likelihood. See Remark R2 on
reading the second term.

### 4.5 Penalised per-arrival objective (legacy)

The legacy predictive JEPA learner (`jepa_mix` unset) minimises, at each arrival
$t \notin \mathcal E$,

$$
G_t(T) = G^{\mathrm{EM}}_t(T) + w_t\, \mathcal L^t(T),
\qquad T \in \mathcal T.
$$

The penalty weight scales with the accumulated data mass:

$$
w_t = \eta\, \big\langle \tilde S^{(k)}, \mathbf 1 \big\rangle
    = \eta\, \big( \lambda \langle S_t, \mathbf 1\rangle + \gamma_\lambda \big).
$$

The second equality holds because $\langle \xi_t(T), \mathbf 1 \rangle = 1$ for every $T$,
so $w_t$ does not depend on the local iteration $k$. With $\lambda = 1$,
$\langle S_t, \mathbf 1 \rangle$ is the number of transitions processed before $t$. Hence
$w_t = \eta\, m_t$, where $m_t$ is the number of transitions up to and including $t$.

*Motivation.* The data term carries total mass $\langle \tilde S, \mathbf 1\rangle$, which
grows without bound when $\lambda = 1$. A fixed penalty weight would become negligible over
time. Scaling by $\eta \langle \tilde S, \mathbf 1\rangle$ keeps the teacher's influence a
constant fraction of the data's influence. Section 5.2 makes this exact.

### 4.6 Explicit mixing objective

The penalty of Section 4.5 never removes the EM term, so pure predictive self-supervision
is not reachable from it, not even as $\eta \to \infty$. The learner therefore exposes the
interpolation between the two objectives directly, through a mixing coefficient
$\rho \in [0, 1]$ (`jepa_mix`).

The EM counts at local iteration $k$ have total mass

$$
\beta_t
= \big\langle B^{(k)}, \mathbf 1 \big\rangle
= \lambda \langle S_t, \mathbf 1\rangle + \gamma_\lambda + \kappa A N^2 .
$$

It does not depend on $k$, because $\langle \xi_t(T), \mathbf 1\rangle = 1$. Note that it
includes the pseudocounts. The normalised per-iteration EM objective is

$$
\widehat{\mathcal L}^{(k)}_{\mathrm{EM}}(T)
= -\frac{1}{\beta_t}\,\big\langle B^{(k)}, \log T \big\rangle ,
$$

and the per-iteration mixed objective is

$$
\mathcal L^{(k)}_{\rho}(T)
= (1-\rho)\, \widehat{\mathcal L}^{(k)}_{\mathrm{EM}}(T) + \rho\, \mathcal L^t(T).
$$

$\mathcal L^t$ is the predictive loss of Section 4.4, unchanged. In particular it keeps its
implicit replayed-evidence component (Proposition 1). It is not replaced by the
normalised latent-matching KL. $\mathcal L^{(k)}_\rho$ is fixed *within* iteration $k$,
because $B^{(k)}$ is frozen there. It is a different function at iteration $k+1$, because
$B^{(k+1)}$ uses $\xi_t(T^{(k+1)})$.

A single function of $T$ that is fixed for the whole arrival is obtained by replacing the
linearised data term with the marginal one:

$$
F_{\rho,t}(T) = \frac{1-\rho}{\beta_t}\, G^{\mathrm{EM}}_t(T) + \rho\, \mathcal L^t(T).
$$

Section 5.8 shows that the local iterations descend both: each $\mathcal L^{(k)}_\rho$ over
its own step, and $F_{\rho,t}$ over the whole arrival.

Three regimes follow.

| `jepa_mix` | Regime | Objective at every local iteration |
|---|---|---|
| $\rho = 0$ | pure EM | $\widehat{\mathcal L}^{(k)}_{\mathrm{EM}}$: short-context online EM |
| $0 < \rho < 1$ | hybrid | $\mathcal L^{(k)}_\rho$ |
| $\rho = 1$ | pure JEPA | $\mathcal L^t$ only |

The coefficient is a fraction of a unit-mass objective. It is not the legacy penalty scale
$\eta$, which multiplies the running count mass; Section 5.10 gives the exact relation.

---

## 5. Optimisation

$G_t$ has no closed-form minimiser. Both of its terms are negative logarithms of mixtures
in $T$. Each is bounded above by a linear function of $\log T$ that is tight at the current
iterate. This yields a majorise–minimise (MM) algorithm with closed-form steps.

### 5.1 Majorisers

Fix the current iterate $T^{(k)}$.

**Lemma 1 (EM bound for the data term).** For all $T \in \mathcal T$,

$$
-\log Z_t(T) \le -\big\langle \xi_t(T^{(k)}),\ \log T \big\rangle + c_1^{(k)},
$$

with equality at $T = T^{(k)}$. This is the standard EM inequality: Jensen's inequality
applied to $\log \sum_{ij} f_{t-1}(i)\, T_{a_{t-1},ij}\, e_t(j)$, with the weights
$\xi_t(T^{(k)})$.

**Lemma 2 (Jensen bound for the predictive term).** Define the responsibilities

$$
r^{u,(k)}_{ij} = \frac{b_u(i)\, T^{(k)}_{a_u,ij}}{p_u(j;T^{(k)})},
\qquad \sum_i r^{u,(k)}_{ij} = 1 .
$$

Then, for all $T \in \mathcal T$,

$$
-\log p_u(j;T) \le -\sum_i r^{u,(k)}_{ij} \log T_{a_u,ij} + c^{u,(k)}_{j},
$$

with equality at $T = T^{(k)}$. Summing with the weights $q^t_{u+1}(j)/n_t$ gives

$$
\mathcal L^t(T) \le -\big\langle D^{(k)},\ \log T \big\rangle + c_2^{(k)},
$$

where

$$
\begin{aligned}
D^{(k)}_{a,ij}
&= \frac{1}{n_t} \sum_{\substack{u \in \mathcal U_t \\ a_u = a}}
   q^t_{u+1}(j)\, r^{u,(k)}_{ij} \\
&= \frac{1}{n_t} \sum_{\substack{u \in \mathcal U_t \\ a_u = a}}
   \frac{b_u(i)\, T^{(k)}_{a,ij}\, q^t_{u+1}(j)}{p_u(j;T^{(k)})} .
\end{aligned}
$$

The sum runs over $j \in C(x_{u+1})$ implicitly, because $q^t_{u+1}$ is supported there.

### 5.2 Interpretation of the auxiliary counts

$D^{(k)}$ is a tensor of expected transition counts. It allocates each teacher target
$q^t_{u+1}(j)$ back to the source clones, in proportion to the student's current
responsibility for $j$. Its total mass is

$$
\langle D^{(k)}, \mathbf 1 \rangle
= \frac{1}{n_t} \sum_{u} \sum_j q^t_{u+1}(j) \sum_i r^{u,(k)}_{ij} = 1,
$$

so $w_t D^{(k)}$ carries mass $w_t = \eta \langle \tilde S^{(k)}, \mathbf 1\rangle$. For
$\eta = 1$, the teacher-derived pseudo-counts in each update weigh as much as all
accumulated data counts together.

### 5.3 Closed-form minimisation

**Lemma 3.** For $C \in \mathbb R^{A\times N\times N}_{>0}$,

$$
\operatorname*{arg\,min}_{T \in \mathcal T}\ -\langle C, \log T\rangle
= \operatorname{rowNorm}(C),
$$

and the minimiser is unique. Each row is an independent problem: maximise
$\sum_j C_{a,ij} \log T_{a,ij}$ subject to $\sum_j T_{a,ij} = 1$, solved by Lagrange
multipliers.

### 5.4 Coupled update

Combining Lemmas 1–3 gives a surrogate,

$$
Q^{(k)}(T)
= -\big\langle B^{(k)} + w_t D^{(k)},\ \log T \big\rangle + c^{(k)}.
$$

It satisfies $Q^{(k)} \ge G_t$ on $\mathcal T$, with equality at $T^{(k)}$. Its minimiser is
the update

$$
\boxed{\;
T^{(k+1)} = \operatorname{rowNorm}\big( B^{(k)} + w_t D^{(k)} \big)
\;}
\qquad k = 0, \dots, K-1,
$$

where $B^{(k)} = \lambda S_t + \gamma_\lambda \xi_t(T^{(k)}) + \kappa\mathbf 1$ as in
Section 3.2. Because $\kappa > 0$, every entry of $B^{(k)}$ is positive. Hence $T^{(k+1)}$
has full support, all logarithms stay finite, and every $p_u(j;\cdot)$ remains positive.

**Proposition 2 (monotone descent).** Under the coupled schedule,
$G_t(T^{(k+1)}) \le G_t(T^{(k)})$ for $k = 0,\dots,K-1$.

*Proof.*

$$
G_t(T^{(k+1)}) \le Q^{(k)}(T^{(k+1)}) \le Q^{(k)}(T^{(k)}) = G_t(T^{(k)}).
$$

$\square$

Each iteration recomputes both $\xi_t$ and the responsibilities under the latest iterate.
The sources $b_u$, the targets $q^t$, the weight $w_t$ and the statistics $S_t$ stay fixed.
This is what makes $G_t$ a single, well-defined objective during the arrival.

### 5.5 Historical (two-update) schedule

In the historical schedule, only the last iteration includes the predictive term:

$$
T^{(k+1)} =
\begin{cases}
\operatorname{rowNorm}\big( B^{(k)} \big), & k < K-1, \\
\operatorname{rowNorm}\big( B^{(k)} + w_t D^{(k)} \big), & k = K-1.
\end{cases}
$$

The first $K-1$ steps are MM steps on $G^{\mathrm{EM}}_t$ only. The last step is an MM step
on $G_t$. See Remark R1.

### 5.6 Commit and teacher update

After the $K$ local iterations, at arrival $t \notin \mathcal E$:

$$
\begin{aligned}
T_{t+1} &= T^{(K)}, \\
S_{t+1} &= \tilde S^{(K-1)} = \lambda S_t + \gamma_\lambda\, \xi_t(T^{(K-1)}), \\
\bar T_{t+1} &= \tau\, \bar T_t + (1-\tau)\, T_{t+1}.
\end{aligned}
$$

Only data counts are committed. The auxiliary counts $w_t D^{(k)}$ shape $T_{t+1}$ but
never enter $S$. As a result, teacher supervision does not accumulate as pseudo-data
across arrivals; it is re-derived from the current window every time.

At an episode start $t \in \mathcal E$ there is no transition. Both filters reset, and
$T$, $\bar T$ and $S$ are unchanged.

### 5.7 Special case: no teacher

With $\eta = 0$, $w_t = 0$, and $G_t = G^{\mathrm{EM}}_t$. Both schedules then reduce to the
online EM of Section 3.2 ("short EM") with the same $K$. The predictive JEPA learner
differs from short EM *only* by the term $w_t \mathcal L^t$. The same holds for
`jepa_mix` $= 0$, which in addition skips the teacher entirely (Section 5.9).

### 5.8 Mixed update

Fix the arrival $t$ and the current iterate $T^{(k)}$. Let $D^{(k)}$ be the teacher-derived
expected counts of Lemma 2,

$$
D^{(k)}_{a,ij}
= \frac{1}{n_t} \sum_{\substack{u \in \mathcal U_t \\ a_u = a}}
  r^{u,(k)}_{ij}\, q^t_{u+1}(j),
\qquad
r^{u,(k)}_{ij} = \frac{b_u(i)\, T^{(k)}_{a_u,ij}}{p_u(j;T^{(k)})} .
$$

As before, $j$ ranges over $C(x_{u+1})$ only, which is the destination-clone support
restriction, and $\langle D^{(k)}, \mathbf 1\rangle = 1$. Define the combined update counts

$$
M^{(k)} = (1-\rho)\, \frac{B^{(k)}}{\beta_t} + \rho\, D^{(k)},
\qquad \langle M^{(k)}, \mathbf 1 \rangle = 1,
$$

and the update

$$
\boxed{\;
T^{(k+1)}_{a,i,:} =
\begin{cases}
\operatorname{rowNorm}\big(M^{(k)}\big)_{a,i,:},
  & \sum_j M^{(k)}_{a,ij} > 0, \\[2pt]
T^{(k)}_{a,i,:}, & \sum_j M^{(k)}_{a,ij} = 0,
\end{cases}
\;}
\qquad k = 0,\dots,K-1 .
$$

The second case can occur only at $\rho = 1$; see Section 5.11.

**Lemma 3′ (zero counts).** Let $C \ge 0$. A row with positive mass has the unique
minimiser $\operatorname{rowNorm}(C)_{a,i,:}$ of $-\sum_j C_{a,ij} \log T_{a,ij}$ over the
simplex, using $0 \log 0 = 0$. That minimiser is zero wherever $C_{a,ij} = 0$. A row with
zero mass makes the objective constant, so every distribution minimises it, including
$T^{(k)}_{a,i,:}$.

**Derivation.** Lemmas 1 and 2 give, for all $T \in \mathcal T$,

$$
\begin{aligned}
G^{\mathrm{EM}}_t(T) &\le -\big\langle B^{(k)}, \log T\big\rangle + c_1^{(k)}, \\
\mathcal L^t(T) &\le -\big\langle D^{(k)}, \log T\big\rangle + c_2^{(k)},
\end{aligned}
$$

both with equality at $T^{(k)}$. Multiplying by $(1-\rho)/\beta_t$ and $\rho$ and adding
gives the surrogate

$$
Q^{(k)}_\rho(T) = -\big\langle M^{(k)}, \log T \big\rangle + c^{(k)},
$$

with $Q^{(k)}_\rho \ge F_{\rho,t}$ and equality at $T^{(k)}$. By Lemma 3′ the update above
minimises $Q^{(k)}_\rho$. For $\rho = 1$, $M^{(k)} = D^{(k)}$ and the update is
$\operatorname{rowNorm}(D^{(k)})$. For $\rho = 0$, $M^{(k)} = B^{(k)}/\beta_t$, and
because $\operatorname{rowNorm}$ is scale invariant the update is exactly
$\operatorname{rowNorm}(B^{(k)})$, the online EM step of Section 3.2.

**Proposition 4 (MM descent within an arrival).** Assume $F_{\rho,t}(T^{(0)}) < \infty$.
Then, for $k = 0, \dots, K-1$:

1. *Per-iteration objective:*
   $\mathcal L^{(k)}_\rho(T^{(k+1)}) \le \mathcal L^{(k)}_\rho(T^{(k)})$.
2. *Fixed arrival objective:*
   $F_{\rho,t}(T^{(k+1)}) \le F_{\rho,t}(T^{(k)})$.

*Proof.* (2) is the usual MM chain:

$$
F_{\rho,t}(T^{(k+1)})
\le Q^{(k)}_\rho(T^{(k+1)})
\le Q^{(k)}_\rho(T^{(k)})
= F_{\rho,t}(T^{(k)}).
$$

For (1), majorise only the predictive term. The resulting surrogate of
$\mathcal L^{(k)}_\rho$ is again $Q^{(k)}_\rho$, up to a constant, with equality at
$T^{(k)}$. The same chain applies. $\square$

What is *not* claimed:

* The values $\mathcal L^{(k)}_\rho(T^{(k)})$, $k = 0, 1, \dots$, do not form a monotone
  sequence. Each one is measured on a different objective, because $B^{(k)}$ changes with
  $k$. Statement (1) is about one step on one fixed objective. Only statement (2) chains
  across iterations.
* No objective is monotone across arrivals. $S_t$, $\beta_t$, the window, $q^t$ and
  $\bar T_t$ all change with $t$ (Remark R1).

The implementation records $\mathcal L^{(k)}_\rho$ before and after every local iteration
at each checkpoint. Tests check statement (1) on recorded runs and statement (2) on a
synthetic arrival.

### 5.9 Endpoint semantics and lifecycle

Inference is the same in every regime. The student filter $f_t$ always runs. The smoothed
source $b_{t-1}$ is computed whenever the JEPA term is active. Only parameter optimisation
differs between regimes.

| | $\rho = 0$ | $0 < \rho < 1$ | $\rho = 1$ |
|---|---|---|---|
| student filter $f_t$ | yes | yes | yes |
| source belief $b_{t-1}$ (smooth mode) | not needed | yes | yes |
| teacher filter $h_t$, targets $q^t$ | skipped | yes | yes |
| EM counts $\xi_t$, $B^{(k)}$ | yes | yes | not computed |
| teacher counts $D^{(k)}$ | skipped | yes | yes |
| pseudocount $\kappa$ in the update | yes | yes, inside $B^{(k)}$ | no |
| commit $S_{t+1} = \tilde S^{(K-1)}$ | yes | yes | no $S$ exists |
| EMA teacher step | skipped | once per arrival | once per arrival |
| EM-only warm-up iterations | — | none | none |

Two kinds of count are kept apart:

* **Persistent statistics** $S_t$ hold data expected counts only. They are committed once
  per arrival, with forgetting.
* **Teacher-derived counts** $D^{(k)}$ are temporary. They exist inside one local iteration,
  shape $T^{(k+1)}$, and are never accumulated as pseudo-observations.

The order of operations at an arrival $t \notin \mathcal E$ is unchanged:

1. advance $f_t$ and, if the JEPA term is active, $h_t$;
2. cache $b_{t-1}$;
3. build $q^t$ over the window;
4. freeze $b_u$, $q^t$ and $\bar T_t$;
5. run $K$ local iterations with the selected objective;
6. commit $T_{t+1} = T^{(K)}$;
7. commit $S_{t+1}$ if $\rho < 1$;
8. take one EMA teacher step if $\rho > 0$.

Neither the sources nor the targets are recomputed during step 5, and the teacher never
moves between local iterations.

### 5.10 The legacy penalty as a time-varying mix

Because $\langle D^{(k)}, \mathbf 1\rangle = 1$, the legacy count tensor of Section 5.4 is
a positive multiple of a mixed one:

$$
B^{(k)} + w_t D^{(k)}
= (\beta_t + w_t)\,
  \Big[ (1-\rho_t)\, \frac{B^{(k)}}{\beta_t} + \rho_t\, D^{(k)} \Big],
\qquad
\rho_t = \frac{w_t}{\beta_t + w_t} .
$$

Row normalisation removes the scalar, so a legacy coupled iteration *is* the mixed update
with $\rho = \rho_t$. It also minimises the same surrogate, because
$G_t = (\beta_t + w_t) F_{\rho_t,t}$. Substituting
$w_t = \eta\, \tilde m_t$, with $\tilde m_t = \lambda\langle S_t,\mathbf 1\rangle + \gamma_\lambda$,
gives

$$
\rho_t
= \frac{\eta\, \tilde m_t}{(1+\eta)\, \tilde m_t + \kappa A N^2} .
$$

So $\rho_t \ne \eta/(1+\eta)$ whenever $\kappa > 0$. Two cases:

* $\lambda = 1$: $\tilde m_t$ is the transition count, and $\rho_t \uparrow \eta/(1+\eta)$
  only as the counts outgrow the pseudocount mass.
* $\lambda < 1$: $\tilde m_t \to 1$, and $\rho_t \to \eta/(1+\eta+\kappa A N^2)$, which
  stays different from $\eta/(1+\eta)$ permanently.

For the level-21 ground-truth allocation ($A = N = 4$, $\kappa A N^2 = 0.128$) with
$\eta = 1$ and $\lambda = 1$, $\rho_1 = 0.470$ and $\rho_{100} = 0.4997$. With 10 clones
per symbol ($N = 30$, $\kappa A N^2 = 7.2$), $\rho_1 = 0.109$.

No constant `jepa_mix` therefore reproduces a legacy run. The minimal reproduction
mechanism is to keep $\eta$ as the legacy parameter and convert it to $\rho_t$ at every
arrival (`legacy_jepa_mix`). The update itself is shared. Legacy schedules:

* *coupled* (`coupled_auxiliary=True`): the mix $\rho_t$ at all $K$ iterations;
* *historical* (default when `jepa_mix` is unset): the coefficient schedule
  $(0, \dots, 0, \rho_t)$, that is, $K-1$ EM-only warm-up iterations and then one mixed
  iteration (Section 5.5).

The historical schedule is kept only as this explicitly identified legacy configuration.
Setting `jepa_mix` always applies $\rho$ at every iteration, and is rejected in
combination with `penalty_weight` or `coupled_auxiliary`. The legacy `nll_control` arm
uses the same update, with its replay-evidence counts in place of $D^{(k)}$.

Floating point: forming $\operatorname{rowNorm}(M_{\rho_t})$ instead of
$\operatorname{rowNorm}(B + wD)$ changes results by rounding only. Against a golden fixture
generated by the pre-change code, every legacy path is bit-identical except the
penalised predictive and `nll_control` updates. Those agree to $\le 5.6\times10^{-16}$ over
60 synthetic arrivals and to $\le 7.8\times10^{-16}$ over 1,000 real level-21 arrivals, in
both legacy schedules, with identical work counters.

### 5.11 Unsupported rows, zero predictive support and empty windows

**Unsupported rows.** A row $(a,i)$ has zero optimisation mass when no window transition
with action $a$ gives source $i$ positive belief and no EM count reaches it. For $\rho < 1$
this cannot happen, because $B^{(k)} \ge \kappa > 0$. For $\rho = 1$ it is the normal case
for every row outside the window's source support. Such a row keeps
$T^{(k)}_{a,i,:}$ (Section 5.8). By Lemma 3′ this is a valid minimiser, so Proposition 4
is unaffected. It is a tie-breaking rule, not an objective change. Uniform, zero and NaN
rows are never produced.

**Zero predictive support.** If $q^t_{u+1}(j) > 0$ and $p_u(j;T) = 0$, then
$\mathcal L^t(T) = +\infty$ and $r^u_{ij} = 0/0$. The implementation never clips the
denominator or adds mass to repair this. It raises `ZeroPredictiveSupportError`, a
`ValueError`, with the arrival, the iteration, the number of affected clones and the
target mass. The student filter raises the same error when $Z_t(T_t) = 0$, the
total-support case in which the student gives probability zero to the observation itself.

*When can it occur?*

1. **$\rho < 1$ (and every legacy configuration): never**, given $T_0 > 0$. Every update
   adds $(1-\rho)\kappa/\beta_t > 0$ to every entry, so $T_t > 0$ for all $t$. The
   runner's $T_0 = \operatorname{rowNorm}(U + \kappa\mathbf 1)$ is strictly positive.
2. **Within an arrival at $\rho = 1$: never, if it does not hold at $T^{(0)}$.** Suppose
   $p_u(j;T^{(k)}) > 0$ for some $j$ with $q^t_{u+1}(j) > 0$. Some $i$ has
   $b_u(i)\, T^{(k)}_{a_u,ij} > 0$, hence $D^{(k)}_{a_u,ij} > 0$. That row then has
   positive mass, so $T^{(k+1)}_{a_u,ij} > 0$ and
   $p_u(j;T^{(k+1)}) \ge b_u(i)\, T^{(k+1)}_{a_u,ij} > 0$.
3. **Across arrivals at $\rho = 1$: yes.** With $T_0 > 0$, the teacher satisfies
   $\bar T_t \ge \tau^{t} T_0 > 0$. So $h_{s_t}$ and $q^t_{u+1}$ cover all of $C(x_{u+1})$.
   The transitions $u < t-1$ of the window were already in the previous window, so by (2)
   they keep their support. Only the newest transition $u = t-1$ can lack it. A row
   $(a,i)$ updated at an earlier arrival $t'$ has mass only on
   $\bigcup \{ C(x_{u+1}) : u \in \mathcal U_{t'},\ a_u = a,\ b_u(i) > 0 \}$. When the
   newest transition reaches a successor that is absent from that set, the student
   predicts probability zero for it.

Case 3 is a property of the objective, not of the optimiser. For any row used by the
window, moving mass from a destination that no window transition reaches onto one that a
window transition does reach never decreases any $p_u(j)$. If $q > 0$ there, it strictly
decreases $\mathcal L^t$. Hence *every* minimiser of $\mathcal L^t$ assigns zero mass to
successors absent from the window. More local iterations, or a different optimiser,
cannot prevent this.

*Empirical check (smoke, Section 10, R7).* On level 21, seed 0, pure JEPA fails at
arrival $2$. The stream begins $x_0 = x_1$ under action $a_0 = a_1$, and then
$x_2 \ne x_1$. Arrival 1 sends every row $(a_0, i \in C(x_0))$ to $C(x_0)$ only, so
$Z_2(T_2) = 0$.

*Numerical stabilisation versus objective modification.* The zero is exact, not an
underflow, so no purely numerical device such as log-space arithmetic or guarded division
can produce a finite objective. Each available remedy changes the objective, and must be
reported as such:

* **(a) JEPA Dirichlet prior.** Minimise
  $\mathcal L^t(T) - \kappa_J \langle \mathbf 1, \log T\rangle$, with update
  $\operatorname{rowNorm}(D^{(k)} + \kappa_J \mathbf 1)$. This is a MAP objective on the
  predictive loss alone. It keeps $T > 0$, and Proposition 4 holds for the modified
  objective. It is *not* the EM pseudocount: $\kappa_J$ is scaled against
  $\langle D, \mathbf 1\rangle = 1$, not against $\beta_t$, and no data counts enter.
* **(b) Teacher-centred prior.** Update $\operatorname{rowNorm}(D^{(k)} + \kappa_J \bar T_t)$.
  This keeps full support because $\bar T_t > 0$, and it anchors unseen successors to the
  slow model instead of to uniform.
* **(c) Treat pure JEPA as ill-posed online** and study $\rho \to 1^-$. A hybrid with
  $\rho$ close to $1$ already behaves like (a), with prior mass $(1-\rho)\kappa/\beta_t$
  per entry, plus a small data term.

Option (a) is the minimal proposal. None of them is implemented. The choice is a
research decision, to be made after the pure-JEPA arm's failure has been recorded.

**Empty windows.** At an episode start $t \in \mathcal E$ there is no transition. Both
filters reset to $\pi$. No targets are built, and $T$, $\bar T$ and $S$ are unchanged. For
$t \notin \mathcal E$, $n_t = t - s_t \ge 1$, so no division by zero can occur. A stream in
which every time is an episode start leaves $T_0$ unchanged in every regime (tested).

---

## 6. The online algorithm and its causality

**Initialisation.**

$$
\begin{aligned}
T_0 &= \operatorname{rowNorm}(U + \kappa \mathbf 1),
  \qquad U_{a,ij} \overset{\text{iid}}{\sim} \mathrm{Uniform}(0,1), \\
\bar T_0 &= T_0, \qquad S_0 = 0 .
\end{aligned}
$$

**Arrival $t$.** Steps 2–7 are skipped if $t \in \mathcal E$.

1. Filters: compute $f_t$ under $T_t$ and $h_t$ under $\bar T_t$, then cache both.
2. Context belief: $b_{t-1} = \Psi_{T_t}(f_{t-1}, a_{t-1}, x_t)$; cache it.
3. Window: $s_t$, $\mathcal U_t$, $n_t$.
4. Targets: $q^t = \mathcal S_{\bar T_t}(h_{s_t}; x_{s_t:t}, a_{s_t:t-1})$, frozen.
5. Coefficient: $\rho$ = `jepa_mix`. In the legacy configuration, instead
   $w_t = \eta(\lambda\langle S_t,\mathbf 1\rangle + \gamma_\lambda)$ and
   $\rho_t = w_t/(\beta_t + w_t)$ (5.10).
6. Local iterations from $T^{(0)} = T_t$. With `jepa_mix`, the mixed update (5.8) runs at
   every iteration. In the legacy configuration, the coupled (5.4) or historical (5.5)
   schedule runs.
7. Commit $T_{t+1}$, then $S_{t+1}$ if $\rho < 1$, then $\bar T_{t+1}$ if $\rho > 0$
   (5.6, 5.9).

With $\rho = 0$ the teacher filter, Step 4 and the teacher update are skipped. With
$\rho = 1$ no EM counts are formed and no $S$ exists. Step 2 still runs, because it is
inference.

**Proposition 3 (causality).** $T_{t+1}$, $\bar T_{t+1}$ and $S_{t+1}$ are functions of
$x_{0:t}$, $a_{0:t-1}$, $\mathcal E \cap [0,t]$ and $T_0$ only.

*Proof sketch.* By induction on $t$. Every quantity used at arrival $t$ is either cached
from earlier arrivals ($f_u$, $h_u$, $b_u$ for $u < t-1$) or computed from $x_{s_t:t}$,
$a_{s_t:t-1}$ and tensors that are themselves causal by the induction hypothesis. These
are $f_t$, $h_t$, $b_{t-1}$, $q^t$ and the local iterates. $\square$

The three kinds of inference sit in time as follows:

* **Causal filtering:** $f_t$ and $h_t$ use $x_{\le t}$.
* **Retrospective smoothing within available context:** $b_u$ (lag 1) and $q^t_{u+1}$
  (lag $t-u-1 \le W-2$) revise beliefs about past states using later observations, all at
  or before $x_t$.
* **Teacher–student distillation:** the student's one-step predictor $T_{a_u}^\top b_u$ is
  fitted to the teacher's window-smoothed posterior $q^t_{u+1}$.

---

## 7. Evaluation

Only the student is evaluated. A checkpoint tensor $T^\star = T_c$ is evaluated with no
parameter updates, on the canonical evaluation-v5 walk set. There are $100$ walks of
$L = 10{,}000$ steps, with random-action probability $0.1$ and walk seed $0$. Each walk
$(y_{0:L-1}, a_{0:L-2})$ is decoded independently.

**Training-matched decoder.** The decoder reuses the student's training inference: a
carried filter under $T^\star$, plus one step of retrospective smoothing.

$$
\begin{aligned}
g_0 &= \mathcal N(\pi \odot e_{y_0}), \\
g_\tau &= \Phi_{T^\star}(g_{\tau-1}, a_{\tau-1}, y_\tau),
  \qquad \tau = 1,\dots,L-1, \\
\hat z_\tau &=
\begin{cases}
\arg\max_i\ \Psi_{T^\star}(g_\tau, a_\tau, y_{\tau+1})_i, & \tau < L-1, \\
\arg\max_i\ g_{L-1}(i), & \tau = L-1.
\end{cases}
\end{aligned}
$$

Ties go to the smallest index. The decoded state is the argmax of the lag-1 marginal
$p_{T^\star}(z_\tau \mid y_{0:\tau+1})$, the same quantity as the training source belief
$b_u$. The same routine with a lag of $W-1$ is used for the windowed EM baseline.

**Held-out NLL** (bits per step), averaged over all steps of all walks:

$$
\mathrm{NLL}
= \frac{1}{100\,L} \sum_{\text{walks}} \sum_{\tau=0}^{L-1} \ell_\tau,
\qquad
\ell_\tau =
\begin{cases}
-\log_2 \big( n_{y_0} / N \big), & \tau = 0, \\
-\log_2 Z_{T^\star}(g_{\tau-1}, a_{\tau-1}, y_\tau), & \tau \ge 1.
\end{cases}
$$

**Structural metrics.** The decoded sequences $\hat z$ are scored with the canonical
evaluation-v5 metrics, as defined in the evaluation-v5 protocol:

* occupancy-weighted transition similarity $1-\mathrm{TV}$, with and without self-loops;
* action-conditioned edge F1, as the area under the threshold curve;
* latent purity and concentration (canonical recall), from a many-to-one alignment with an
  explicit unmatched sink.

---

## 8. Hyperparameters (reported runs)

| Quantity | Symbol | Value |
|---|---|---|
| Context window | $W$ | $40$ (also $100$) |
| Teacher window | $W_T$ | equal to $W$ |
| Teacher half-life | $H$ | $40$ transitions, $\tau = 2^{-1/40}$ |
| Penalty scale (legacy) | $\eta$ | $1$ |
| JEPA mix | $\rho$ | unset in reported runs (legacy $\eta$); sweep $\{0, .25, .5, .75, 1\}$ prepared |
| Pseudocount | $\kappa$ | $0.002$ |
| Forgetting | $\lambda$ | $1$, so $\gamma_\lambda = 1$ |
| Local iterations | $K$ | $5$ coupled (also $2$ historical) |
| Student context belief | $b_u$ | lag-1 smoothed, $\Psi_{T_{u+1}}$ |
| Clone allocation | $n_o$ | ground truth (also $10$ per symbol) |
| Stream length | $n$ | $20{,}000$ observations, seeds $0$–$2$ |

---

## 9. Correspondence with the implementation

Learner: `src/mercury_runs/algorithms/cscg/distillation.py`, function `train_experiment`,
with `method="predictive"` and `student_inference="smooth"`. Runner:
`scripts/run_cscg_distillation_experiment.py`. Evaluation:
`scripts/analysis/cscg_distillation_matched_eval.py`.

| Symbol | Code |
|---|---|
| $\Phi$, $Z$ | `CarriedFilter.step` |
| $\Xi$ | `pair_counts` |
| $\Psi$, $\mathcal S$ | `smooth_window` (two-node window for $\Psi$) |
| $f_t$, $h_t$ | `cached_student[t]`, `cached_teacher[t]` |
| $b_u$ | `cached_source[u]` |
| $q^t$ | `q` from `smooth_window(teacher, ...)` |
| $\xi_t(T^{(k)})$ | `fast` at $k=0$, otherwise `pair_counts(T, previous, ...)` |
| $\tilde S^{(k)}$, $B^{(k)}$ | `candidate_counts`, `B` |
| $w_t$ | `weight = eta * candidate_counts.sum()`, `eta = config.legacy_penalty_weight` |
| $\rho$, $\rho_t$ | `jepa_mix`; `legacy_jepa_mix(weight, B)` |
| $M^{(k)}$ | `mixed_counts(B, D, rho)` (returned unscaled at the endpoints) |
| update of Section 5.8 | `mm_update(T, counts)` (retains zero-mass rows) |
| $\mathcal L^{(k)}_\rho$ before/after | `diagnostics[...]["local_iterations"]` |
| zero predictive support | `ZeroPredictiveSupportError` |
| $\mathcal L^t$, latent KL, replayed evidence, $D^{(k)}$ | `replay_terms(...)` outputs `[0]`, `[1]`, `[2]`, `[3]` |
| $K$, coupled schedule | `local_updates`, `coupled_auxiliary` |
| $\lambda$, $\kappa$, $\eta$ | `forgetting`, `pseudocount`, `penalty_weight` (`None` = legacy $1$) |
| $H$, $W$, $W_T$ | `teacher_half_life`, `window`, `teacher_window` |
| $T_0$ | runner: `row_normalize(RandomState(seed).rand(A, N, N) + pseudocount)` |
| Decoder, NLL | `fixed_lag_decode`, `decoder_lag` |

---

## 10. Remarks and flagged issues

These points were found while checking the formulation against the implementation. None of
them changes the algorithm. They concern what the mathematics does and does not guarantee.

**R1. Descent guarantee: a stronger claim, and one limitation.**

* The earlier draft stated $F^{(k)}(T^{(k+1)}) \le F^{(k)}(T^{(k)})$. Its objective
  $F^{(k)}$ used the linearised data term $-\langle B^{(k)}, \log T\rangle$ and a weight
  $w^{(k)}$ indexed by $k$. Because $w_t$ is in fact independent of $k$ (Section 4.5), the
  coupled iterations are MM steps on one fixed objective $G_t$. That is the stronger
  Proposition 2.
* Under the historical schedule, the $K-1$ EM-only steps ignore $w_t\mathcal L^t$.
  $G_t$ may therefore *increase* during them. There is no descent guarantee for $G_t$ over
  a whole historical arrival.
* Under either schedule, no single objective is monotone across arrivals: $S_t$, $w_t$,
  $\bar T_t$, the window and $q^t$ all change with $t$.

**R2. Reading the replayed-evidence term with smoothed sources.** The smoothed source
$b_u = \Psi_{T_{u+1}}(f_u, a_u, x_{u+1})$ has already conditioned on $x_{u+1}$, the very
observation whose evidence $Z_u(T)$ measures. So $-\log Z_u$ is not a held-out predictive
likelihood. Let $m = T_{u+1,a_u}\, e_{u+1}$. At $T = T_{u+1}$,

$$
\begin{aligned}
Z_u
&= \frac{\sum_i f_u(i)\, m_i^2}{\sum_i f_u(i)\, m_i} \\
&\ge \sum_i f_u(i)\, m_i
 = Z_{T_{u+1}}(f_u, a_u, x_{u+1}),
\end{aligned}
$$

by $\mathbb E_{f_u}[m^2] \ge (\mathbb E_{f_u}[m])^2$. The smoothed source thus inflates the
replayed evidence relative to the causal prediction. Proposition 1 remains an exact
identity. Only the interpretation of its second term as "fitting replayed observations"
should be qualified. With the filtered variant $b_u = f_u$, the term is a genuine causal
predictive likelihood.

**R3. Carried beliefs are not single-model posteriors.** $f_u$, $h_u$ and $b_u$ are each
computed under the tensor in force at their own arrival, and are never revised. Hence:

* $f_{t-1}$ inside $\xi_t(T^{(k)})$ was computed under $T_{t-1}$, not under $T^{(k)}$;
* $b_u$ uses $T_{u+1}$;
* $h_{s_t}$ was built under $\bar T_{e(t)}, \dots, \bar T_{s_t}$, while the window smoothing
  uses $\bar T_t$.

The identity $q^t_u = p(z_u \mid x_{s_t:t})$ holds for the teacher HMM initialised at
$s_t$ with prior $h_{s_t}$. It does not hold for the CSCG under $\bar T_t$ started from
$\pi$ at $e(t)$. The earlier draft's notation $\bar p(z_u \mid x_{s:t}, h_s)$ is consistent
with this, but left it implicit.

**R4. The newest transition appears twice, asymmetrically.**

* In the data term, it uses the filtered source $f_{t-1}$, with $\xi_t$ recomputed at every
  local iteration and weight $\gamma_\lambda$.
* In the predictive term, as $u = t-1$, it uses the frozen smoothed source $b_{t-1}$, with
  target $q^t_t = \alpha_t$ and weight $w_t/n_t$.

This is by design, but it means the newest transition receives both data and teacher
supervision, while older window transitions receive teacher supervision only.

**R5. The teacher-only window is not exactly a no-op.** With $W_T > W$, the forward message
at $s_t$ is recomputed under $\bar T_t$ from $h_{s'_t}$, instead of taken from the cached
$h_{s_t}$. The two agree, and the targets coincide, when $\bar T$ is constant over
$[s'_t, s_t]$; in general they differ by the teacher's drift. The earlier description of
this option as a "near no-op" is an empirical observation from a stopped pilot run (maximum
$\lvert \Delta T \rvert \approx 3\times10^{-5}$), not a mathematical identity.

**R6. Default mismatch (implementation note).** `ExperimentConfig.forgetting` defaults to
$0.99$, while the runner's `--forgetting` defaults to $1.0$. The reported runs record
$\lambda = 1$ in `experiment.json`. A direct library call that omits `forgetting` would use
$\lambda = 0.99$.

**R7. Pure JEPA loses transition support immediately (smoke result).** The unmodified
$\rho = 1$ objective was run on level 21, seed 0, with ground-truth clones, smoothed
sources and $K = 5$. It stopped with `ZeroPredictiveSupportError` at arrival $2$, in the
student filter. Section 5.11 shows that this is forced by the objective: its minimisers
forget every successor outside the window. Under the same settings, $\rho = 0$ reproduced
short EM bit for bit, and $\rho = 0.5$ completed with strictly positive, row-stochastic
tensors. No preventive mechanism has been added; Section 5.11 lists the candidate
objective modifications.

**R8. Parameter and notation changes.** The forgetting increment previously written
$\rho_\lambda$ is now $\gamma_\lambda$, so that $\rho$ denotes `jepa_mix` alone.
`ExperimentConfig.penalty_weight` now defaults to `None`, which means the legacy value
$\eta = 1$ when `jepa_mix` is unset. The runner still records an explicit `1.0` for
legacy runs. Previously reported configurations therefore run unchanged.

---

## 11. Summary of distinctions

* **Inference versus optimisation.** Filtering ($f_t$, $h_t$), the student's one-step
  smoothing ($b_u$) and the teacher's window smoothing ($q^t$) are probabilistic inference.
  They are identical in every regime whenever their outputs are needed. `jepa_mix` changes
  only which counts form the transition update.
* **Pure EM, hybrid, pure JEPA.** $\rho = 0$ is short-context online EM, with no teacher.
  $0 < \rho < 1$ minimises the mixed surrogate at every iteration. $\rho = 1$ minimises
  the predictive loss alone: no EM counts, no statistics, no pseudocounts and no warm-up.
* **Persistent versus temporary counts.** $S_t$ holds data expected counts and persists
  across arrivals. $D^{(k)}$ is rebuilt at every local iteration from frozen targets, and is
  never committed.
* **Predictive objective versus posterior matching.** $\mathcal L^t$ is the unnormalised
  predictive KL. It equals the latent-matching KL plus the replayed evidence
  (Proposition 1). The normalised posterior-matching objective is a different arm
  (`posterior`) and is not part of the mix.
* **Per-iteration descent versus online behaviour.** Proposition 4 guarantees descent of
  $\mathcal L^{(k)}_\rho$ over one step, and of $F_{\rho,t}$ over the local iterations of
  one arrival. Nothing is claimed across arrivals.
* **Mixing coefficient versus legacy penalty.** $\rho$ is a fixed fraction of two
  unit-mass objectives. The legacy $\eta$ scales the penalty by the running count mass, and
  is equivalent to the time-varying $\rho_t = \eta\tilde m_t / ((1+\eta)\tilde m_t + \kappa A N^2)$,
  combined with its own iteration schedule.
