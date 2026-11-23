# First-Class Snapshot Evaluation Architecture

## Objective

Refactor paper evaluation so that training, evaluation and visualisation are
three independent stages:

1. **Training produces immutable model snapshots.**
2. **Evaluation consumes those snapshots and produces versioned evaluation
   artefacts.**
3. **Plotting consumes evaluation artefacts only.**

No evaluation walks, decoding, alignment, transition projection or paper
metrics should be computed inside the training loop or independently inside
plotting scripts.

The canonical scientific question is:

> After limited online experience, does the frozen learned representation
> support accurate online latent-state inference and faithfully encode the
> one-step action-conditioned transition dynamics encountered under the
> evaluation policy?

The evaluation therefore separates:

* **online latent-state inference quality**, assessed by latent precision and
  node recall;
* **aligned action-conditioned transition fidelity**, assessed primarily by
  occupancy-weighted action-conditioned TV similarity;
* **non-action-conditioned transition fidelity under the evaluation policy**,
  assessed by policy-collapsed TV similarity;
* **transition-support and local-error diagnostics**, assessed by thresholded
  edge metrics and residual maps.

No single metric is interpreted as a complete measure of cognitive-map quality
or downstream planning performance.

The canonical architectural rule is:

> A snapshot is evaluated once under an explicitly versioned protocol. All
> reported metrics and diagnostics for that snapshot must be derived from the
> same decoded walks, cooccurrence data, alignment, empirical held-out
> transition counts, projected transition tensors and validity masks.

# 1. System architecture

## 1.1 Training stage

Training is responsible for:

* loading the training dataset;
* fitting the model;
* saving snapshots at configured checkpoints;
* saving the run configuration;
* saving training-only diagnostics;
* recording the snapshot schema and model type;
* recording all inference-relevant configuration, including the CSCG online
  batch size, initial latent-state policy and transition-tensor convention.

Training is not responsible for:

* generating held-out walks;
* decoding evaluation sequences;
* constructing cooccurrence matrices;
* aligning learned states to true states;
* computing precision, recall, TV or edge metrics;
* producing residual maps;
* writing paper-precision histories.

A training option such as:

```text
save_snapshots: true
snapshot_checkpoints: [...]
```

should control snapshot production. It must not implicitly enable evaluation.

The snapshots must be immutable after training.

---

## 1.2 Study-level evaluation stage

Evaluation should be implemented as a first-class application service, for
example:

```text
src/mercury_runs/application/evaluation_service.py
```

with a command such as:

```bash
python -m mercury_runs evaluate-study \
    --study-root results/current/studies/<study> \
    --protocol configs/evaluation/canonical_v4.yaml
```

The service should:

1. Discover completed runs.
2. Discover snapshots within each run.
3. Group comparable runs by evaluation condition.
4. Create or load one shared evaluation-walk cache.
5. Construct the empirical ground-truth transition counts and distributions
   from the cached held-out walks.
6. Load each snapshot through a method-specific adapter.
7. Decode the shared walks using frozen online inference.
8. Construct the cooccurrence, occupancy and alignment statistics.
9. Call the canonical transition evaluator.
10. Save a complete, versioned evaluation record.
11. Resume safely when some snapshots have already been evaluated.

Evaluation must be independently rerunnable without retraining.

Changing any of the following should create a new evaluation protocol rather
than overwrite an existing one:

* decoder or online-inference semantics;
* inference-state initialisation or reset policy;
* treatment of initial timesteps or burn-in;
* evaluation seed;
* number of walks;
* walk length;
* start-state policy;
* environment implementation version;
* transition ground-truth source;
* self-loop policy;
* action marginalisation policy;
* edge thresholds;
* metric schema version.

The primary protocol uses empirical transition distributions estimated from the
shared held-out evaluation walks, retains all initial timesteps and performs no
burn-in. This deliberately measures in-distribution performance under the
specified evaluation policy. State--action contexts that do not occur under
that policy are not included in the transition target or scored as failures.
CSCG evaluation uses the recursive online inference state associated with the
snapshot's configured batch size; evaluation never updates the graph or its
learned parameters.

---

## 1.3 Plotting and analysis stage

Plotting scripts should consume saved evaluation records only.

They must not:

* load model snapshots;
* generate random walks;
* decode sequences;
* load legacy training-loop cooccurrence matrices;
* compute alignments;
* project learned transitions;
* define their own TV or edge metric implementations.

The plotting layer may derive deterministic visual quantities from stored canonical matrices, such as:

\[
R_{ij}=\hat P_{ij}-P^{GT}_{ij},
\]

\[
E^+_{ij}=\max(R_{ij},0),
\]

\[
E^-_{ij}=\max(-R_{ij},0),
\]

and

\[
M_{ij}=\min(\hat P_{ij},P^{GT}_{ij}).
\]

These calculations do not alter metric semantics.

---

# 2. Evaluation protocol

Each evaluation must have an explicit protocol object.

A protocol should include at least:

```text
schema_version
protocol_name
decoder
decoder_configuration_source
online_inference_mode
inference_state_initialisation
inference_state_reset_policy
initial_timestep_policy
burn_in_steps
evaluation_seed
number_of_walks
walk_length
start_state_policy
environment
environment_implementation_version
sensor
sensor_range
rand_prob
valid_trajectories_only
ground_truth_source
primary_tv_aggregation
secondary_tv_aggregations
primary_self_loop_policy
diagnostic_self_loop_policy
collapsed_action_marginalisation
collapsed_state_aggregation
optional_collapsed_action_marginalisations
edge_thresholds
unmatched_state_policy
alignment_policy
alignment_tie_breaking
```

The canonical defaults are:

```text
decoder: method_standard_online
decoder_configuration_source: snapshot
online_inference_mode: frozen_recursive
inference_state_initialisation: method_standard_reset
inference_state_reset_policy: reset_per_walk
initial_timestep_policy: include_all
burn_in_steps: 0
ground_truth_source: empirical_held_out_counts
primary_tv_aggregation: evaluation_occupancy
secondary_tv_aggregations: [macro]
primary_self_loop_policy: include
diagnostic_self_loop_policy: conditional_non_self_loop
collapsed_action_marginalisation: evaluation_policy
collapsed_state_aggregation: evaluation_occupancy
optional_collapsed_action_marginalisations: []
unmatched_state_policy: explicit_sink
alignment_policy: many_to_one_wta
alignment_tie_breaking: smallest_sorted_state_id
```

The empirical ground-truth tensor is constructed once from the shared held-out
walk cache and stored with the evaluation result. Its state/action ordering,
walk-cache identifier and row-validity mask must be serialised. Because all
methods and checkpoints use byte-identical evaluation walks, they are compared
against exactly the same finite-sample target.

The serialised protocol should be hashed. The hash forms the evaluation
identifier:

```text
canonical-v4-<protocol-hash>
```

The full protocol must be stored with every evaluation result. Any change to
online inference, cold-start treatment, the evaluation walks, TV aggregation,
action marginalisation, self-loop treatment or unmatched-state policy must
produce a different protocol identifier.

## 2.1 Evaluation question and metric roles

| Quantity | Intended question | Significant assumptions | Interpretation |
|---|---|---|---|
| Latent precision | Does each online-decoded learned state identify a specific ground-truth state? | Hard decoded states; many-to-one WTA cooccurrence alignment; evaluation-occupancy weighting. | Penalises unresolved aliasing within a learned state but not redundant pure clone splitting. |
| Node recall | Are visits to each ground-truth state concentrated in one dominant learned state? | Reverse WTA mapping; evaluation-occupancy weighting. | Measures occupancy-weighted fragmentation, not raw node count or graph compactness. |
| Action-conditioned TV similarity | Does the aligned learned graph reproduce the empirical successor distribution for each state--action context encountered under the evaluation policy? | Shared held-out transition counts; post-hoc WTA alignment; action-specific clone mixing; evaluation state--action weighting. | Primary in-distribution transition metric. It is alignment-conditional and must be interpreted jointly with latent precision. |
| Policy-collapsed TV similarity | Does the graph reproduce the non-action-conditioned transition distribution induced by the evaluation policy? | Truth and prediction use the same empirical action prior \(q_{\mathrm{eval}}(a\mid i)\). | Complementary and policy-specific. Action errors may cancel after marginalisation. |
| Thresholded edge precision/recall/F1 | Does the model recover empirically observed transition support above an explicit probability threshold? | Same canonical projected tensors; explicit threshold; explicit self-loop convention. | Support diagnostic, not a probability-calibration metric. |
| Residual maps | Where is probability mass missing or excessive? | Derived from the saved canonical tensors. | Diagnostic only; not an independent metric. |

The canonical protocol measures within-distribution performance under the
specified held-out behaviour policy. It does not evaluate unvisited
counterfactual actions or claim recovery of the complete environment transition
kernel.

---

# 3. Shared evaluation walks

Evaluation walks are study-level artefacts, not run-level artefacts.

The cache key must include every variable that determines the generated trajectories:

```text
environment or level
sensor
sensor range
random-policy probability
valid-trajectory setting
evaluation seed
number of walks
walk length
start-state protocol
environment implementation version
```

The same cached walks must be used across:

* methods;
* checkpoints;
* CSCG batch sizes;
* Hybrid and vanilla CSCG;
* hyperparameter settings;
* decoder variants, where the observation sequence is unchanged.

The training seed should not change the evaluation walks unless this is an intentional part of the protocol.

Every evaluation walk must be decoded from a reset transient inference state.
All timesteps are retained. No common or method-specific burn-in is removed.
This makes the evaluation include cold-start inference before each method has
accumulated its full temporal context.

A suitable study layout is:

```text
study/
├── evaluation_caches/
│   └── canonical-v4-<hash>/
│       ├── protocol.json
│       └── walks/
│           └── level=13_rand=0.3_eval-seed=0.npz
└── runs/
```

---

# 4. Method adapters

Every model type should expose a common snapshot-evaluation interface.

For example:

```python
class SnapshotEvaluationAdapter(Protocol):
    def load_snapshot(self, snapshot_path: Path) -> SnapshotModel:
        ...

    def decode_walks(
        self,
        model: SnapshotModel,
        walks: Sequence[EvaluationWalk],
        protocol: EvaluationProtocol,
    ) -> DecodedEvaluationWalks:
        ...

    def transition_tensor(
        self,
        model: SnapshotModel,
    ) -> np.ndarray:
        ...

    def inference_metadata(
        self,
        model: SnapshotModel,
    ) -> Mapping[str, Any]:
        ...
```

The adapter must produce:

```text
decoded learned-state walks
learned action-conditioned transition tensor T_hat
number of learned states
number of actions
model and decoder metadata
initial inference-state metadata
effective online batch size
confirmation that parameters remained frozen
```

MERCURY, CSCG, Hybrid and other methods should differ only in their adapters.
They should not have separate metric implementations.

## 4.1 MERCURY evaluation inference

MERCURY evaluation uses the frozen winner-take-all inference rule with the
snapshot's temporal and sensory admissibility constraints. Strip memory and
other transient inference state are reset at the start of each independent
walk. They are then filled recursively from the first retained timestep.
Evaluation must not update node templates, edge weights, ambiguity scores,
ages, pruning state or graph structure.

## 4.2 CSCG and Hybrid evaluation inference

CSCG and Hybrid CSCG use the recursive online latent-state inference associated
with the online-EM batch size \(B\) stored in the snapshot configuration.

For each independent evaluation walk:

1. initialise the standard recursive latent-state inference state;
2. process observations sequentially in batches of \(B\);
3. propagate the terminal recursive inference state of one batch into the next;
4. retain the hard state read-out at every timestep;
5. reset the transient state only at the next walk boundary.

The evaluation walk must not update transition counts, transition
probabilities, emission structure, clone allocation or any other learned
parameter. EM is the training procedure; it is not run during evaluation.

The primary decoder metadata should therefore identify this as, for example:

```text
recursive_online_batch
```

with:

```text
effective_batch_size: <B>
parameter_updates: false
state_reset: per_walk
initial_timesteps: retained
```

Full-trajectory Viterbi decoding or Viterbi parameter training must be exposed
only as separately versioned diagnostic protocols. They must not be silently
substituted for the primary recursive online inference.

---

# 5. Canonical transition indexing

Before metric computation, all walks must be converted to one transition convention.

For each transition (t):

\[
\hat z_t \xrightarrow{a_t} \hat z_{t+1},
\qquad
s_t \xrightarrow{a_t} s_{t+1},
\]

where:

* (\hat z_t) is the decoded learned source state;
* (s_t) is the true source state;
* (a_t) is the action causing the transition;
* (\hat z_{t+1}) and (s_{t+1}) are target states.

If the stored walk convention places `action[t]` beside the post-step state at index (t), the evaluator must convert it once:

```python
learned_sources = decoded_states[:-1]
true_sources = true_states[:-1]
learned_targets = decoded_states[1:]
true_targets = true_states[1:]
transition_actions = actions[1:]
```

This conversion belongs in the evaluation data preparation layer. Metric functions should receive already transition-aligned arrays.

---

# 6. Cooccurrence and alignment

Let:

* \(k\in\{1,\ldots,K\}\) index learned states;
* \(i\in\{1,\ldots,N\}\) index true states;
* \(a\in\{1,\ldots,A\}\) index actions.

Alignment is recomputed independently for every method, training seed and
frozen checkpoint from that snapshot's decoded shared held-out walks.

## 6.1 Total state cooccurrence

Construct:

\[
C_{ki}
=
\sum_t
\mathbf{1}
\left[
\hat z_t=k,\;
s_t=i
\right].
\]

This matrix is computed from the shared held-out evaluation walks, including
the retained cold-start timesteps.

It must not come from:

* the training trajectory;
* an earlier training-loop evaluation;
* a plotting script;
* a different evaluation-walk set.

## 6.2 Hard many-to-one alignment

Following the established latent-precision procedure, for every observed
learned state:

\[
A(k)
=
\arg\max_i C_{ki}.
\]

Exact ties are resolved by selecting the smallest ground-truth state identifier
after deterministic sorting.

A learned state with no evaluation evidence must remain explicitly unaligned:

\[
A(k)=-1
\quad\text{when}\quad
\sum_i C_{ki}=0.
\]

It must not default to true state zero through `argmax` on an all-zero row.

Define:

\[
\mathcal K_i
=
\{k:A(k)=i\}.
\]

For the reverse node-recall diagnostic, define:

\[
R(i)
=
\arg\max_k C_{ki},
\]

using the same deterministic tie-breaking convention.

The evaluator should store:

```text
cooccurrence
learned_to_true_alignment
true_to_learned_reverse_mapping
aligned_learned_state_mask
unaligned_learned_states
alignment_tie_count
```

The alignment is a post-hoc evaluation operation. It is not available to the
learner during training or deployment.

---

# 7. Action-conditioned source occupancy

A single unconditional learned-state occupancy is insufficient for action-conditioned projection.

Construct the action-conditioned source cooccurrence:

\[
H_{aki}
=
\sum_t
\mathbf{1}
\left[
a_t=a,\;
\hat z_t=k,\;
s_t=i
\right].
\]

For a true source state (i) and action (a), this measures how frequently learned clone (k) represented that source under that action.

The corresponding projection weight is:

\[
\omega_{aki}
=
\frac{H_{aki}}
{\sum_{k'\in\mathcal K_i}H_{ak'i}},
\]

when the denominator is non-zero.

This makes the projection sensitive to the fact that different learned clones aligned to the same true state may be occupied under different actions.

Using only:

\[
\mu_k=\sum_i C_{ki}
\]

for every action would incorrectly apply the same clone mixture to all action-conditioned rows.

---

# 8. Learned transition probabilities

Let the raw learned transition tensor be:

\[
\hat T_{ak\ell},
\]

where (k) is the learned source and (\ell) the learned target.

Normalise each learned state-action row:

\[
\hat P(\ell\mid k,a)
=
\frac{\hat T_{ak\ell}}
{\sum_{\ell'}\hat T_{ak\ell'}}.
\]

Rows with no outgoing mass remain empty at this stage.

---

# 8.1 Empirical held-out transition ground truth

The canonical transition target is estimated from the same shared held-out
evaluation walks used for alignment and occupancy. Construct:

\[
T^{GT}_{aij}
=
\sum_t
\mathbf{1}
\left[
a_t=a,\;
s_t=i,\;
s_{t+1}=j
\right].
\]

For every empirically observed state--action row:

\[
P^{GT}(j\mid i,a)
=
\frac{T^{GT}_{aij}}
{\sum_{j'}T^{GT}_{aij'}},
\qquad
P^{GT}(\bot\mid i,a)=0.
\]

The valid scoring rows are:

\[
\mathcal V
=
\left\{
(a,i):
\sum_j T^{GT}_{aij}>0
\right\}.
\]

Thus, transition metrics evaluate only state--action contexts encountered under
the held-out evaluation policy. A valid environment action that is never
executed under that policy is outside the evaluation distribution and is not
treated as a failed prediction.

The evaluator must construct and store:

```text
empirical_ground_truth_transition_counts
empirical_ground_truth_action_distributions
ground_truth_state_order
action_order
valid_action_rows
walk_cache_id
ground_truth_count_checksum
```

The empirical target must be computed once per shared walk cache and reused
unchanged across methods, training seeds, hyperparameters and checkpoints.

An exact environment-kernel evaluation may be implemented as a separately
named out-of-distribution or counterfactual diagnostic. It is not canonical and
must create a distinct protocol identifier.

---

# 9. Canonical projection into ground-truth space

## 9.1 Matched target mass

For true source state (i), action (a) and true target state (j):

\[
\hat P_{\mathrm{proj}}(j\mid i,a)
=
\sum_{k\in\mathcal K_i}
\omega_{aki}
\sum_{\ell:A(\ell)=j}
\hat P(\ell\mid k,a).
\]

This performs two operations:

1. mixes learned source clones using action-conditioned source occupancy;
2. maps learned target states through the shared hard alignment.

## 9.2 Unmatched target mass

Transitions into unaligned learned targets must not be silently assigned to state zero or silently discarded.

Introduce an unmatched target symbol (\bot):

\[
\hat P_{\mathrm{proj}}(\bot\mid i,a)
=
\sum_{k\in\mathcal K_i}
\omega_{aki}
\sum_{\ell:A(\ell)=-1}
\hat P(\ell\mid k,a).
\]

The canonical projected target space is therefore:

\[
\mathcal S^{GT}\cup\{\bot}.
\]

Ground truth assigns zero probability to (\bot).

This ensures that transition mass directed towards unaligned learned states is penalised rather than hidden.

The evaluator should separately record:

```text
unmatched_target_mass_by_action_state
total_unmatched_target_mass
```

Residual-map plots may omit (\bot) from the graph while showing its mass as a separate diagnostic.

---

# 10. Self-loop policy

Self-loop treatment must be explicit because removing the diagonal and renormalising changes the estimand.

## 10.1 Primary TV: include self-loops

The primary action-conditioned TV should compare the complete projected transition distributions, including self-loops:

\[
TV_{\mathrm{full}}(a,i)
=
\frac{1}{2}
\sum_{j\in\mathcal S^{GT}\cup\{\bot\}}
\left|
\hat P^{GT}(j\mid i,a)
-
P^{GT}(j\mid i,a)
\right|.
\]

This penalises probability assigned to incorrect stay transitions and retains collision or failed-movement behaviour when those transitions are part of the evaluation protocol.

No diagonal removal is applied for the primary TV.

## 10.2 Diagnostic TV: conditional non-self-loop transitions

A separate topology diagnostic may condition on leaving the source state. For each valid row, remove the diagonal after projection:

\[
\hat P_{\mathrm{move,raw}}(j\mid i,a)
=
\begin{cases}
0, & j=i,\\
\hat P^{GT}(j\mid i,a), & j\neq i,
\end{cases}
\]

and analogously for \(P^{GT}_{\mathrm{move,raw}}\). Renormalise over \(j\neq i\) and the unmatched target \(\bot\).

The diagnostic is defined only for rows with positive ground-truth non-self-loop mass:

\[
\mathcal V_{\mathrm{move}}
=
\left\{
(a,i):
\sum_{j\neq i}P^{GT}(j\mid i,a)>0
\right\}.
\]

If a row in \(\mathcal V_{\mathrm{move}}\) has no predicted non-self-loop mass, assign all predicted mass to \(\bot\), giving TV \(=1\).

Learned transitions must never be zeroed on their learned-state diagonal before projection. Distinct learned clones can map to the same true state, so a learned inter-clone transition may become a self-loop only after alignment.

---

# 11. Final row completion and mass-preservation checks

The source-clone weights, normalised learned rows, aligned target sets and
unmatched target \(\bot\) jointly define a complete projected distribution.
Therefore, a correctly implemented positive-mass projected row should already
sum to one.

The evaluator may correct only floating-point error. It must not use
renormalisation to hide substantively dropped probability mass.

For every positive-mass projected row, assert:

\[
\left|
1-
\sum_{j\in\mathcal S^{GT}\cup\{\bot\}}
\hat P_{\mathrm{proj}}(j\mid i,a)
\right|
<\delta_{\mathrm{numeric}}.
\]

A larger deficit is an evaluation error and should fail the snapshot record.

If a ground-truth-valid state-action pair has no predicted aligned source
occupancy or no learned outgoing mass, assign all predicted probability to
\(\bot\):

\[
\hat P^{GT}(\bot\mid i,a)=1.
\]

This gives an empty predicted row a TV distance of one without a special-case
metric formula.

Ground-truth rows are the normalised empirical held-out distributions in the
same extended target space, with:

\[
P^{GT}(\bot\mid i,a)=0.
\]

For the conditional non-self-loop diagnostic, removal and renormalisation occur
only after canonical projection, as specified in Section 10.2.

---

# 12. Action-conditioned TV

For each empirically observed row \((a,i)\in\mathcal V\), compute:

\[
D_{i,a}
=
\frac{1}{2}
\sum_{j\in\mathcal S^{GT}\cup\{\bot\}}
\left|
\hat P^{GT}(j\mid i,a)
-
P^{GT}(j\mid i,a)
\right|.
\]

## 12.1 Evaluation occupancy

Let:

\[
N_{ai}
=
\sum_j T^{GT}_{aij}
=
\sum_t
\mathbf 1[a_t=a,\;s_t=i].
\]

The same empirical counts both identify the within-distribution scoring rows and
define their operational importance. They depend only on the true held-out
walks and cannot be influenced by the evaluated model.

Define:

\[
d_{\mathrm{eval}}(a,i)
=
\frac{N_{ai}}
{\sum_{(a',i')\in\mathcal V}N_{a'i'}}.
\]

## 12.2 Primary metric: occupancy-weighted action-conditioned TV

The primary transition metric is:

\[
TV_{\mathrm{action,occ}}
=
\sum_{(a,i)\in\mathcal V}
d_{\mathrm{eval}}(a,i)\,D_{i,a}.
\]

The corresponding reported similarity is:

\[
S_{\mathrm{action,occ}}
=
1-TV_{\mathrm{action,occ}}.
\]

This quantity answers:

> For a transition context sampled from the shared held-out walks, how different
> is the aligned learned action-conditioned successor distribution from the
> empirical successor distribution observed under the evaluation policy?

This is an in-distribution and alignment-conditional transition-fidelity
metric. It does not by itself include source-state mislocalisation error and
must be interpreted jointly with latent precision.

## 12.3 Secondary robustness metric: macro action-conditioned TV

Also save:

\[
TV_{\mathrm{action,macro}}
=
\frac{1}{|\mathcal V|}
\sum_{(a,i)\in\mathcal V}
D_{i,a}.
\]

This gives every empirically observed state--action row equal influence and
checks whether strong performance is confined to the most frequent encountered
contexts. It does not evaluate state--action rows absent from the held-out
policy.

## 12.4 Per-state diagnostics

Let:

\[
N_i=\sum_a N_{ai},
\qquad
q_{\mathrm{eval}}(a\mid i)=\frac{N_{ai}}{N_i}.
\]

The occupancy-weighted per-state discrepancy is:

\[
D^{\mathrm{action}}_i
=
\sum_{a:(a,i)\in\mathcal V}
q_{\mathrm{eval}}(a\mid i)\,D_{i,a}.
\]

This should be the default node value for action-conditioned residual maps. A
macro per-state diagnostic may also be stored:

\[
D^{\mathrm{action,macro}}_i
=
\frac{1}{|\mathcal A_i|}
\sum_{a\in\mathcal A_i}D_{i,a},
\]

where \(\mathcal A_i=\{a:(a,i)\in\mathcal V\}\).

The evaluator must save the complete \(D_{i,a}\) matrix, \(N_{ai}\), normalised
weights, both global aggregations and both per-state aggregations.

---

# 13. Action-conditioned edge metrics

Thresholded edge metrics are support diagnostics derived from the same
canonical empirical ground-truth and projected distributions used for TV.

For an explicit threshold \(\epsilon\), define:

\[
E^{GT}_{\epsilon}
=
\left\{
(a,i,j):
(a,i)\in\mathcal V,\;
j\neq i,\;
P^{GT}(j\mid i,a)>\epsilon
\right\},
\]

\[
\hat E_{\epsilon}
=
\left\{
(a,i,j):
(a,i)\in\mathcal V,\;
j\neq i,\;
\hat P^{GT}(j\mid i,a)>\epsilon
\right\}.
\]

Predicted edges into \(\bot\) are retained and count as false positives.
Self-transition edges are excluded from edge-support scoring, but their
probability mass remains in the full-row denominator before thresholding.
Consequently, these metrics measure absolute non-self-loop transition support
under the complete dynamics, not movement topology conditional on leaving the
state.

True positives, false positives and false negatives are pooled across all
valid action-conditioned tuples before precision, recall and F1 are computed.

The implementation must define and serialise empty-set conventions. The
canonical conventions are:

```text
precision = 0 when TP + FP = 0
recall = 0 when TP + FN = 0
F1 = 0 when precision + recall = 0
```

The threshold must always be explicit, and the configured threshold sweep must
be stored with the evaluation result.

Canonical names should reflect probability thresholding:

```text
action_probability_edge_precision
action_probability_edge_recall
action_probability_edge_f1
```

A conditional non-self-loop support metric may be retained only as a
separately named diagnostic using the Section 10.2 distributions.

---

# 14. Policy-collapsed transition definition

Policy-collapsed and action-conditioned TV are different metrics, but both must
be derived from the same canonical projected action tensor.

For states with \(N_i>0\), define the empirical evaluation-policy action prior:

\[
q_{\mathrm{eval}}(a\mid i)
=
\frac{N_{ai}}{\sum_{a'}N_{a'i}}.
\]

Then:

\[
P^{GT}_{\mathrm{col}}(j\mid i)
=
\sum_a
q_{\mathrm{eval}}(a\mid i)
P^{GT}(j\mid i,a),
\]

\[
\hat P_{\mathrm{col}}(j\mid i)
=
\sum_a
q_{\mathrm{eval}}(a\mid i)
\hat P^{GT}(j\mid i,a).
\]

Truth and prediction must use exactly the same \(q_{\mathrm{eval}}(a\mid i)\).
The implementation must never collapse row-normalised prediction while
collapsing raw empirical truth counts.

This metric measures the non-action-conditioned state-transition distribution
induced by the held-out evaluation policy. It is therefore named:

```text
policy_collapsed_tv
```

or, when the aggregation is explicit:

```text
policy_collapsed_occupancy_tv
policy_collapsed_macro_tv
```

## 14.1 Relationship to action-conditioned TV

Policy-collapsed and action-conditioned TV are not interchangeable:

\[
TV\left(
\sum_a q_aP_a,
\sum_a q_a\hat P_a
\right)
\leq
\sum_a q_aTV(P_a,\hat P_a).
\]

Action-specific errors can cancel after marginalisation. Therefore,
occupancy-weighted action-conditioned TV remains the primary transition metric,
while policy-collapsed TV is complementary.

## 14.2 Optional alternative action priors

Policy-independent action marginalisations, such as uniform valid-action
weighting, are not part of the canonical paper protocol. They may be supported
as separately named optional diagnostics, but enabling one must create a new
protocol identifier and must not replace the policy-collapsed metric silently.

---

# 15. Policy-collapsed TV and edge metrics

Let the true source-state occupancy on the shared evaluation walks be:

\[
N_i=\sum_a N_{ai},
\qquad
d_{\mathrm{eval}}(i)
=
\frac{N_i}{\sum_{i'}N_{i'}}.
\]

For every evaluation-covered true source state, compute:

\[
D^{\mathrm{col}}_i
=
\frac{1}{2}
\sum_{j\in\mathcal S^{GT}\cup\{\bot\}}
\left|
\hat P_{\mathrm{col}}(j\mid i)
-
P^{GT}_{\mathrm{col}}(j\mid i)
\right|.
\]

The canonical global policy-collapsed metric is:

\[
TV_{\mathrm{col,occ}}
=
\sum_{i:N_i>0}
d_{\mathrm{eval}}(i)\,
D^{\mathrm{col}}_i,
\qquad
S_{\mathrm{col,occ}}=1-TV_{\mathrm{col,occ}}.
\]

Also save the macro robustness diagnostic:

\[
TV_{\mathrm{col,macro}}
=
\frac{1}{|\mathcal S_{\mathrm{eval}}|}
\sum_{i\in\mathcal S_{\mathrm{eval}}}
D^{\mathrm{col}}_i.
\]

Canonical stored names are:

```text
policy_collapsed_occupancy_tv
policy_collapsed_occupancy_similarity
policy_collapsed_macro_tv
```

Policy-collapsed edge metrics use the same explicit probability-threshold rule
over \((i,j)\), exclude self-transition edges while retaining self-loop mass in
the row denominator, and record the evaluation-policy action marginalisation
that produced the matrix.

---

# 16. Canonical evaluation result

The canonical evaluator should return and serialise a complete result object.

At minimum:

```text
protocol metadata
snapshot metadata
walk-cache identifier
decoder and recursive-inference metadata
cold-start and reset metadata
cooccurrence C
reverse alignment mapping
action-source cooccurrence H
empirical ground-truth transition counts T_GT
true source state-action counts N_ai
alignment
aligned-state mask
source occupancy
action-conditioned source occupancy
unmatched learned states
unmatched target mass
empirical ground-truth action distributions
projected predicted action distributions
valid empirical action rows
latent precision
node recall
full action TV by pair
non-self-loop action TV by pair
evaluation occupancy weights by pair
action TV by state, occupancy weighted
action TV by state, macro
global action TV, occupancy weighted
global action TV, macro
action edge metrics by threshold
evaluation-policy action weights q(a|i)
empirical ground-truth policy-collapsed distributions
predicted policy-collapsed distributions
valid collapsed rows
state occupancy weights
policy-collapsed TV by state
global policy-collapsed TV, occupancy weighted
global policy-collapsed TV, macro
policy-collapsed edge metrics by threshold
```

A suitable schema is:

```json
{
  "schema": "transition_structure_evaluation_v4",
  "protocol_id": "canonical-v4-<hash>",
  "method": "cscg",
  "snapshot_step": 2000,
  "walk_cache_id": "<hash>",
  "ground_truth": {
    "source": "empirical_held_out_counts",
    "transition_count_checksum": "<hash>",
    "state_order": [],
    "action_order": [],
    "T_true_action": [],
    "P_true_action": [],
    "valid_action_rows": []
  },
  "inference": {
    "decoder": "recursive_online_batch",
    "effective_batch_size": 40,
    "parameter_updates": false,
    "state_reset": "per_walk",
    "initial_timestep_policy": "include_all",
    "burn_in_steps": 0
  },
  "alignment": {
    "cooccurrence": [],
    "action_source_cooccurrence": [],
    "learned_to_true": [],
    "true_to_learned": [],
    "aligned_learned_states": [],
    "unmatched_learned_states": [],
    "tie_breaking": "smallest_sorted_state_id"
  },
  "state_representation": {
    "latent_precision": 0.0,
    "node_recall": 0.0
  },
  "action_conditioned": {
    "P_true_full": [],
    "P_hat_full": [],
    "valid_rows": [],
    "source_action_counts": [],
    "evaluation_occupancy_weights": [],
    "tv_by_pair_full": [],
    "tv_by_pair_non_self_loop": [],
    "tv_by_state_occupancy": [],
    "tv_by_state_macro": [],
    "tv_occupancy": 0.0,
    "tv_similarity_occupancy": 0.0,
    "tv_macro": 0.0,
    "unmatched_target_mass": [],
    "edge_metrics_by_threshold": {}
  },
  "policy_collapsed": {
    "action_weights_evaluation_policy": [],
    "state_occupancy_weights": [],
    "P_true": [],
    "P_hat": [],
    "tv_by_state": [],
    "tv_occupancy": 0.0,
    "tv_similarity_occupancy": 0.0,
    "tv_macro": 0.0,
    "edge_metrics_by_threshold": {}
  }
}
```

Optional exact-kernel or counterfactual diagnostics should be namespaced
separately and must not alter the canonical fields.

The saved result must be sufficient to reproduce every reported scalar metric,
threshold sweep and residual map without loading the model snapshot again.

---

# 16.1 Study-level statistical aggregation

Snapshot evaluation produces one result per independently trained run and
checkpoint. Statistical aggregation must occur only after those per-run
results have been saved.

The independent replication unit is the training seed, not an individual
evaluation walk or transition. The canonical aggregation procedure is:

1. pool the shared evaluation walks within each trained seed to compute one
   snapshot metric value;
2. retain one value per training seed, environment, policy and checkpoint;
3. compute means and confidence intervals across training seeds;
4. treat comparisons as paired when methods share matched training data and
   byte-identical evaluation walks.

Where multiple evaluation-policy settings are shown in one summary, compute the
metric separately for each policy and average policy-level values equally
within each training seed unless another weighting rule is explicitly
versioned in the aggregation protocol.

Evaluation walks provide stable metric estimation and common random numbers.
They are not independent experimental replicates.

---

# 17. Code ownership

## `paper_precision.py`

This module should own pure mathematical and data-conversion primitives:

```text
cooccurrence construction
alignment
action-conditioned occupancy construction
action-row normalisation
projection
self-loop masking
action marginalisation
TV
edge metrics
result dataclasses
serialisation helpers
```

It should not own:

* study discovery;
* snapshot discovery;
* CLI behaviour;
* model-specific loading;
* plotting;
* training callbacks.

## `evaluation_service.py`

This should own:

```text
study discovery
run discovery
snapshot iteration
walk-cache management
method adapters
decoding
protocol hashing
resume and overwrite behaviour
evaluation-result persistence
```

## Plotting modules

These should own only:

```text
evaluation-record loading
residual derivation
graph layout
visual encoding
figure writing
CSV summarisation
```

---

# 18. Legacy API policy

Existing functions such as:

```text
compute_weighted_structure_metrics
compute_action_conditioned_structure_metrics
reevaluate_paper_precision_snapshots
regenerate_eval_metrics_common_walks
```

must not remain independent metric implementations.

They should either:

1. be removed; or
2. become thin compatibility wrappers around the canonical evaluator.

Any legacy repair script should be explicitly marked as a migration tool and must not be used to produce final paper results.

There should be exactly one implementation of:

```text
alignment
occupancy weighting
projection
self-loop handling
TV denominator selection
edge thresholding
```

for reported metrics.

---

# 19. Evaluation output layout

A run should contain:

```text
run/
├── config.json
├── run_status.json
├── snapshots/
│   ├── mercury/
│   │   └── step=...
│   └── cscg/
│       └── step=...
└── evaluations/
    └── canonical-v4-<hash>/
        ├── protocol.json
        ├── status.json
        ├── history.json
        └── snapshots/
            ├── step=000040.json
            ├── step=000080.json
            └── step=002000.json
```

Evaluation should never modify:

```text
snapshots/
training artefacts
training configuration
```

A partially completed evaluation should be resumable from its per-snapshot records.

---

# 20. Acceptance criteria

The refactor is complete when all of the following are true:

1. Training runs successfully with no paper evaluation code executed.
2. Every requested checkpoint produces a restorable snapshot containing all
   inference-relevant configuration.
3. A study can be evaluated after training with one command.
4. Comparable snapshots use byte-identical cached evaluation walks.
5. All methods are evaluated through a common adapter interface.
6. MERCURY evaluation resets transient state per walk and performs no graph
   updates.
7. CSCG and Hybrid evaluation use the recursive online state associated with
   the configured batch size and perform no parameter updates.
8. All initial evaluation timesteps are retained and `burn_in_steps` is zero.
9. Empirical ground-truth transition counts are constructed once from the
   shared held-out walks and reused across every method and checkpoint.
10. The canonical valid-row mask includes exactly the state--action rows
    observed in the held-out evaluation walks.
11. Unvisited counterfactual state--action rows are not scored as failures.
12. All metrics use one cooccurrence matrix and one alignment per snapshot.
13. Alignment is many-to-one WTA with deterministic tie-breaking and explicit
    unaligned states.
14. Action-conditioned projection uses action-specific source occupancy.
15. Unobserved learned states are represented explicitly rather than mapped to
    state zero.
16. Unmatched target mass is penalised and recorded.
17. Canonical projection preserves probability mass; substantive deficits fail
    evaluation rather than being silently renormalised.
18. Primary action-conditioned TV includes self-loops.
19. Conditional non-self-loop TV is stored only as an explicitly named
    diagnostic.
20. Primary action-conditioned TV is weighted by true state--action occupancy
    from the shared evaluation walks.
21. Macro action-conditioned TV averages only over empirically observed rows.
22. Action-conditioned TV is documented as in-distribution and
    alignment-conditional and is interpreted jointly with latent precision.
23. Policy-collapsed truth and prediction use the same empirical evaluation
    action prior.
24. Policy-collapsed state rows are aggregated using true evaluation-state
    occupancy, with a macro diagnostic also stored.
25. Edge metrics use empirical held-out support, explicit thresholds and
    serialised empty-set conventions.
26. Self-transition edges are excluded from support metrics while self-loop
    probability remains in the full-row denominator.
27. Residual maps load canonical matrices rather than recomputing them.
28. Legacy metric functions delegate to the canonical evaluator or are removed.
29. Evaluation results contain sufficient information to reproduce every
    reported scalar and plot without loading the model snapshot.
30. Re-evaluating the same snapshot under the same protocol produces identical
    results.
31. Changing inference semantics, cold-start treatment, evaluation walks or
    metric policy creates a new evaluation namespace.
32. Unit tests verify that evaluation never mutates a snapshot.
33. Unit tests verify recursive CSCG state propagation across batch boundaries
    and reset between walks.
34. Unit tests verify that all initial timesteps are included.
35. Unit tests verify that empirical ground-truth counts and valid rows match
    the cached true state/action sequences exactly.
36. Unit tests verify that occupancy weights are derived from true evaluation
    counts and cannot be influenced by the evaluated model.
37. Unit tests verify that occupancy-weighted and macro TV differ as expected
    on deliberately imbalanced evaluation walks.
38. Unit tests verify that action-conditioned TV is no smaller than the
    corresponding policy-collapsed TV in constructed cancellation cases.
39. Unit tests verify that projected mass is either aligned or assigned to
    \(\bot\), with no silent loss.
40. Study summaries compute uncertainty across independently trained seeds, not
    across evaluation transitions or walks.

---

# 21. Recommended implementation sequence

Implement the refactor in this order:

1. Define the canonical-v4 evaluation protocol and result schemas.
2. Implement transition-aligned walk conversion with explicit boundary masks.
3. Implement empirical ground-truth transition counts, row normalisation and
   valid-row masks from the shared held-out walks.
4. Implement total and action-conditioned cooccurrence construction.
5. Implement true state--action and state occupancy counts from the same walks.
6. Implement many-to-one WTA alignment, reverse mapping, deterministic
   tie-breaking and explicit unaligned states.
7. Implement learned action-row normalisation.
8. Implement action-conditioned projection with action-specific clone occupancy
   and unmatched-target mass.
9. Implement projection mass-preservation assertions and missing-row assignment
   to \(\bot\).
10. Implement latent precision and node recall from the same decoded walks and
    alignment.
11. Implement primary full action-conditioned TV against the empirical
    held-out distributions with evaluation-occupancy aggregation.
12. Implement macro action-conditioned TV and per-state diagnostics over
    empirically observed rows.
13. Implement conditional non-self-loop TV as a separately named diagnostic.
14. Implement thresholded action-conditioned edge metrics with explicit
    self-loop and empty-set conventions.
15. Implement policy-collapsed action marginalisation from empirical
    action-conditioned truth and prediction using the same evaluation-policy
    action prior.
16. Implement occupancy-weighted and macro policy-collapsed TV and edge metrics.
17. Add unit tests for every mathematical primitive and aggregation policy.
18. Implement the MERCURY frozen online adapter.
19. Implement the CSCG/Hybrid recursive online batch adapter, including
    cross-batch state propagation and per-walk reset.
20. Add snapshot-mutation tests and cold-start inclusion tests.
21. Implement the study-level walk cache.
22. Implement the evaluation service, protocol hashing and CLI.
23. Implement per-training-seed study aggregation and paired confidence
    summaries.
24. Convert legacy metric functions into wrappers.
25. Convert residual and threshold scripts into pure consumers.
26. Regenerate all paper evaluation artefacts from snapshots under
    canonical-v4.

The final architectural invariant is:

> Training produces snapshots. Evaluation applies frozen online inference and
> transforms snapshots plus shared held-out true trajectories into versioned
> canonical distributions and metrics. Plotting presents those saved
> distributions and metrics without reconstructing them.

The primary transition estimand is:

\[
\boxed{
TV_{\mathrm{action,occ}}
=
\mathbb E_{(S,A)\sim d_{\mathrm{eval}}}
\left[
TV\left(
P_{\mathrm{eval}}^{GT}(\cdot\mid S,A),
\hat P^{GT}(\cdot\mid S,A)
\right)
\right]
}
\]

where \(P_{\mathrm{eval}}^{GT}\) is estimated from the shared held-out walks.
Latent precision is reported alongside it to assess source-state inference,
policy-collapsed TV is retained as the complementary non-action-conditioned
summary, and macro TV exposes failures across encountered but infrequent rows.
