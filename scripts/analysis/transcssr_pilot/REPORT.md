# Evaluation-walk pilot results

Run: 29 September 2026. The machine-readable results and every held-out state
path are in [`results/current/transcssr_pilot/`](../../../results/current/transcssr_pilot/).
All 36 fits completed; there were no failed runs. The 72 pairwise comparisons
cover walks 1 and 2 separately.

## Preflight and method

The selected caches are from `hybrid_cscg_full` under canonical-v4, with
cardinal-distance range 1, random-action probability 0.3, seed 0, and 10,000
steps per walk. Level 20 has five physical positions and three cardinal output
tokens; level 13 has eight positions and six cardinal output tokens. Every
selected walk has 10,000 aligned observation and action rows. The action
alphabet has four one-hot vectors on both levels. There are no output tokens
seen only in held-out walks. Walks 0, 1, and 2 retain respectively 5,329,
5,291, and 5,385 collision steps on level 20, and 4,957, 4,956, and 5,066 on
level 13. Cached collision flags match physical self-transitions from row 1
onward, and the project's action-alignment check passed on every walk.

Fits used the token-first transCSSR API at pinned commit
`23a53ba21101cf26bdfcb83570d60bb6147d07ba`, with `alpha=0.001`,
`test="chi2"`, `stats_mode="correct"`, training lengths 2,000/5,000/10,000,
and `L_max` 3/5/10. Each fit used only walk 0. The integer token mapping and
full cache SHA-256 are in the level input files. The Cartesian-output control
used the same actions and walks; only its output token was the cached exact
physical-state ID. Cardinal fits never received physical labels.

The tables compare filtered state `t` with physical state `t-1`, since the
state predicts the observation resulting from action `t`. Row 0 has no cached
pre-action physical state. The synchronized fraction uses rows 1–9,999;
precision and recall use only rows with a synchronized state. Fixed-map
precision and recall use the project's existing implementation. Numbers below
are the range across held-out walks 1 and 2.

## Cardinal-distance output

| Level | Train steps | `L_max` | Fitted states | Synchronized | Physical precision | Physical recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 2,000 | 3 | 16 | 0.999–1.000 | 0.868–0.871 | 0.421–0.422 |
| 20 | 2,000 | 5 | 251 | 0.725–0.726 | 0.969–0.971 | 0.234–0.239 |
| 20 | 2,000 | 10 | 1,387 | 0.065–0.071 | 0.998–1.000 | 0.145–0.182 |
| 20 | 5,000 | 3 | 13 | 1.000–1.000 | 0.907–0.910 | 0.709–0.711 |
| 20 | 5,000 | 5 | 321 | 0.876–0.881 | 0.967–0.967 | 0.232–0.238 |
| 20 | 5,000 | 10 | 3,277 | 0.113–0.113 | 1.000–1.000 | 0.099–0.131 |
| 20 | 10,000 | 3 | 15 | 1.000–1.000 | 0.908–0.911 | 0.709–0.711 |
| 20 | 10,000 | 5 | 357 | 0.942–0.946 | 0.960–0.963 | 0.233–0.238 |
| 20 | 10,000 | 10 | 5,986 | 0.163–0.168 | 1.000–1.000 | 0.068–0.092 |
| 13 | 2,000 | 3 | 13 | 0.991–0.993 | 0.900–0.902 | 0.817–0.818 |
| 13 | 2,000 | 5 | 240 | 0.742–0.744 | 0.982–0.984 | 0.205–0.210 |
| 13 | 2,000 | 10 | 1,466 | 0.038–0.043 | 1.000–1.000 | 0.085–0.107 |
| 13 | 5,000 | 3 | 10 | 1.000–1.000 | 0.963–0.964 | 0.925–0.925 |
| 13 | 5,000 | 5 | 10 | 1.000–1.000 | 0.963–0.964 | 0.925–0.925 |
| 13 | 5,000 | 10 | 10 | 0.996–0.999 | 0.963–0.964 | 0.925–0.925 |
| 13 | 10,000 | 3 | 10 | 1.000–1.000 | 0.963–0.964 | 0.925–0.925 |
| 13 | 10,000 | 5 | 10 | 1.000–1.000 | 0.963–0.964 | 0.925–0.925 |
| 13 | 10,000 | 10 | 10 | 0.996–0.999 | 0.963–0.964 | 0.925–0.925 |

For the 10,000-step cardinal fits, every synchronized state maps to exactly
one *preceding observation* on both levels (fixed-map observation precision
1.0). Current-output precision is 0.815–0.897 on level 20 and 0.564–0.565 on
level 13, depending on depth. The state is a predictor of the current output,
so a one-to-one current-output relationship is not expected. Full observation
and physical cooccurrence matrices, ARIs, and per-walk rates are in
`summary.json`.

## Stability across training lengths

Adjusted Rand index compares partitions on the same held-out walk, using only
steps synchronized by both models. Each cell gives the range across walks 1
and 2. The fraction of steps eligible for each score is in `summary.json`;
it is especially small at `L_max=10` with 2,000 training steps.

| Level | `L_max` | 2k vs 5k | 2k vs 10k | 5k vs 10k |
| --- | ---: | ---: | ---: | ---: |
| 20 | 3 | 0.658 | 0.627–0.629 | 0.978–0.980 |
| 20 | 5 | 0.832–0.835 | 0.789–0.802 | 0.810–0.812 |
| 20 | 10 | 0.759–0.789 | 0.729–0.810 | 0.903–0.922 |
| 13 | 3 | 0.832 | 0.832 | 1.000 |
| 13 | 5 | 0.159–0.168 | 0.159–0.168 | 1.000 |
| 13 | 10 | 0.025–0.033 | 0.025–0.033 | 1.000 |

The 5k and 10k cardinal partitions are identical on level 13 for every depth
where both synchronize. On level 20, `L_max=3` is the most stable and broadly
synchronized cardinal fit. The `L_max=5` and `10` partitions fragment into
hundreds or thousands of states; their physical recall falls as training grows.
For example, the level-20 10k/`L_max=10` model fits 5,986 states, visits only
785 or 688 on the held-out walks, and synchronizes only 16–17% of scored
steps. Its physical precision of 1.0 therefore does not indicate a good
global physical reconstruction.

The Cartesian control fits exactly five states on level 20 and eight on level
13 for all nine length/depth combinations. Every control fit has physical
precision, recall, and ARI of 1.0 on its synchronized rows; synchronization is
at least 0.996. Its partitions are identical across training lengths (pairwise
ARI 1.0). This confirms that the action alignment and filter can recover the
physical partition when the outputs expose position directly.

## Structural interpretation and limits

Level 13's stable cardinal model has ten states for eight physical positions.
On held-out walk 1, eight states are pure by physical position, while two
states each mix a physical pair: `{3,4}` and `{1,6}`. This same structure
persists from 5k to 10k. Level 20's 10k/`L_max=3` model has 15 states for five
positions; most states are position-specific, but several mix positions 1–3.
These are structured, repeatable overlaps, although the physical scores show
they are not an exact position partition.

The observed deterministic transition/output table covers every physical
state and all four actions on both levels. Refining physical states by their
action-conditioned outputs and successor classes leaves **five singleton
classes on level 20 and eight on level 13**. Therefore none of the physical
pairs mixed by these particular fits is exactly predictive-equivalent in the
complete observed one-step transition system. More history or a different
inference method could separate them. An exact state-count match is still not
a criterion imposed on transCSSR: causal states concern predictive histories,
and finite data, history truncation, and synchronization change the fitted
partition.

The deepest fits show the main incompatibility in this pilot: sparse observed
histories at the requested depth cause state proliferation and poor held-out
synchronization, especially on level 20. The transCSSR token-first filter also
rejects tuple-token streams as ensemble-like, so vectors were losslessly
encoded as integers. No benchmark machinery or project dependencies changed.
