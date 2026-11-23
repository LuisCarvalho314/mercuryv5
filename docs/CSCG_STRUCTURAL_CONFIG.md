# CSCG Structural Initialization Config

This document covers the CSCG structural initialization controls used by:

- `cscg`
- `mercury_cscg_hybrid`
- `tmgwr_cscg_hybrid`

All controls are CSCG-stage parameters. TMGWR and Mercury are only possible structure providers when `cscg_structural_source` is `source_graph`.

## Core Switches

`cscg_structural_init_mode`

- `none`: do not write structural counts into `model.C`.
- `binary`: write binary support only. Any nonzero source edge becomes support value `1`.
- `count`: write weighted/count values from the source.

Default CLI value: `binary`.

`cscg_structural_source`

- `none`: no structural source. Equivalent to no structural initialization.
- `oracle_maze_topology`: uses exact environment topology.
- `oracle_empirical_ground_truth`: uses ground-truth transitions observed in the dataset.
- `source_graph`: uses the source graph produced by the hybrid bootstrap stage.

For pure `cscg`, oracle sources are the meaningful structural sources. `source_graph` is only meaningful in hybrid methods because pure CSCG has no source graph stage.

`cscg_structural_transfer_level`

- `clone_state`: transfer state/clone-level edges directly when the source has state-level structure.
- `first_order_observation`: collapse structure to action-conditioned observation-to-observation edges, then apply each observation edge uniformly across all source-observation and target-observation clone pairs.

`first_order_observation` is the general safe transfer mode. It is the intended mode for TMGWR graph transfer.

`cscg_clone_capacity_mode`

- `source`: derive clone capacity from the source assignment where applicable. In pure CSCG, this falls back to the selected clone allocation because there is no source graph.
- `ground_truth`: allocate clones from distinct ground-truth states per observation, multiplied by `cscg_clones_per_state`.
- `fixed`: use fixed configured capacity, normally `cscg_clones_per_obs`, unless a ground-truth clone allocation is explicitly selected.

Capacity controls clone counts. Structural initialization controls transition counts. Disabling structural initialization does not disable clone allocation.

## Study Presets

Study configs may use `cscg_structural_preset` to name a complete CSCG structural/capacity condition. Presets are expanded by the study service before worker commands are built; the expanded run still carries the explicit `cscg_structural_*`, clone allocation, and capacity values.

Supported presets:

- `vanilla_cscg`: fixed uniform clone capacity, no structural initialization.
- `oracle_capacity`: ground-truth/oracle capacity, no structural initialization.
- `oracle_capacity_structural`: ground-truth/oracle capacity plus binary oracle maze topology at clone/state level.
- `oracle_capacity_first_order`: ground-truth/oracle capacity plus binary oracle maze topology collapsed to first-order observation edges.
- `source_graph_clone_state`: source graph capacity plus binary clone/state source-graph transfer.
- `source_graph_first_order`: source graph capacity plus binary first-order observation transfer from the source graph.

If both `cscg_clone_preset` and `cscg_structural_preset` are present, clone preset expansion happens first and structural preset expansion happens second. Structural presets are therefore allowed to overwrite clone allocation and capacity fields when the condition requires it.

## What Happens With No Structural Initialization

Structural initialization is disabled when either of these is set:

```text
--cscg_structural_init_mode none
--cscg_structural_source none
```

In that case:

- no structural prior counts are written into `model.C`;
- no structural transfer call updates `model.T`;
- CSCG trains from the normal CHMM initialization and pseudocount behavior;
- clone allocation still happens normally.

## Method Interactions

### Pure CSCG

Useful source choices:

- `oracle_maze_topology`
- `oracle_empirical_ground_truth`
- `none`

`source_graph` is accepted as a config value but pure CSCG has no source graph, so it produces no structural support.

Useful transfer modes:

- `first_order_observation` works with oracle topology and empirical ground-truth transitions.
- `clone_state` is meaningful for oracle maze topology when clone capacity matches ground-truth state structure.

### Mercury-CSCG Hybrid

`source_graph` means the Mercury latent graph.

Useful combinations:

- `source_graph` + `clone_state`: direct Mercury latent graph edge transfer into CSCG clone states.
- `source_graph` + `first_order_observation`: collapse Mercury graph edges to observation edges, then spread uniformly across clone blocks.
- `oracle_*` sources: use CSCG oracle structural initialization instead of Mercury graph structure.
- `none`: train CSCG after Mercury capacity/bootstrap without a structural transition prior.

### TMGWR-CSCG Hybrid

`source_graph` means the standalone sensory/TMGWR graph trained inside the hybrid run.

Intended combination:

```text
--method tmgwr_cscg_hybrid
--cscg_structural_source source_graph
--cscg_structural_transfer_level first_order_observation
--cscg_structural_init_mode binary
--cscg_clone_capacity_mode source
```

This uses TMGWR graph edge `action` features, collapses graph edges to action-conditioned observation edges, and applies them uniformly across CSCG clone blocks.

Oracle source choices remain CSCG oracle choices even in `tmgwr_cscg_hybrid`; they do not mean TMGWR oracle behavior.

## Common Presets

No structural prior:

```text
--cscg_structural_init_mode none
--cscg_structural_source none
```

Default oracle CSCG support:

```text
--cscg_structural_source oracle_maze_topology
--cscg_structural_transfer_level first_order_observation
--cscg_structural_init_mode binary
```

TMGWR first-order source graph support:

```text
--method tmgwr_cscg_hybrid
--cscg_structural_source source_graph
--cscg_structural_transfer_level first_order_observation
--cscg_structural_init_mode binary
```

Mercury clone/state source graph support:

```text
--method mercury_cscg_hybrid
--cscg_structural_source source_graph
--cscg_structural_transfer_level clone_state
--cscg_structural_init_mode binary
```

## Study Config Notes

`configs/studies/study.json` is now a runnable default matrix, not a full Cartesian catalog. It schedules standalone method runs for:

- vanilla CSCG
- CSCG with oracle capacity
- CSCG with oracle capacity and clone/state structural initialization
- CSCG with oracle capacity and first-order structural initialization
- Mercury-CSCG hybrid
- TMGWR-CSCG hybrid

The allowed low-level CSCG structural values are:

- `cscg_structural_transfer_level`: `clone_state`, `first_order_observation`
- `cscg_structural_init_mode`: `none`, `binary`, `count`
- `cscg_structural_source`: `none`, `oracle_maze_topology`, `oracle_empirical_ground_truth`, `source_graph`
- `cscg_clone_capacity_mode`: `source`, `ground_truth`, `fixed`

Do not enumerate every low-level value in a single enabled method grid unless you intentionally want the full Cartesian product. Some combinations are diagnostic or no-op combinations, such as pure CSCG with `source_graph`.

For a runnable TMGWR smoke configuration, use:

```text
configs/studies/tmgwr_cscg_hybrid_3seed_smoke.json
```
