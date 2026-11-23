# Study Configuration

`configs/studies/study.json` is the default runnable study configuration. It schedules the current standalone comparison matrix directly:

- vanilla CSCG
- CSCG with oracle capacity
- CSCG with oracle capacity and clone/state structural initialization
- CSCG with oracle capacity and first-order structural initialization
- Mercury-CSCG hybrid
- TMGWR-CSCG hybrid

Use this file as the default experiment template. The sections below document the full set of available knobs; do not expand every documented value in the same enabled grid unless you intentionally want the full Cartesian product.

## File Shape

The study config is hierarchical:

```json
{
  "environment": {},
  "evaluation": {},
  "methods": {
    "mercury": {"enabled": false, "grid": {}},
    "sensory": {"enabled": false, "grid": {}},
    "pocml": {"enabled": false, "grid": {}},
    "cscg": {"enabled": true, "grid": {}},
    "mercury_cscg_hybrid": {"enabled": true, "grid": {}},
    "tmgwr_cscg_hybrid": {"enabled": true, "grid": {}}
  }
}
```

Each value in `environment`, `evaluation`, and each method `grid` must be a list. Study expansion computes the Cartesian product of:

- `environment`
- `evaluation`
- one enabled method's `grid`

Method sections are expanded independently. Enabling multiple methods schedules separate runs for each method.

## Running

Run the default config:

```bash
PYTHONPATH=. python main.py --study --study_config configs/studies/study.json --study_name my_study
```

Parallel workers:

```bash
PYTHONPATH=. python main.py --study --study_config configs/studies/study.json --study_name my_study --max_workers 4
```

Resume:

```bash
PYTHONPATH=. python main.py --study --study_config configs/studies/study.json --study_name my_study --resume
```

Important environment note: study workers use `python3` from `StudyDefaults` unless changed in code. In this workspace, the active virtualenv `python` has dependencies while system `python3` may not. If workers fail with missing dependencies, run with the environment's Python or adjust `StudyDefaults.python_executable`.

## Environment Section

Environment values apply to every enabled method.

`level`

Maze level index. Each value schedules a separate dataset/run combination.

`sensor`

Allowed values:

- `cartesian`
- `cardinal distance`

`sensor_range`

Required for `cardinal distance`; usually `1`.

`seed`

Dataset and method seed unless a method-specific seed overrides it, such as `cscg_seed`.

`rand_prob`

Random action probability for generated walks.

`num_steps`

Training dataset length and source-stage training budget.

`datasets_root`

Dataset root directory, normally `datasets`.

`valid_trajectories_only`

Global valid-transition filter. Method-specific filters can override behavior for specific stages.

`mercury_valid_trajectories_only`

Filter for Mercury training.

`mercury_split_sensory_raw_latent_valid`

When enabled, Mercury can use different validity filtering between sensory/raw and latent stages.

`mercury_input_mode`

Allowed values:

- `pretrained_raw`
- `pretrained_cscg_one_hot`
- `direct_cscg_one_hot`

Controls Mercury's sensory input representation.

`sensory_observation_mode`

Allowed values:

- `raw`
- `cscg_one_hot`

Controls standalone sensory/TMGWR input representation.

`pocml_valid_trajectories_only`

Filter for POCML baseline data.

`cscg_valid_trajectories_only`

Filter for CSCG baseline and CSCG hybrid stage data.

`random_start`

Controls whether generated walks start randomly.

`no_reuse_existing_dataset`

CLI flag form. In config files this is normally `[false]`.

`reuse_existing_run`

Allows matching completed runs to be reused.

`no_reuse_existing_run`

CLI flag form. In config files this is normally `[false]`.

`run_id`

Explicit run id. Usually `null`; deterministic run identities are derived from config.

`snapshot_source_run_id`

Used by snapshot/evaluation workflows. Usually `null` for normal study runs.

`window_length`

Window length for methods that consume sequence windows or history context.

## W&B Section

These live in `environment` because they apply to every scheduled run.

`wandb`

Enable W&B logging.

`wandb_project`, `wandb_entity`, `wandb_group`, `wandb_tags`, `wandb_mode`, `wandb_job_type`, `wandb_log_artifacts`, `wandb_run_name`

Standard W&B run metadata and logging controls. `wandb_mode` supports:

- `online`
- `offline`
- `disabled`

## Evaluation Section

`paper_precision`

Enables paper-precision evaluation artifacts.

`paper_precision_mode`

Allowed values:

- `off`
- `final`
- `per_iteration`

`final` evaluates only final artifacts. `per_iteration` writes snapshots during training according to the schedule.

`paper_precision_schedule`

Allowed values:

- `log`
- `linear`

Used when checkpoints are generated from interval/point settings.

`paper_precision_eval_interval`

Interval between evaluations when not using explicit checkpoints.

`paper_precision_eval_checkpoints`

Explicit checkpoint list. Wrap the list as a single grid value:

```json
"paper_precision_eval_checkpoints": [[10, 100, 1000]]
```

`paper_precision_num_points`

Number of generated schedule points. Use `null` with explicit checkpoints.

`paper_precision_num_walks`

Number of evaluation walks.

`paper_precision_walk_length`

Length of each evaluation walk.

`structure_metrics`

Enable structural metrics as part of paper precision.

`structure_metrics_epsilon`

Primary edge threshold.

`structure_metrics_epsilons`

List of edge thresholds. Wrap as one grid value:

```json
"structure_metrics_epsilons": [[1e-6, 0.2]]
```

`structure_metrics_ignore_self_loops`

Exclude self loops from structural metrics.

`structure_metrics_scope`

Allowed values:

- `exact_level`
- `walk_local`

`exact_level` uses the full exact maze state space. `walk_local` restricts to states reached by evaluation walks.

`structure_metrics_ground_truth_source`

Allowed values:

- `empirical_walks`
- `maze_topology`

`computational_eval`

Enable computational measurement artifacts.

`computational_eval_profile`

Allowed value:

- `full`

## Method Sections

Each method has:

```json
"method_name": {
  "enabled": false,
  "grid": {}
}
```

Enabled methods generate runs independently. A parameter belongs in a method grid when it should vary only for that method.

## Mercury

Method name: `mercury`.

Mercury can optionally run paired baselines in the same run:

- `baseline_pocml`
- `baseline_cscg`
- `baseline_all`

This is optional paired-comparison behavior. The default study runs CSCG as its own standalone method instead.

Core sensory parameters:

- `sensory_allow_self_loops`
- `activation_threshold`
- `topological_neighbourhood_threshold`
- `sensory_max_neurons`
- `sensory_weighting`
- `winning_node_lr`
- `topological_neighbourhood_lr`
- `sensory_action_lr`
- `global_context_lr`
- `sensory_max_age`
- `sensory_gaussian_shape`

Core latent parameters:

- `latent_allow_self_loops`
- `latent_max_neurons`
- `latent_action_lr`
- `latent_gaussian_shape`
- `latent_max_age`
- `ambiguity_threshold`
- `ambiguity_decay`
- `ambiguity_counter_mode`
- `trace_decay`
- `lambda_trace`
- `weight_memory`
- `weight_undirected`
- `weight_base`
- `weight_action`
- `memory_replay`
- `memory_disambiguation`

Action-map parameters:

- `am_n_codebook`
- `am_lr`
- `am_sigma`
- `am_key`
- `am_identity_for_one_hot`

`weight_preset`

Optional preset expanded by `study_service.py`. Supported values:

- `null`
- `baseline`
- `memory_dominant_action_only_80`
- `memory_dominant_action_only_90`
- `memory_dominant_action_only_95`
- `memory_dominant_action_only_99`
- `only_memory`

When a preset is not `null`, it overwrites the explicit weight values while preserving the preset label in the run config.

## Sensory / TMGWR

Method name: `sensory`.

This trains only the standalone sensory/TMGWR graph. It uses sensory and action-map parameters:

- `memory_length`
- `sensory_allow_self_loops`
- `activation_threshold`
- `topological_neighbourhood_threshold`
- `sensory_max_neurons`
- `sensory_weighting`
- `winning_node_lr`
- `topological_neighbourhood_lr`
- `sensory_action_lr`
- `global_context_lr`
- `sensory_max_age`
- `sensory_gaussian_shape`
- `am_n_codebook`
- `am_lr`
- `am_sigma`
- `am_key`
- `am_identity_for_one_hot`

Standalone sensory ignores Mercury latent parameters and CSCG parameters.

## POCML

Method name: `pocml`.

POCML parameters:

- `pocml_epochs`
- `pocml_trajectory_length`
- `pocml_max_trajectories`
- `pocml_state_dim`
- `pocml_random_feature_dim`
- `pocml_n_states`
- `pocml_precision_capacities`
- `pocml_alpha`
- `pocml_lr_q`
- `pocml_lr_v`
- `pocml_lr_all`
- `pocml_lr_m`
- `pocml_reg_m`
- `pocml_max_iter_m`
- `pocml_eps_m`
- `pocml_batch_size`
- `pocml_memory_bias`
- `pocml_use_ground_truth_node_ids`

POCML ignores Mercury, sensory/TMGWR, and CSCG structural parameters.

## CSCG

Method name: `cscg`.

Clone allocation parameters:

- `cscg_clones_per_obs`
- `cscg_clone_allocation`
- `cscg_clones_per_state`
- `cscg_clone_preset`
- `cscg_clone_capacity_mode`

`cscg_clone_allocation` supports:

- `uniform`
- `ground_truth`
- `oracle_ground_truth`

`cscg_clone_preset` supports:

- `null`
- `ground_truth_state_1`
- `ground_truth_state_2`
- `uniform_obs_1`
- `uniform_obs_2`
- `uniform_obs_4`
- `uniform_obs_8`

Presets overwrite `cscg_clone_allocation`, `cscg_clones_per_state`, and `cscg_clones_per_obs`.

Training parameters:

- `cscg_n_iter`
- `cscg_pseudocount`
- `cscg_term_early`
- `cscg_train_algorithm`
- `cscg_training_mode`
- `cscg_batch_size`
- `cscg_online_lambda`
- `cscg_eval_viterbi`
- `cscg_seed`

`cscg_train_algorithm` supports:

- `em`
- `viterbi`

`cscg_training_mode` supports:

- `full`
- `online_em`

`online_em` interprets `cscg_n_iter` as an observed-sample budget in the current pipeline.

Structural initialization parameters:

- `cscg_structural_preset`
- `cscg_structural_transfer_level`
- `cscg_structural_init_mode`
- `cscg_structural_source`
- `cscg_clone_capacity_mode`

`cscg_structural_preset` supports:

- `null`
- `vanilla_cscg`
- `oracle_capacity`
- `oracle_capacity_structural`
- `oracle_capacity_first_order`
- `source_graph_clone_state`
- `source_graph_first_order`

Structural presets are study-level convenience names. They expand to ordinary CSCG clone, capacity, and structural initialization parameters before worker commands are built. If a run includes both `cscg_clone_preset` and `cscg_structural_preset`, the structural preset is applied last and can overwrite clone allocation/capacity fields.

See [CSCG_STRUCTURAL_CONFIG.md](CSCG_STRUCTURAL_CONFIG.md) for detailed structural-transfer behavior.

## Mercury-CSCG Hybrid

Method name: `mercury_cscg_hybrid`.

This runs a Mercury source stage, then initializes and trains CSCG.

It uses:

- Mercury sensory, latent, action-map, memory, and weight parameters.
- CSCG clone, structural, and training parameters.
- `mercury_cscg_prior_weight` for online EM initial count weighting.

When `cscg_structural_source` is `source_graph`, the source graph is the Mercury latent graph. Oracle structural sources remain CSCG oracle sources.

Common Mercury hybrid source-graph transfer:

```json
"cscg_structural_preset": ["source_graph_clone_state"]
```

## TMGWR-CSCG Hybrid

Method name: `tmgwr_cscg_hybrid`.

This runs a standalone sensory/TMGWR source stage, then initializes and trains CSCG.

It uses:

- Sensory/TMGWR and action-map parameters.
- CSCG clone, structural, and training parameters.
- `mercury_cscg_prior_weight` for online EM initial count weighting. The name is legacy; it applies to the hybrid CSCG prior weight.

When `cscg_structural_source` is `source_graph`, the source graph is the TMGWR graph. Oracle structural sources remain CSCG oracle sources and do not become TMGWR-specific behavior.

Recommended TMGWR source-graph transfer:

```json
"cscg_structural_preset": ["source_graph_first_order"]
```

## Baseline Interactions

`baseline_pocml`, `baseline_cscg`, and `baseline_all` are meaningful for `mercury`, `mercury_cscg_hybrid`, and `tmgwr_cscg_hybrid`.

- `baseline_pocml`: run POCML baseline alongside the target method.
- `baseline_cscg`: run CSCG baseline alongside the target method.
- `baseline_all`: enable all supported baselines.

For standalone `pocml` and `cscg`, run the method directly with `method: "pocml"` or `method: "cscg"` instead of scheduling them as baselines under another method.

## Grid Size

Every list multiplies the number of runs. For example:

```json
"seed": [0, 1, 2],
"rand_prob": [0.0, 0.3],
"cscg_train_algorithm": ["em", "viterbi"]
```

This produces `3 * 2 * 2 = 12` runs before any other grid keys are counted.

`study.json` currently keeps only `mercury` enabled, so validation resolves a manageable run count. Enabling disabled catalog sections without narrowing their grids will create a very large study.

## Narrowing Examples

Three-seed TMGWR hybrid source-graph smoke:

```json
"environment": {
  "seed": [0, 1, 2],
  "num_steps": [200]
},
"methods": {
  "mercury": {"enabled": false, "grid": {}},
  "tmgwr_cscg_hybrid": {
    "enabled": true,
    "grid": {
      "cscg_structural_source": ["source_graph"],
      "cscg_structural_transfer_level": ["first_order_observation"],
      "cscg_structural_init_mode": ["binary"],
      "cscg_training_mode": ["full"],
      "cscg_train_algorithm": ["em"],
      "cscg_n_iter": [2]
    }
  }
}
```

No structural CSCG baseline:

```json
"methods": {
  "cscg": {
    "enabled": true,
    "grid": {
      "cscg_structural_source": ["none"],
      "cscg_structural_init_mode": ["none"]
    }
  }
}
```

Oracle CSCG structural baseline:

```json
"methods": {
  "cscg": {
    "enabled": true,
    "grid": {
      "cscg_structural_source": ["oracle_maze_topology"],
      "cscg_structural_transfer_level": ["first_order_observation"],
      "cscg_structural_init_mode": ["binary"]
    }
  }
}
```

## Keys Not Meant For `study.json`

These are command modes or orchestration flags, not normal per-run catalog values:

- `study`
- `_worker_run`
- `study_config`
- `max_workers`
- `no_parallel`
- `resume`
- `continue_on_error`
- `retry_failed`
- `retry_incomplete`
- `evaluate_study`
- `evaluate_study_snapshots`
- `reevaluate_paper_precision_snapshots`
- `eval_force`

Pass these on the command line when needed.
