from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.analysis.evaluate_v5 import (
    EvaluationTask,
    EvaluationV5MatchedBudgetProtocol,
    EvaluationV5Protocol,
    _cscg_checkpoints_for_task,
    _initial_snapshot,
    _trained_snapshot,
    _resolve_physical_run_ids,
    assert_model_invariants,
    evaluate_frozen_cscg_snapshot,
    evaluation_v5_protocol_from_dict,
    iter_training_matched_batches,
    posterior_marginal_states,
    protocol_type_for_study,
)
from scripts.analysis import evaluate_v5
from scripts.analysis.plot_config_metric_trajectories import load_rows
from scripts.analysis.postprocess_evaluation_v5 import _select_residual_panels
from mercury_runs.algorithms.cscg.prepare import iter_cscg_training_batches
from mercury_runs.application.study_service import resolve_run_configurations
from mercury_runs.domain.models import StudyDefaults


def test_main_study_residual_panel_selection_is_explicit_and_ordered() -> None:
    payload = {
        "panels": {
            "L13/Oracle capacity": {"value": "oracle"},
            "L13/MERCURY": {"value": "mercury"},
            "L13/Standard CSCG": {"value": "cscg"},
            "L13/TMGWR": {"value": "tmgwr"},
            "L21/Oracle capacity": {"value": "oracle"},
            "L21/MERCURY": {"value": "mercury"},
            "L21/Standard CSCG": {"value": "cscg"},
            "L21/TMGWR": {"value": "tmgwr"},
        }
    }

    selected = _select_residual_panels(
        payload,
        methods=("MERCURY", "TMGWR", "Standard CSCG"),
        scenario="one pass",
        output_name="one-pass.png",
    )

    assert list(selected["panels"]) == [
        "L13/MERCURY",
        "L13/TMGWR",
        "L13/Standard CSCG",
        "L21/MERCURY",
        "L21/TMGWR",
        "L21/Standard CSCG",
    ]
    assert selected["scenario"] == "one pass"
    assert selected["output_name"] == "one-pass.png"


def test_resolve_physical_run_prefers_completed_retry_over_failed_attempt(
    tmp_path: Path,
) -> None:
    identity = {
        "requested_run_id": "requested-run",
        "method": "cscg",
        "level": 22,
        "seed": 11,
        "rand_prob": 0.1,
        "cscg_batch_size": 40,
        "cscg_structural_preset": "vanilla_cscg",
    }
    for run_id, status in (("failed-run", "failed"), ("retry-run", "completed")):
        run_root = tmp_path / run_id
        run_root.mkdir()
        (run_root / "run_config.json").write_text(
            json.dumps({"cli_args": identity}), encoding="utf-8"
        )
        (run_root / "run_status.json").write_text(
            json.dumps({"status": status}), encoding="utf-8"
        )

    resolved = _resolve_physical_run_ids(source_study=tmp_path, configs=[identity])

    assert resolved[0]["_resolved_run_id"] == "retry-run"


@pytest.mark.parametrize("batch_size", [2, 3, 5])
def test_array_batches_match_training_utility(tmp_path: Path, batch_size: int) -> None:
    length = 11
    observations = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    actions = np.eye(4, dtype=np.float32)[np.arange(length) % 4]
    frame = pl.DataFrame(
        {
            "observation_0": observations[:, 0],
            "observation_1": observations[:, 1],
            "action_0": actions[:, 0],
            "action_1": actions[:, 1],
            "action_2": actions[:, 2],
            "action_3": actions[:, 3],
            "collision": np.zeros(length, dtype=bool),
        }
    )
    path = tmp_path / "walk.parquet"
    frame.write_parquet(path)

    expected = list(
        iter_cscg_training_batches(
            parquet_path=path,
            batch_size=batch_size,
            include_prev=True,
            valid_trajectories_only=False,
        )
    )
    actual = list(
        iter_training_matched_batches(observations, actions, batch_size=batch_size)
    )

    assert len(actual) == len(expected)
    for index, (batch, reference) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(batch.observations, reference["observations"])
        np.testing.assert_array_equal(batch.actions, reference["actions"])
        assert batch.overlap == (0 if index == 0 else 1)
        assert batch.new_samples == int(reference["filtered_steps"])
    assert sum(batch.new_samples for batch in actual) == length


def test_posterior_marginal_argmax_uses_global_clone_offsets_and_smallest_tie() -> None:
    n_clones = np.asarray([2, 3], dtype=np.int64)
    observations = np.asarray([1, 0, 1], dtype=np.int64)
    # Packed slices: [3 clones], [2 clones], [3 clones].
    forward = np.asarray([1, 4, 4, 2, 1, 1, 2, 6], dtype=np.float64)
    backward = np.asarray([1, 1, 1, 2, 1, 1, 2, 1], dtype=np.float64)

    states = posterior_marginal_states(
        n_clones=n_clones,
        observations=observations,
        forward_messages=forward,
        backward_messages=backward,
    )

    # Observation 1 starts at global clone 2. First slice ties clones 1/2 -> clone 1.
    assert states.tolist() == [3, 0, 4]


class _FrozenModel:
    def __init__(self, *, n_clones, x, a, pseudocount, dtype, seed) -> None:
        del x, seed
        self.n_clones = np.asarray(n_clones, dtype=np.int64)
        self.pseudocount = float(pseudocount)
        self.dtype = dtype
        n_states = int(self.n_clones.sum())
        n_actions = int(np.asarray(a).max()) + 1
        self.C = np.zeros((n_actions, n_states, n_states), dtype=np.float32)
        self.T = np.ones_like(self.C) / n_states
        self.Pi_x = np.ones(n_states, dtype=np.float64) / n_states
        self.Pi_a = np.ones(n_actions, dtype=np.float64) / n_actions

    def update_T(self) -> None:  # pragma: no cover - must never be called
        raise AssertionError("update_T must not be called during evaluation-v5")


class _FrozenModule:
    CHMM = _FrozenModel
    initial_distributions: list[np.ndarray] = []
    forward_actions: list[np.ndarray] = []
    backward_actions: list[np.ndarray] = []

    @staticmethod
    def forward(T_tr, Pi_x, n_clones, x, a, store_messages=False):
        del T_tr
        assert store_messages is True
        _FrozenModule.initial_distributions.append(np.asarray(Pi_x).copy())
        _FrozenModule.forward_actions.append(np.asarray(a).copy())
        lengths = np.asarray(n_clones)[np.asarray(x)]
        messages = np.concatenate(
            [np.arange(1, int(length) + 1, dtype=np.float32) for length in lengths]
        )
        return -np.ones(len(x), dtype=np.float32), messages

    @staticmethod
    def backward(T, n_clones, x, a):
        del T
        _FrozenModule.backward_actions.append(np.asarray(a).copy())
        return np.ones(int(np.asarray(n_clones)[np.asarray(x)].sum()), dtype=np.float32)

    @staticmethod
    def updateC(*args, **kwargs):  # pragma: no cover - must never be called
        del args, kwargs
        raise AssertionError("updateC must not be called during evaluation-v5")


def test_frozen_batch_eval_resets_pi_excludes_overlap_and_never_updates() -> None:
    _FrozenModule.initial_distributions = []
    _FrozenModule.forward_actions = []
    _FrozenModule.backward_actions = []
    snapshot = {
        "T": np.ones((2, 4, 4), dtype=np.float32) / 4.0,
        "n_clones": np.asarray([2, 2], dtype=np.int64),
        "obs_uniques": np.asarray([10, 20], dtype=np.int64),
        "action_uniques": np.asarray([0, 1], dtype=np.int64),
    }
    walk_data = {
        "n_walks": 1,
        "filtered_obs_per_walk": [np.asarray([10, 20, 10, 20, 10], dtype=np.int64)],
        "filtered_actions_per_walk": [np.asarray([0, 1, 0, 1, 0], dtype=np.int64)],
        "gt_exact_per_walk": [np.asarray([0, 1, 0, 1, 0], dtype=np.int64)],
        "gt_action_per_walk": [np.asarray([0, 1, 0, 1, 0], dtype=np.int64)],
    }

    decoded, model_fit, inference = evaluate_frozen_cscg_snapshot(
        snapshot=snapshot,
        walk_data=walk_data,
        batch_size=2,
        pseudocount=0.002,
        chmm_mod=_FrozenModule,
    )

    assert len(decoded["decoded_walks"]) == 1
    assert decoded["decoded_walks"][0].shape == (5,)
    assert model_fit["scored_symbols"] == 5
    assert model_fit["negative_log2_likelihood_sum"] == pytest.approx(5.0)
    assert model_fit["overlap_terms_excluded"] == 2
    assert inference["decoded_walks"] == 1
    assert inference["parameter_updates"] is False
    assert len(_FrozenModule.initial_distributions) == 3
    for value in _FrozenModule.initial_distributions[1:]:
        np.testing.assert_array_equal(value, _FrozenModule.initial_distributions[0])
    expected_actions = [[1, 1], [0, 1, 1], [0, 0]]
    assert [values.tolist() for values in _FrozenModule.forward_actions] == expected_actions
    assert [values.tolist() for values in _FrozenModule.backward_actions] == expected_actions


def test_invariant_check_detects_mutation() -> None:
    model = _FrozenModel(
        n_clones=np.asarray([2], dtype=np.int64),
        x=np.asarray([0, 0], dtype=np.int64),
        a=np.asarray([0, 0], dtype=np.int64),
        pseudocount=0.002,
        dtype=np.float32,
        seed=0,
    )
    before = {"T": model.T.copy(), "Pi_x": model.Pi_x.copy()}
    model.T[0, 0, 0] = 99.0
    with pytest.raises(RuntimeError, match="model.T"):
        assert_model_invariants(model, before)


def test_evaluation_v5_protocol_roundtrip_and_identity() -> None:
    protocol = EvaluationV5Protocol(
        sensor="cardinal distance",
        sensor_range=1,
        rand_prob=0.1,
        number_of_walks=100,
        walk_length=10_000,
    )
    restored = evaluation_v5_protocol_from_dict(protocol.to_dict())
    assert restored.protocol_id() == protocol.protocol_id()
    assert protocol.protocol_id().startswith("evaluation-v5-nonself-v2-")
    assert protocol.parameter_updates is False
    assert protocol.inference_state_reset_policy == "reset_per_training_batch"


def test_matched_budget_protocol_preserves_legacy_identity_and_rejects_mixed_studies() -> None:
    old = EvaluationV5Protocol(sensor="cardinal distance", sensor_range=1, rand_prob=0.1)
    matched = EvaluationV5MatchedBudgetProtocol(sensor="cardinal distance", sensor_range=1, rand_prob=0.1)
    assert old.protocol_id() == "evaluation-v5-nonself-v2-835996d8e9ad60d3"
    assert matched.protocol_id() != old.protocol_id()
    assert isinstance(evaluation_v5_protocol_from_dict(matched.to_dict()), EvaluationV5MatchedBudgetProtocol)

    hybrid = {"method": "mercury_cscg_hybrid", "num_steps": 2_000, "cscg_n_iter": 1_998_000}
    tmgwr = {"method": "tmgwr_cscg_hybrid", "num_steps": 2_000, "cscg_n_iter": 1_998_000}
    assert protocol_type_for_study([hybrid, tmgwr, {"method": "cscg"}]) is EvaluationV5MatchedBudgetProtocol
    assert protocol_type_for_study([{**hybrid, "cscg_n_iter": 2_000_000}]) is EvaluationV5Protocol
    with pytest.raises(ValueError, match="cannot mix"):
        protocol_type_for_study([hybrid, {**tmgwr, "cscg_n_iter": 2_000_000}])


@pytest.mark.parametrize("method", ["mercury_cscg_hybrid", "tmgwr_cscg_hybrid"])
def test_matched_hybrid_evaluates_only_saved_phase_snapshots(method: str, tmp_path: Path) -> None:
    task = EvaluationTask(
        study_name="main_study",
        source_run=tmp_path,
        output_run=tmp_path,
        config={"method": method, "num_steps": 2_000, "cscg_n_iter": 1_998_000},
        checkpoints=(0, 2_000, 4_000, 2_000_000),
    )
    matched = EvaluationV5MatchedBudgetProtocol(sensor="cardinal distance", sensor_range=1, rand_prob=0.1)
    legacy = EvaluationV5Protocol(sensor="cardinal distance", sensor_range=1, rand_prob=0.1)
    assert _cscg_checkpoints_for_task(task, matched) == (0, 2_000, 2_000_000)
    assert _cscg_checkpoints_for_task(task, legacy) == task.checkpoints


@pytest.mark.parametrize("method", ["mercury_cscg_hybrid", "tmgwr_cscg_hybrid"])
def test_matched_hybrid_initialisation_uses_saved_two_thousand_step_reference(
    method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = EvaluationTask(
        study_name="main_study",
        source_run=tmp_path / "source",
        output_run=tmp_path / "shadow",
        config={
            "method": method,
            "num_steps": 2_000,
            "cscg_n_iter": 1_998_000,
            "paper_precision_eval_checkpoints": [2_000, 1_998_000],
        },
        checkpoints=(0,),
        reference_step=4_000,
    )
    loaded_paths: list[Path] = []
    reference = {
        "T": np.asarray([[[1.0]]]),
        "n_clones": np.asarray([1]),
        "meta": {"step": 2_000, "observed_samples": 2_000},
    }
    monkeypatch.setattr(
        evaluate_v5,
        "load_cscg_paper_precision_snapshot",
        lambda path: loaded_paths.append(path) or reference,
    )
    monkeypatch.setattr(
        evaluate_v5,
        "reconstruct_initial_counts",
        lambda **kwargs: (np.asarray([[[1.0]]]), {"source_graph_path": "source-step-2000"}),
    )
    initial, provenance = _initial_snapshot(task)
    assert loaded_paths == [
        task.source_run / "snapshots" / method / "paper_precision" / "step=2000"
    ]
    assert initial["meta"]["step"] == 0
    assert initial["meta"]["observed_samples"] == 0
    assert provenance["reference_snapshot_step"] == 2_000
    legacy_task = EvaluationTask(
        study_name=task.study_name,
        source_run=task.source_run,
        output_run=task.output_run,
        config={**task.config, "cscg_n_iter": 2_000_000, "paper_precision_eval_checkpoints": [2_000, 4_000, 2_000_000]},
        checkpoints=task.checkpoints,
        reference_step=4_000,
    )
    loaded_paths.clear()
    _, legacy_provenance = _initial_snapshot(legacy_task)
    assert loaded_paths == [
        task.source_run / "snapshots" / method / "paper_precision" / "step=4000"
    ]
    assert legacy_provenance["reference_snapshot_step"] == 4_000


@pytest.mark.parametrize("method", ["mercury_cscg_hybrid", "tmgwr_cscg_hybrid"])
def test_matched_budget_final_snapshot_uses_phase_step_and_checks_metadata(
    tmp_path: Path, method: str
) -> None:
    run_root = tmp_path / "run"
    snapshot_dir = run_root / "snapshots" / method / "paper_precision" / "step=1998000"
    snapshot_dir.mkdir(parents=True)
    np.savez_compressed(
        snapshot_dir / "model.npz",
        T=np.asarray([[[0.5, 0.5], [0.5, 0.5]]], dtype=np.float32),
        n_clones=np.asarray([1, 1], dtype=np.int64),
        obs_uniques=np.asarray([0, 1]),
        action_uniques=np.asarray([0]),
    )
    (snapshot_dir / "snapshot.json").write_text(
        json.dumps({"step": 1_998_000, "observed_samples": 1_998_000}), encoding="utf-8"
    )
    task = EvaluationTask(
        study_name="main_study",
        source_run=run_root,
        output_run=tmp_path / "shadow",
        config={"method": method, "num_steps": 2_000, "cscg_n_iter": 1_998_000},
        checkpoints=(2_000_000,),
    )
    snapshot, provenance = _trained_snapshot(task, 2_000_000)
    assert snapshot["meta"]["observed_samples"] == 1_998_000
    assert provenance["snapshot_phase_step"] == 1_998_000

    (snapshot_dir / "snapshot.json").write_text(
        json.dumps({"step": 1_998_000, "observed_samples": 1_997_960}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not contain"):
        _trained_snapshot(task, 2_000_000)


def test_matched_budget_record_keeps_nominal_final_step_and_actual_sample_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = EvaluationV5MatchedBudgetProtocol(sensor="cardinal distance", sensor_range=1)
    task = EvaluationTask(
        study_name="main_study",
        source_run=tmp_path / "source",
        output_run=tmp_path / "shadow",
        config={"method": "mercury_cscg_hybrid", "level": 13, "num_steps": 2_000},
        checkpoints=(2_000_000,),
    )
    snapshot = {
        "meta": {"observed_samples": 1_998_000},
        "T": np.asarray([[[1.0]]]),
        "n_clones": np.asarray([1]),
    }
    monkeypatch.setitem(evaluate_v5._WORKER_CONTEXT, "walk_data", {})
    monkeypatch.setitem(evaluate_v5._WORKER_CONTEXT, "gt", {})
    monkeypatch.setattr(evaluate_v5, "_trained_snapshot", lambda *_: (snapshot, {}))
    monkeypatch.setattr(
        evaluate_v5,
        "_initial_snapshot",
        lambda *_: (
            {**snapshot, "meta": {"observed_samples": 0}},
            {"source_graph_path": "bundles/snapshots/mercury/paper_precision/step=2000/latent_graph.npz"},
        ),
    )
    monkeypatch.setattr(
        evaluate_v5,
        "evaluate_frozen_cscg_snapshot",
        lambda **_: ({"decoded_walks": [0]}, {}, {}),
    )
    monkeypatch.setattr(
        evaluate_v5,
        "_evaluate_metrics",
        lambda **kwargs: {
            "snapshot_step": kwargs["step"],
            "observed_samples": kwargs["observed_samples"],
        },
    )
    monkeypatch.setattr(evaluate_v5, "save_evaluation_result", lambda **_: None)

    record = evaluate_v5._evaluate_cscg_step(task=task, protocol=protocol, step=2_000_000)
    assert record["snapshot_step"] == 2_000_000
    assert record["observed_samples"] == 1_998_000
    assert record["training_budget"] == {
        "source_samples": 2_000,
        "cscg_phase_samples": 1_998_000,
        "cumulative_samples": 2_000_000,
    }
    transferred = evaluate_v5._evaluate_cscg_step(task=task, protocol=protocol, step=0)
    assert transferred["snapshot_step"] == 0
    assert transferred["observed_samples"] == 0
    assert transferred["training_budget"] == {
        "source_samples": 2_000,
        "cscg_phase_samples": 0,
        "cumulative_samples": 2_000,
    }
    assert "mercury/paper_precision/step=2000" in transferred["initialization"]["source_graph_path"]


def test_main_study_grid_overrides_hybrid_checkpoints_without_expanding_arms() -> None:
    from argparse import Namespace

    rows = resolve_run_configurations(
        Namespace(study_config=str(PROJECT_ROOT / "configs/studies/default_matrix_smoke_30_seed.json")),
        StudyDefaults(study_root="results/current/studies", study_name="main_study"),
    )
    assert len(rows) == 720
    for row in rows:
        if row["method"] == "cscg":
            assert row["cscg_n_iter"] == 2_000_000
            assert row["paper_precision_eval_checkpoints"] == [2_000, 4_000, 2_000_000]
        else:
            assert row["method"] in {"mercury_cscg_hybrid", "tmgwr_cscg_hybrid"}
            assert row["num_steps"] + row["cscg_n_iter"] == 2_000_000
            assert row["paper_precision_eval_checkpoints"] == [2_000, 1_998_000]


def test_matched_budget_record_plots_at_two_million_total_samples(tmp_path: Path) -> None:
    study_root = tmp_path / "study"
    run_root = study_root / "hybrid-run"
    config = {
        "requested_run_id": "hybrid-run",
        "method": "mercury_cscg_hybrid",
        "level": 13,
        "seed": 0,
        "rand_prob": 0.1,
        "cscg_batch_size": 40,
        "num_steps": 2_000,
    }
    run_root.mkdir(parents=True)
    (study_root / "study_manifest.jsonl").write_text(
        json.dumps({"config": config}) + "\n", encoding="utf-8"
    )
    (run_root / "run_status.json").write_text("{}", encoding="utf-8")
    record_dir = run_root / "evaluations" / "matched-protocol" / "snapshots" / "mercury_cscg_hybrid"
    record_dir.mkdir(parents=True)
    source_record_dir = record_dir.parent / "mercury_cscg_hybrid_mercury"
    source_record_dir.mkdir()
    (source_record_dir / "step=00002000.json").write_text(
        json.dumps({"snapshot_step": 2_000, "state_representation": {"latent_precision": 0.25}}),
        encoding="utf-8",
    )
    (record_dir / "step=02000000.json").write_text(
        json.dumps(
            {
                "snapshot_step": 2_000_000,
                "observed_samples": 1_998_000,
                "state_representation": {"latent_precision": 0.5},
                "training_budget": {
                    "source_samples": 2_000,
                    "cscg_phase_samples": 1_998_000,
                    "cumulative_samples": 2_000_000,
                },
            }
        ),
        encoding="utf-8",
    )
    for phase_step in (0, 2_000):
        (record_dir / f"step={phase_step:08d}.json").write_text(
            json.dumps(
                {
                    "snapshot_step": phase_step,
                    "observed_samples": phase_step,
                    "state_representation": {"latent_precision": 0.75 if phase_step == 0 else 0.5},
                }
            ),
            encoding="utf-8",
        )

    rows = load_rows(study_root, "matched-protocol", 40, {13})
    by_step = rows.set_index("step")
    assert set(by_step.index) == {0, 2_000, 2_000_000}
    assert by_step.loc[0, "value"] == pytest.approx(0.75)
    assert by_step.loc[0, "plot_step"] == 2_000
    assert by_step.loc[0, "comparison_step"] == 0
    assert by_step.loc[2_000, "comparison_step"] == 4_000
    assert by_step.loc[2_000_000, "observed_samples"] == 1_998_000
    assert by_step.loc[2_000_000, "plot_step"] == 2_000_000
    assert by_step.loc[2_000_000, "comparison_step"] == 2_000_000
