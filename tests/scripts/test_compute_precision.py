from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from scripts.analysis.compute_precision import _plot_paper_precision, compute_metrics_for_states


def _write_states(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "ground_truth_bmu": [0, 1, 0, 1],
            "cscg_state_id": [0, 1, 0, 1],
            "latent_node_count": [2, 2, 2, 2],
        }
    ).write_parquet(path)


def test_compute_metrics_for_states_uses_paper_precision_json_for_png(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    out_dir = tmp_path / "metrics"
    _write_states(states_path)
    paper_precision_path = out_dir / "run-1_paper_precision.json"
    paper_precision_path.parent.mkdir(parents=True, exist_ok=True)
    paper_precision_path.write_text(
        json.dumps(
            {
                "method": "cscg",
                "protocol": {"mode": "per_iteration", "eval_interval": 2, "num_walks": 10, "walk_length": 2000},
                "metrics": {"latent_precision": 0.8},
                "history": [
                    {"stage": "em", "iteration": 2, "latent_precision": 0.4},
                    {"stage": "em", "iteration": 4, "latent_precision": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    compute_metrics_for_states(
        states_path=states_path,
        out_dir=out_dir,
        run_id="run-1",
        window_length=2,
        precision_columns={"latent_precision": "cscg_state_id"},
        method_metadata={"method": "cscg", "level": 0, "latent_sensor": "cartesian", "latent_sensor_range": None},
    )

    metadata = json.loads((out_dir / "run-1_precision.metadata.json").read_text(encoding="utf-8"))
    assert (out_dir / "run-1_precision.png").exists()
    assert metadata["outputs"]["plot_png"] == str(out_dir / "run-1_precision.png")


def test_compute_metrics_for_states_skips_png_without_paper_precision_json(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    out_dir = tmp_path / "metrics"
    _write_states(states_path)

    compute_metrics_for_states(
        states_path=states_path,
        out_dir=out_dir,
        run_id="run-2",
        window_length=2,
        precision_columns={"latent_precision": "cscg_state_id"},
        method_metadata={"method": "cscg", "level": 0, "latent_sensor": "cartesian", "latent_sensor_range": None},
    )

    metadata = json.loads((out_dir / "run-2_precision.metadata.json").read_text(encoding="utf-8"))
    assert not (out_dir / "run-2_precision.png").exists()
    assert metadata["outputs"]["plot_png"] is None


def test_compute_metrics_for_states_plots_multi_capacity_paper_precision(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    out_dir = tmp_path / "metrics"
    _write_states(states_path)
    paper_precision_path = out_dir / "run-3_paper_precision.json"
    paper_precision_path.parent.mkdir(parents=True, exist_ok=True)
    paper_precision_path.write_text(
        json.dumps(
            {
                "method": "pocml",
                "protocol": {"mode": "per_iteration", "resolved_capacities": [3, 5]},
                "metrics_by_capacity": {
                    "3": {"capacity_precision": 0.4},
                    "5": {"capacity_precision": 0.6},
                },
                "history_by_capacity": {
                    "3": [{"step": 1, "capacity_precision": 0.3}, {"step": 2, "capacity_precision": 0.4}],
                    "5": [{"step": 1, "capacity_precision": 0.5}, {"step": 2, "capacity_precision": 0.6}],
                },
            }
        ),
        encoding="utf-8",
    )

    compute_metrics_for_states(
        states_path=states_path,
        out_dir=out_dir,
        run_id="run-3",
        window_length=2,
        precision_columns={"latent_precision": "cscg_state_id"},
        method_metadata={"method": "pocml", "level": 0, "latent_sensor": "cartesian", "latent_sensor_range": None},
    )

    assert (out_dir / "run-3_precision.png").exists()


def test_compute_metrics_for_states_plots_pocml_history_as_one_line_per_capacity(tmp_path: Path) -> None:
    states_path = tmp_path / "states.parquet"
    out_dir = tmp_path / "metrics"
    _write_states(states_path)
    paper_precision_path = out_dir / "run-4_paper_precision.json"
    paper_precision_path.parent.mkdir(parents=True, exist_ok=True)
    paper_precision_path.write_text(
        json.dumps(
            {
                "method": "pocml",
                "protocol": {"mode": "per_iteration", "resolved_capacities": [5, 10]},
                "metrics_by_capacity": {
                    "5": {"capacity_precision": 0.4},
                    "10": {"capacity_precision": 0.6},
                },
                "history_by_capacity": {
                    "5": [{"step": 1, "capacity_precision": 0.3}, {"step": 2, "capacity_precision": 0.4}],
                    "10": [{"step": 1, "capacity_precision": 0.5}, {"step": 2, "capacity_precision": 0.6}],
                },
            }
        ),
        encoding="utf-8",
    )

    compute_metrics_for_states(
        states_path=states_path,
        out_dir=out_dir,
        run_id="run-4",
        window_length=2,
        precision_columns={"latent_precision": "cscg_state_id"},
        method_metadata={"method": "pocml", "level": 0, "latent_sensor": "cartesian", "latent_sensor_range": None},
    )

    metadata = json.loads((out_dir / "run-4_precision.metadata.json").read_text(encoding="utf-8"))
    assert (out_dir / "run-4_precision.png").exists()
    assert metadata["outputs"]["plot_png"] == str(out_dir / "run-4_precision.png")


def test_plot_paper_precision_prefers_observed_samples_axis(monkeypatch, tmp_path: Path) -> None:
    paper_precision_path = tmp_path / "run-5_paper_precision.json"
    plot_path = tmp_path / "run-5_precision.png"
    paper_precision_path.write_text(
        json.dumps(
            {
                "method": "cscg",
                "protocol": {"x_axis_unit": "observed_samples"},
                "metrics": {"latent_precision": 0.8},
                "history": [
                    {"step": 2, "observed_samples": 100, "latent_precision": 0.4},
                    {"step": 4, "observed_samples": 200, "latent_precision": 0.6},
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class _FakeAxis:
        def plot(self, x, y, marker=None, label=None):
            captured["x"] = list(x)
            captured["y"] = list(y)
            captured["label"] = label

        def set_xlabel(self, value):
            captured["xlabel"] = value

        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_title(self, value):
            captured["title"] = value

        def legend(self):
            captured["legend"] = True

        def bar(self, *args, **kwargs):
            raise AssertionError("unexpected bar plot")

        def text(self, *args, **kwargs):
            raise AssertionError("unexpected text")

    class _FakeFigure:
        def tight_layout(self):
            captured["tight_layout"] = True

        def savefig(self, path, dpi=200):
            Path(path).write_text("png", encoding="utf-8")
            captured["saved"] = str(path)

    monkeypatch.setattr("matplotlib.use", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.subplots", lambda **kwargs: (_FakeFigure(), _FakeAxis()))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda figure: None)

    plot_error = _plot_paper_precision(
        paper_precision_path=paper_precision_path,
        plot_path=plot_path,
        method_metadata={"method": "cscg", "level": 0},
    )

    assert plot_error is None
    assert captured["x"] == [100, 200]
    assert captured["xlabel"] == "Observed Samples"
    assert plot_path.exists()


def test_plot_paper_precision_prefers_edge_f1_over_edge_precision(monkeypatch, tmp_path: Path) -> None:
    paper_precision_path = tmp_path / "run-6_paper_precision.json"
    plot_path = tmp_path / "run-6_precision.png"
    paper_precision_path.write_text(
        json.dumps(
            {
                "method": "mercury",
                "metrics": {
                    "sensory_precision": 0.7,
                    "latent_precision": 0.6,
                    "sensory_edge_precision": 1.0,
                    "sensory_edge_f1": 0.5,
                    "latent_edge_f1": 0.4,
                    "sensory_mean_total_variation": 0.3,
                    "latent_mean_total_variation": 0.2,
                    "sensory_action_conditioned_edge_precision": 0.9,
                    "sensory_action_conditioned_edge_f1": 0.45,
                    "latent_action_conditioned_edge_f1": 0.35,
                    "sensory_action_conditioned_mean_total_variation": 0.25,
                    "latent_action_conditioned_mean_total_variation": 0.15,
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class _FakeAxis:
        def plot(self, *args, **kwargs):
            raise AssertionError("unexpected line plot")

        def set_xlabel(self, value):
            captured["xlabel"] = value

        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_title(self, value):
            captured["title"] = value

        def legend(self):
            captured["legend"] = True

        def set_ylim(self, low, high):
            captured["ylim"] = (low, high)

        def bar(self, keys, values, color=None):
            captured["keys"] = list(keys)
            captured["values"] = list(values)

        def text(self, *args, **kwargs):
            captured["text"] = True

    class _FakeFigure:
        def tight_layout(self):
            captured["tight_layout"] = True

        def savefig(self, path, dpi=200):
            Path(path).write_text("png", encoding="utf-8")
            captured["saved"] = str(path)

    monkeypatch.setattr("matplotlib.use", lambda *args, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.subplots", lambda **kwargs: (_FakeFigure(), _FakeAxis()))
    monkeypatch.setattr("matplotlib.pyplot.close", lambda figure: None)

    plot_error = _plot_paper_precision(
        paper_precision_path=paper_precision_path,
        plot_path=plot_path,
        method_metadata={"method": "mercury", "level": 13},
    )

    assert plot_error is None
    assert captured["keys"] == [
        "Precision | Sensory",
        "Precision | Latent",
        "Edge F1 | Sensory",
        "Edge F1 | Latent",
        "Action-Conditioned Edge F1 | Sensory",
        "Action-Conditioned Edge F1 | Latent",
        "Mean Total Variation | Sensory",
        "Mean Total Variation | Latent",
        "Action-Conditioned Mean Total Variation | Sensory",
        "Action-Conditioned Mean Total Variation | Latent",
    ]
    assert captured["values"] == [0.7, 0.6, 0.5, 0.4, 0.45, 0.35, 0.3, 0.2, 0.25, 0.15]
    assert plot_path.exists()
