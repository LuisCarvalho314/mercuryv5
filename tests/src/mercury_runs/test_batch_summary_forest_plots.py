from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from mercury_runs.analysis.plots import batch_summary_forest as plots
from mercury_runs.analysis.plots import publication_style as pub


def test_transposed_method_legend_is_balanced_and_uses_standard_cscg_label() -> None:
    assert plots._transposed_method_legend_columns(6) == 3
    assert plots._transposed_method_legend_columns(7) == 4
    assert plots._transposed_method_legend_columns(10) == 5
    assert plots.SERIES_LABELS["vanilla_cscg"] == "Standard CSCG"
    assert plots.SERIES_COLORS["tmgwr_hybrid"] != plots.SERIES_COLORS["oracle_capacity"]
    assert pub.method_label("hybrid") == "MERCURY initialised"
    assert pub.method_label("tmgwr_hybrid") == "TMGWR initialised"
    assert pub.method_color("tmgwr_hybrid", comparison="checkpoint_2000") == "#E69F00"
    assert pub.method_color("tmgwr_hybrid", comparison="final") == "#7B61A8"
    assert pub.metric_label("latent_precision") == "Latent purity"
    assert pub.metric_label("latent_recall") == "Latent concentration"


def _frame(*, comparisons: tuple[str, ...] = ("checkpoint_2000",)) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        for level in plots.CANONICAL_LEVEL_ORDER:
            for series_name, base_value in (("mercury", 0.82), ("tmgwr_hybrid", 0.72)):
                for rand_prob in (0.0, 0.3):
                    for seed in range(3):
                        row: dict[str, object] = {
                            "comparison": comparison,
                            "level": level,
                            "seed": seed,
                            "rand_prob": rand_prob,
                            "cscg_batch_size": 40,
                            "plot_series": series_name,
                            plots.CSCG_NLL_METRIC: 1.1 + 0.04 * seed + 0.1 * rand_prob,
                        }
                        row.update(
                            {
                                metric: base_value + 0.01 * seed - 0.02 * rand_prob
                                for metric, _ in plots.METRICS
                            }
                        )
                        rows.append(row)
    return pl.DataFrame(rows)


def test_transposed_profile_uses_metric_rows_method_legend_and_nll_axis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)

    output_path = plots.plot_metric_profile_transposed_compact(
        _frame(),
        series_order=["mercury", "tmgwr_hybrid"],
        title="Synthetic transposed profile",
        output_name="focused_transposed.png",
        plot_dir=tmp_path,
        metric_keys=plots.PROFILE_FOCUSED_METRICS,
        include_nll_axis=True,
    )
    figure = plots.plt.gcf()
    try:
        assert output_path.exists()
        assert output_path.with_suffix(".pdf").exists()
        assert tuple(round(value, 2) for value in figure.get_size_inches()) == (3.68, 4.75)
        assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == [
            r"$P_Z$",
            r"$C_Z$",
            r"$S_{\mathrm{TV}}^A$",
            r"$S_{\mathrm{TV}}^{A,\neg\mathrm{self}}$",
            "CSCG eval NLL",
        ]
        assert [label.get_text() for label in figure.axes[2].get_yticklabels()] == [
            r"$P_Z$",
            r"$C_Z$",
            r"$S_{\mathrm{TV}}^A$",
            r"$S_{\mathrm{TV}}^{A,\neg\mathrm{self}}$",
            "CSCG eval NLL",
        ]
        assert not figure.axes[1].get_yticklabels()
        assert not figure.axes[3].get_yticklabels()
        assert figure.axes[0].get_xticks().tolist() == [0.0, 0.5, 1.0]
        assert all(
            any(collection.__class__.__name__ == "PathCollection" for collection in axis.collections)
            for axis in figure.axes[:4]
        )
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "MERCURY",
            "TMGWR",
        ]
        assert figure._suptitle is None
        assert any(axis.get_xlabel() == "NLL (bits/symbol)" for axis in figure.axes)
    finally:
        original_close(figure)


def test_transposed_profile_by_rand_has_separate_persistence_marker_legend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_close = plots.plt.close
    monkeypatch.setattr(plots.plt, "close", lambda figure: None)

    output_path = plots.plot_metric_profile_by_rand_transposed(
        _frame(),
        series_order=["mercury", "tmgwr_hybrid"],
        title="Synthetic persistence profile",
        output_name="focused_rand_rows_transposed.png",
        plot_dir=tmp_path,
        metric_keys=plots.PROFILE_FOCUSED_METRICS,
    )
    figure = plots.plt.gcf()
    try:
        assert output_path.exists()
        assert not output_path.with_suffix(".pdf").exists()
        assert len(figure.legends) == 2
        assert [text.get_text() for text in figure.legends[1].get_texts()] == [
            "Persistence 0.0",
            "Persistence 0.3",
        ]
        markers = [handle.get_marker() for handle in figure.legends[1].legend_handles]
        assert markers == ["o", "s"]
    finally:
        original_close(figure)


def test_main_emits_transposed_counterpart_for_every_focused_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = _frame(comparisons=("final", "checkpoint_4000", "checkpoint_2000"))
    transposed_names: list[str] = []
    transposed_titles: dict[str, str] = {}
    compact_names: list[str] = []

    def fake_plot(*args, output_name: str, plot_dir: Path, **kwargs) -> Path:
        return plot_dir / output_name

    def fake_transposed_plot(*args, output_name: str, plot_dir: Path, **kwargs) -> Path:
        transposed_names.append(output_name)
        transposed_titles[output_name] = str(kwargs["title"])
        return plot_dir / output_name

    def fake_compact_plot(*args, output_name: str, plot_dir: Path, **kwargs) -> Path:
        compact_names.append(output_name)
        return plot_dir / output_name

    monkeypatch.setattr(plots, "_build_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(plots, "plot_forest", fake_plot)
    monkeypatch.setattr(plots, "plot_metric_matrix", fake_plot)
    monkeypatch.setattr(plots, "plot_metric_profile", fake_plot)
    monkeypatch.setattr(plots, "plot_metric_profile_by_rand", fake_plot)
    monkeypatch.setattr(plots, "plot_metric_profile_transposed", fake_transposed_plot)
    monkeypatch.setattr(plots, "plot_metric_profile_transposed_compact", fake_compact_plot)
    monkeypatch.setattr(plots, "plot_metric_profile_by_rand_transposed", fake_transposed_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_summary_forest",
            "--study_root",
            str(tmp_path / "study"),
            "--protocol_id",
            "canonical-v4-test",
            "--output_dir",
            str(tmp_path / "plots"),
        ],
    )

    plots.main()

    assert len(transposed_names) == 30
    assert len(set(transposed_names)) == 30
    assert compact_names == [
        "batch_40_final_selected_metric_profile_focused_transposed_compact.png",
        "batch_40_checkpoint_4000_selected_metric_profile_focused_transposed_compact.png",
        "batch_40_checkpoint_2000_selected_metric_profile_focused_transposed_compact.png",
    ]
    assert all(name.endswith("_transposed.png") for name in transposed_names)
    assert "batch_40_checkpoint_2000_selected_metric_profile_focused_transposed.png" in transposed_names
    assert "batch_40_final_selected_metric_profile_focused_with_nll_rand_0p3_transposed.png" in transposed_names
    assert "batch_40_checkpoint_4000_selected_metric_profile_focused_with_nll_rand_rows_transposed.png" in transposed_names
    assert transposed_titles["batch_40_checkpoint_2000_selected_metric_profile_focused_transposed.png"] == "After one training pass"
    assert transposed_titles["batch_40_checkpoint_4000_selected_metric_profile_focused_transposed.png"] == "After two training passes"
    assert transposed_titles["batch_40_final_selected_metric_profile_focused_transposed.png"] == "Final comparison"
