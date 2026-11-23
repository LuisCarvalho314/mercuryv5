from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch

from mercury_runs.infrastructure.paper_precision import compute_alignment
from scripts.analysis.plot_selected_transition_residual_maps import (
    CSCG_PRESETS,
    RAND_PROB,
    SCENARIOS,
    SEEDS,
    _cscg_run,
    _hybrid_run,
    _load_manifest,
)
from scripts.analysis.plot_selected_supernode_transition_residual_maps import (
    _native_learned_for_run,
    _pad_cooccurrence,
)
from scripts.analysis.rerender_selected_transition_residual_maps import (
    DEFAULT_INPUT_DIR,
    LEVEL_LABELS,
    LEVEL_ORDER,
    METHOD_REPLACEMENTS,
    _draw_level_label,
    _display_title,
    _positions_for_level,
)


DEFAULT_FILES = (
    "supernode_transition_residual_batch_40_selected_final_hybrid_vs_cscg.json",
    "supernode_transition_residual_batch_40_selected_hybrid_vs_cscg_checkpoint_4000.json",
    "supernode_transition_residual_batch_40_selected_mercury_vs_cscg_checkpoint_2000.json",
)
CACHE_DIR = DEFAULT_INPUT_DIR / "canonical_cooccurrence"
PANEL_RE = re.compile(r"^L(?P<level>\d+)/(?P<method>.+)$")
PURITY_CMAP = plt.get_cmap("Greens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot WTA co-occurrence alignment evidence projected onto ground-truth nodes."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    matplotlib.use("Agg", force=True)
    for filename in args.files:
        sidecar = Path(args.input_dir) / str(filename)
        source_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        output_path = sidecar.with_name(sidecar.stem.replace("supernode_transition_residual_", "wta_purity_", 1) + ".png")
        json_path = output_path.with_suffix(".json")
        evidence_payload = _build_evidence_payload(source_payload, sidecar=sidecar, output_path=output_path)
        json_path.write_text(json.dumps(evidence_payload, indent=2), encoding="utf-8")
        _save_alignment_evidence_figure(evidence_payload, output_path=output_path)
        print(output_path)


def _build_evidence_payload(payload: dict[str, Any], *, sidecar: Path, output_path: Path) -> dict[str, Any]:
    manifest = _load_manifest()
    scenario = _scenario_for_sidecar(sidecar)
    panels = {
        str(key): _seed_averaged_alignment_evidence(
            manifest=manifest,
            scenario=scenario,
            panel_key=str(key),
            panel=dict(panel),
        )
        for key, panel in dict(payload["panels"]).items()
    }
    return {
        "scenario": _display_title(str(payload.get("scenario", "WTA-assigned co-occurrence purity"))),
        "figure": str(output_path),
        "source": payload.get("figure") or payload.get("output_name"),
        "panels": panels,
        "visual_encoding": {
            "node_fill": "seed-averaged occupancy-weighted purity 1-H(Z_GT|u) of learned states WTA-assigned to each ground-truth state",
            "node_label": "ground-truth state id and mean WTA-assigned purity",
            "node_size": "fixed",
            "node_outline_width": "fixed",
            "ground_truth_edges": "thin light-grey support edges from P_gt",
            "cooccurrence_basis": "raw exact-state co-occurrence matrix; WTA determines which learned-state row contributes to each ground-truth node",
            "entropy_normalization": "each learned row entropy H(Z_GT|u) is normalized by log(number of ground-truth states with nonzero co-occurrence for that learned row)",
            "projection": "learned-row entropy is averaged over rows with phi(u)=i, weighted by learned-row occupancy",
            "seed_count": len(SEEDS),
        },
    }


def _scenario_for_sidecar(sidecar: Path) -> dict[str, Any]:
    transition_name = sidecar.name.replace("supernode_transition_residual_", "transition_residual_", 1)
    for scenario in SCENARIOS:
        if transition_name == str(scenario["output_name"]).replace(".png", ".json"):
            return scenario
    raise ValueError(f"Could not identify scenario for {sidecar}")


def _seed_averaged_alignment_evidence(
    *,
    manifest: dict[tuple[Any, ...], Path],
    scenario: dict[str, Any],
    panel_key: str,
    panel: dict[str, Any],
) -> dict[str, Any]:
    match = PANEL_RE.match(panel_key)
    if match is None:
        raise ValueError(f"Unexpected panel key: {panel_key!r}")
    level = int(match.group("level"))
    method = _display_method(str(match.group("method")))
    target_method = _display_method(str(scenario["target_label"]))
    if method == target_method:
        preset = None
        kind = str(scenario["target_kind"])
    else:
        preset_by_label = {_display_method(label): preset for label, preset in CSCG_PRESETS}
        preset = preset_by_label[method]
        kind = str(scenario["cscg_kind"])
    n_true = int(np.asarray(panel["P_gt"], dtype=np.float64).shape[0])
    entropy_rows: list[np.ndarray] = []
    effective_count_rows: list[np.ndarray] = []
    support_rows: list[np.ndarray] = []
    for seed in SEEDS:
        run_root = (
            _hybrid_run(manifest, level, int(seed), rand_prob=RAND_PROB)
            if preset is None
            else _cscg_run(manifest, level, int(seed), str(preset), rand_prob=RAND_PROB)
        )
        cooccurrence_matrix = _canonical_cooccurrence_matrix(
            run_root=run_root,
            kind=_canonical_kind(kind),
            n_true=n_true,
        )
        entropy, effective_count, support = _wta_assigned_impurity_for_seed(
            cooccurrence_matrix=cooccurrence_matrix,
        )
        entropy_rows.append(entropy)
        effective_count_rows.append(effective_count)
        support_rows.append(support)
    entropy_by_seed = np.stack(entropy_rows, axis=0)
    effective_count_by_seed = np.stack(effective_count_rows, axis=0)
    support_by_seed = np.stack(support_rows, axis=0)
    return {
        "method": str(panel.get("method", "")),
        "level": level,
        "P_gt": panel["P_gt"],
        "mean_wta_assigned_impurity": entropy_by_seed.mean(axis=0).tolist(),
        "std_wta_assigned_impurity": entropy_by_seed.std(axis=0).tolist(),
        "mean_wta_assigned_purity": (1.0 - entropy_by_seed).mean(axis=0).tolist(),
        "std_wta_assigned_purity": (1.0 - entropy_by_seed).std(axis=0).tolist(),
        "mean_wta_assigned_effective_gt_count": effective_count_by_seed.mean(axis=0).tolist(),
        "mean_wta_assigned_learned_state_count": support_by_seed.mean(axis=0).tolist(),
        "seed_count": int(entropy_by_seed.shape[0]),
        "cooccurrence_source": "canonical saved co-occurrence matrices",
    }


def _canonical_kind(kind: str) -> str:
    if str(kind) == "hybrid_final":
        return "hybrid_cscg_checkpoint_4000"
    return str(kind)


def _canonical_cooccurrence_matrix(*, run_root: Path, kind: str, n_true: int) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{Path(run_root).name}_{str(kind)}_ntrue={int(n_true)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        return np.asarray(cached["cooccurrence"], dtype=np.int64)

    learned, cooccurrence, _, _, _ = _native_learned_for_run(
        run_root=Path(run_root),
        kind=str(kind),
        viterbi_processed=False,
        action_conditioned=False,
    )
    n_learned = int(learned.shape[1] if np.asarray(learned).ndim == 3 else np.asarray(learned).shape[0])
    cooccurrence_matrix = _pad_cooccurrence(
        np.asarray(cooccurrence, dtype=np.int64),
        n_learned=n_learned,
        n_true=int(n_true),
    )
    np.savez_compressed(
        cache_path,
        cooccurrence=cooccurrence_matrix,
        run_id=str(Path(run_root).name),
        kind=str(kind),
        n_true=np.asarray(int(n_true), dtype=np.int64),
    )
    return cooccurrence_matrix


def _wta_assigned_impurity_for_seed(
    *,
    cooccurrence_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(cooccurrence_matrix, dtype=np.float64)
    n_true = int(matrix.shape[1])
    alignment = np.asarray(compute_alignment(matrix), dtype=np.int64)
    row_totals = matrix.sum(axis=1).astype(np.float64)
    occupied = row_totals > 0.0
    alignment[~occupied] = -1
    row_entropy = np.zeros(matrix.shape[0], dtype=np.float64)
    row_effective_count = np.zeros(matrix.shape[0], dtype=np.float64)
    for learned_state in np.where(occupied)[0].tolist():
        row = matrix[int(learned_state)]
        active = row > 0.0
        probs = row[active] / float(row_totals[int(learned_state)])
        raw_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        row_effective_count[int(learned_state)] = float(np.exp(raw_entropy))
        if probs.size > 1:
            row_entropy[int(learned_state)] = raw_entropy / np.log(float(probs.size))
    impurity = np.zeros(n_true, dtype=np.float64)
    effective_count = np.zeros(n_true, dtype=np.float64)
    assigned_count = np.zeros(n_true, dtype=np.float64)
    for true_state in range(n_true):
        members = np.where((alignment == true_state) & occupied)[0]
        assigned_count[true_state] = float(members.size)
        if members.size == 0:
            continue
        weights = row_totals[members]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        impurity[true_state] = float(np.sum(weights * row_entropy[members]) / total)
        effective_count[true_state] = float(np.sum(weights * row_effective_count[members]) / total)
    return impurity, effective_count, assigned_count


def _save_alignment_evidence_figure(payload: dict[str, Any], *, output_path: Path) -> None:
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    levels: list[int] = []
    methods: list[str] = []
    for key, panel in dict(payload["panels"]).items():
        match = PANEL_RE.match(str(key))
        if match is None:
            continue
        level = int(match.group("level"))
        method = _display_method(str(match.group("method")))
        parsed[(level, method)] = dict(panel)
        if level not in levels:
            levels.append(level)
        if method not in methods:
            methods.append(method)
    order_index = {level: idx for idx, level in enumerate(LEVEL_ORDER)}
    levels.sort(key=lambda level: (order_index.get(level, len(order_index)), level))

    nrows = len(levels)
    ncols = len(methods)
    fig_width = max(11.4, 3.05 * ncols + 1.2)
    fig_height = max(6.05, 1.74 * nrows + 0.82)
    height_ratios = [2.05 if int(level) == 13 else 1.0 for level in levels]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        dpi=220,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes_array = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.09, top=0.835, wspace=0.012, hspace=0.015)

    row_axes: dict[int, list[plt.Axes]] = {}
    for row_idx, level in enumerate(levels):
        positions = _positions_for_level(level)
        row_axes[level] = []
        for col_idx, method in enumerate(methods):
            ax = axes_array[row_idx, col_idx]
            row_axes[level].append(ax)
            _draw_evidence_panel(
                ax=ax,
                panel=parsed[(level, method)],
                positions=positions,
            )
            if row_idx == 0:
                ax.set_title(method, fontsize=9.8, fontweight="bold", pad=3)
            else:
                ax.set_title("")

    fig.canvas.draw()
    for level, axes_for_row in row_axes.items():
        _draw_level_label(fig, axes_for_row, LEVEL_LABELS.get(int(level), f"L{int(level)}"))

    entropy_cax = fig.add_axes([0.915, 0.18, 0.018, 0.62])
    entropy_mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=PURITY_CMAP)
    entropy_mappable.set_array([])
    cb = fig.colorbar(entropy_mappable, cax=entropy_cax, label=r"WTA-assigned purity")
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0 impure", "0.5", "1 pure"])
    cb.ax.tick_params(labelsize=8, pad=2)
    fig.suptitle(f"{str(payload['scenario'])}\nSeed-averaged WTA-assigned learned-state purity", fontsize=12, y=0.976)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _draw_evidence_panel(
    *,
    ax: plt.Axes,
    panel: dict[str, Any],
    positions: dict[int, np.ndarray],
) -> None:
    p_gt = np.asarray(panel["P_gt"], dtype=np.float64)
    entropy = np.asarray(panel["mean_wta_assigned_purity"], dtype=np.float64)
    n_true = int(p_gt.shape[0])
    pos = {node: np.asarray(positions[node], dtype=np.float64) for node in range(n_true)}
    for source, target in np.argwhere(p_gt > 0.0).tolist():
        _draw_arrow(ax, pos[int(source)], pos[int(target)], color="#d1d5db", width=1.0, alpha=0.9, zorder=1)
    for node in range(n_true):
        value = float(np.clip(entropy[node], 0.0, 1.0))
        color = PURITY_CMAP(value)
        center = pos[node]
        ax.add_patch(
            Circle(
                center,
                radius=0.285,
                facecolor=color,
                edgecolor="#111827",
                linewidth=0.55,
                alpha=0.96,
                zorder=3,
            )
        )
        text_color = "white" if value >= 0.58 else "#111827"
        ax.text(
            center[0],
            center[1],
            f"{node}\n{value:.2f}",
            ha="center",
            va="center",
            fontsize=6.3,
            fontweight="bold",
            color=text_color,
            linespacing=0.82,
            zorder=4,
        )
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    _pad_axis(ax, pos)


def _draw_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
    width: float,
    alpha: float,
    zorder: int,
) -> None:
    if float(np.linalg.norm(np.asarray(end) - np.asarray(start))) <= 1e-9:
        return
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=float(width),
        color=color,
        alpha=float(alpha),
        connectionstyle="arc3,rad=0.0",
        shrinkA=14,
        shrinkB=14,
        zorder=int(zorder),
    )
    ax.add_patch(patch)
def _pad_axis(ax: plt.Axes, pos: dict[int, np.ndarray]) -> None:
    coords = np.stack(list(pos.values()), axis=0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    ax.set_xlim(float(x_min - 0.28 * x_span), float(x_max + 0.28 * x_span))
    ax.set_ylim(float(y_min - 0.36 * y_span), float(y_max + 0.36 * y_span))


def _display_method(method: str) -> str:
    return METHOD_REPLACEMENTS.get(str(method), str(method))


if __name__ == "__main__":
    main()
