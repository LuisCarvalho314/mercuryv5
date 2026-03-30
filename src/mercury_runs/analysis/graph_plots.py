from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import networkx as nx
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from mercury.graph.core import Graph
from ..algorithms.pocml.artifacts import capacity_run_id
from ..save_results import read_bundle_meta


@dataclass(frozen=True)
class GraphSpec:
    method_name: str
    title: str
    nodes: list[int]
    node_obs_modes: dict[int, int]
    edge_summary: dict[tuple[int, int], dict[str, float]]
    action_names: list[str]
    pos: dict[int, np.ndarray]
    node_labels: dict[int, str]
    action_vectors: np.ndarray | None = None


def _load_run_artifacts(run_root: Path) -> dict[str, str]:
    status_path = run_root / "run_status.json"
    if not status_path.exists():
        raise FileNotFoundError(f"Missing run status: {status_path}")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts") or {}
    return {key: value for key, value in artifacts.items() if isinstance(value, str) and value}


def _resolve_dataset_parquet(states_parquet: Path, datasets_root: Path) -> Path:
    meta = read_bundle_meta(states_parquet)
    if meta is None or not meta.source.dataset_parquet_name:
        raise RuntimeError(f"No source dataset metadata found in {states_parquet}")
    matches = sorted(datasets_root.rglob(meta.source.dataset_parquet_name), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"Could not locate dataset parquet {meta.source.dataset_parquet_name} under {datasets_root}")
    return matches[0]


def _extract_dataset_sequences(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pl.read_parquet(dataset_path)
    obs_cols = [column for column in frame.columns if column.startswith("observation_")]
    action_cols = [column for column in frame.columns if column.startswith("action_")]
    if not obs_cols or not action_cols:
        raise RuntimeError(f"Dataset schema missing observation/action columns: {dataset_path}")
    observations = frame.select(obs_cols).to_numpy()
    actions = frame.select(action_cols).to_numpy()
    action_names = [name.replace("action_", "") for name in action_cols]
    return observations, actions, action_names


def _obs_ids(observations: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(observations, axis=0, return_inverse=True)
    return inverse.astype(np.int32, copy=False)


def _build_transition_summary(node_series: np.ndarray, action_series: np.ndarray) -> dict[tuple[int, int], dict[str, float]]:
    edge_action_counts: dict[tuple[int, int], dict[int, int]] = {}
    limit = max(0, min(len(node_series) - 1, len(action_series)))
    for idx in range(limit):
        edge = (int(node_series[idx]), int(node_series[idx + 1]))
        edge_action_counts.setdefault(edge, {})
        action = int(action_series[idx])
        edge_action_counts[edge][action] = edge_action_counts[edge].get(action, 0) + 1

    edge_summary: dict[tuple[int, int], dict[str, float]] = {}
    for edge, counts in edge_action_counts.items():
        action_mode = max(counts.items(), key=lambda item: item[1])[0]
        edge_summary[edge] = {
            "action_mode": float(action_mode),
            "count": float(sum(counts.values())),
            "weight": float(sum(counts.values())),
        }
    return edge_summary


def _build_edge_summary_from_graph(graph: Graph) -> dict[tuple[int, int], dict[str, float]]:
    adjacency = np.asarray(graph.adj, dtype=np.float32)
    action_feature = graph.edge_features.get("action")
    if action_feature is None:
        raise RuntimeError("Mercury graph is missing edge feature 'action'.")

    actions = np.asarray(action_feature[:, :, 0], dtype=np.int32)
    edge_summary: dict[tuple[int, int], dict[str, float]] = {}
    sources, targets = np.nonzero(adjacency != 0.0)
    for source, target in zip(sources.tolist(), targets.tolist(), strict=False):
        edge_summary[(int(source), int(target))] = {
            "action_mode": float(actions[source, target]),
            "count": float(adjacency[source, target]),
            "weight": float(adjacency[source, target]),
        }
    return edge_summary


def _filter_edge_summary_to_nodes(
    edge_summary: dict[tuple[int, int], dict[str, float]],
    nodes: Iterable[int],
) -> dict[tuple[int, int], dict[str, float]]:
    node_set = {int(node) for node in nodes}
    return {
        (int(u), int(v)): info
        for (u, v), info in edge_summary.items()
        if int(u) in node_set and int(v) in node_set
    }


def _build_node_obs_modes(node_series: np.ndarray, obs_ids: np.ndarray) -> dict[int, int]:
    buckets: dict[int, list[int]] = {}
    for node, obs_id in zip(node_series.tolist(), obs_ids.tolist()):
        buckets.setdefault(int(node), []).append(int(obs_id))
    result: dict[int, int] = {}
    for node, values in buckets.items():
        counts = np.bincount(np.asarray(values, dtype=np.int64))
        result[node] = int(np.argmax(counts))
    return result


def _label_nodes(nodes: Iterable[int], obs_modes: dict[int, int], prefix: str) -> dict[int, str]:
    return {int(node): f"{prefix}{int(node)}\nO{int(obs_modes.get(int(node), 0))}" for node in nodes}


def _spring_positions(nodes: list[int], edge_summary: dict[tuple[int, int], dict[str, float]], *, seed: int) -> dict[int, np.ndarray]:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for edge, info in edge_summary.items():
        graph.add_edge(edge[0], edge[1], weight=float(info.get("weight", 1.0)))
    if graph.number_of_edges() == 0:
        return {int(node): np.asarray([float(index), 0.0], dtype=np.float32) for index, node in enumerate(nodes)}
    pos = nx.spring_layout(graph, seed=seed, weight="weight")
    return {int(node): np.asarray(coords, dtype=np.float32) for node, coords in pos.items()}


def _action_ids(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] == 0:
        raise RuntimeError(f"Unsupported action array shape: {actions.shape}")
    return np.argmax(actions, axis=1).astype(np.int32, copy=False)


def _load_state_columns(states_path: Path, *columns: str) -> dict[str, np.ndarray]:
    frame = pl.read_parquet(states_path).select(*columns)
    return {
        column: frame.get_column(column).to_numpy().astype(np.int32, copy=False)
        for column in columns
    }


def _recognized_action_vectors(action_names: list[str]) -> dict[int, np.ndarray]:
    vectors: dict[int, np.ndarray] = {}
    lookup = {
        "north": np.asarray([0.0, 1.0], dtype=np.float32),
        "up": np.asarray([0.0, 1.0], dtype=np.float32),
        "east": np.asarray([1.0, 0.0], dtype=np.float32),
        "right": np.asarray([1.0, 0.0], dtype=np.float32),
        "south": np.asarray([0.0, -1.0], dtype=np.float32),
        "down": np.asarray([0.0, -1.0], dtype=np.float32),
        "west": np.asarray([-1.0, 0.0], dtype=np.float32),
        "left": np.asarray([-1.0, 0.0], dtype=np.float32),
    }
    for idx, name in enumerate(action_names):
        vector = lookup.get(str(name).strip().lower())
        if vector is not None:
            vectors[idx] = vector
    return vectors


def _anchor_positions_from_ground_truth(
    *,
    gt_series: np.ndarray,
    action_series: np.ndarray,
    action_names: list[str],
) -> dict[int, np.ndarray]:
    gt_nodes = sorted(set(int(value) for value in gt_series.tolist()))
    if not gt_nodes:
        return {}
    directions = _recognized_action_vectors(action_names)
    if not directions:
        return {node: np.asarray([float(index), 0.0], dtype=np.float32) for index, node in enumerate(gt_nodes)}

    node_to_index = {node: index for index, node in enumerate(gt_nodes)}
    n = len(gt_nodes)
    edges: list[tuple[int, int, float, np.ndarray]] = []
    limit = max(0, min(len(gt_series) - 1, len(action_series)))
    for idx in range(limit):
        action_id = int(action_series[idx])
        direction = directions.get(action_id)
        if direction is None:
            continue
        u = int(gt_series[idx])
        v = int(gt_series[idx + 1])
        if u == v:
            continue
        edges.append((node_to_index[u], node_to_index[v], 1.0, direction))

    if not edges:
        return {node: np.asarray([float(index), 0.0], dtype=np.float32) for index, node in enumerate(gt_nodes)}

    def _solve_axis(axis: int) -> np.ndarray:
        anchor_index = 0
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for source, target, weight, direction in edges:
            row = np.zeros(n - 1, dtype=np.float64)
            if source != anchor_index:
                row[source - 1 if source > anchor_index else source] -= np.sqrt(weight)
            if target != anchor_index:
                row[target - 1 if target > anchor_index else target] += np.sqrt(weight)
            rows.append(row)
            targets.append(float(np.sqrt(weight) * direction[axis]))
        matrix = np.vstack(rows)
        vector = np.asarray(targets, dtype=np.float64)
        solved, *_ = np.linalg.lstsq(matrix, vector, rcond=None)
        coords = np.zeros(n, dtype=np.float64)
        coords[1:] = solved if anchor_index == 0 else np.insert(solved, anchor_index, 0.0)
        return coords.astype(np.float32, copy=False)

    x = _solve_axis(0)
    y = _solve_axis(1)
    positions = np.stack([x, y], axis=1)
    positions -= positions.mean(axis=0, keepdims=True)

    undirected = nx.Graph()
    undirected.add_nodes_from(range(n))
    for source, target, _weight, _direction in edges:
        undirected.add_edge(source, target)
    components = list(nx.connected_components(undirected))
    if len(components) > 1:
        ordered = sorted(components, key=lambda component: min(component))
        cursor = 0.0
        spacing = 4.0
        shifted = positions.copy()
        for component in ordered:
            indices = np.asarray(sorted(component), dtype=np.int32)
            comp = shifted[indices]
            comp_center = comp.mean(axis=0)
            shifted[indices, 0] += cursor - comp_center[0]
            shifted[indices, 1] -= comp_center[1]
            cursor += spacing
        positions = shifted

    return {
        node: np.asarray(positions[index], dtype=np.float32)
        for node, index in node_to_index.items()
    }


def _node_to_ground_truth_assignments(
    *,
    node_series: np.ndarray,
    gt_series: np.ndarray,
    node_obs_modes: dict[int, int],
) -> tuple[dict[int, int], dict[int, float]]:
    counts: dict[int, dict[int, int]] = {}
    limit = min(len(node_series), len(gt_series))
    for node, gt in zip(node_series[:limit].tolist(), gt_series[:limit].tolist()):
        counts.setdefault(int(node), {})
        counts[int(node)][int(gt)] = counts[int(node)].get(int(gt), 0) + 1
    assignments: dict[int, int] = {}
    confidence: dict[int, float] = {}
    for node, gt_counts in counts.items():
        total = float(sum(gt_counts.values()))
        best_gt, best_count = max(gt_counts.items(), key=lambda item: (item[1], -item[0]))
        assignments[node] = int(best_gt)
        confidence[node] = float(best_count / total) if total > 0 else 0.0
    for node in node_obs_modes:
        assignments.setdefault(int(node), int(min(gt_series.tolist())) if gt_series.size else 0)
        confidence.setdefault(int(node), 0.0)
    return assignments, confidence


def _cluster_offsets(
    *,
    nodes: list[int],
    node_obs_modes: dict[int, int],
    confidence: dict[int, float],
    base_radius: float,
) -> dict[int, np.ndarray]:
    if not nodes:
        return {}
    grouped: dict[int, list[int]] = {}
    for node in nodes:
        grouped.setdefault(int(node_obs_modes.get(node, 0)), []).append(int(node))
    offsets: dict[int, np.ndarray] = {}
    group_keys = sorted(grouped)
    for group_index, group_key in enumerate(group_keys):
        group_nodes = sorted(grouped[group_key], key=lambda node: (-confidence.get(node, 0.0), node))
        angle = (2.0 * np.pi * group_index) / max(len(group_keys), 1)
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        for rank, node in enumerate(group_nodes):
            if rank == 0:
                offsets[node] = direction * (base_radius * 0.15)
                continue
            ring = 1 + (rank - 1) // 4
            step = (rank - 1) % 4
            radial = direction * (base_radius * (0.6 + 0.45 * ring))
            tangential = tangent * (base_radius * 0.35 * (step - 1.5))
            offsets[node] = radial + tangential
    return offsets


def _resolve_node_overlaps(
    *,
    pos: dict[int, np.ndarray],
    nodes: list[int],
    node_obs_modes: dict[int, int],
    assignments: dict[int, int],
    gt_positions: dict[int, np.ndarray],
    confidence: dict[int, float],
    base_radius: float,
) -> dict[int, np.ndarray]:
    if len(nodes) < 2:
        return pos

    adjusted = {
        int(node): np.asarray(coords, dtype=np.float32).copy()
        for node, coords in pos.items()
    }
    max_radius = max(base_radius * 3.2, 0.5)
    for _ in range(28):
        moved = False
        for idx, node_a in enumerate(nodes):
            for node_b in nodes[idx + 1 :]:
                pa = adjusted[int(node_a)]
                pb = adjusted[int(node_b)]
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                same_obs = int(node_obs_modes.get(int(node_a), -1)) == int(node_obs_modes.get(int(node_b), -2))
                target = base_radius * (1.65 if same_obs else 2.15)
                if dist >= target:
                    continue
                direction = delta / dist if dist > 1e-6 else np.asarray([1.0, 0.0], dtype=np.float32)
                push = float((target - dist) * 0.5)
                conf_a = float(confidence.get(int(node_a), 0.0))
                conf_b = float(confidence.get(int(node_b), 0.0))
                total_conf = conf_a + conf_b + 1e-6
                move_a = push * (conf_b + 1e-6) / total_conf
                move_b = push * (conf_a + 1e-6) / total_conf
                adjusted[int(node_a)] = pa - direction * move_a
                adjusted[int(node_b)] = pb + direction * move_b
                moved = True
        for node in nodes:
            anchor = np.asarray(gt_positions.get(int(assignments.get(int(node), 0)), [0.0, 0.0]), dtype=np.float32)
            offset = adjusted[int(node)] - anchor
            norm = float(np.linalg.norm(offset))
            if norm > max_radius:
                adjusted[int(node)] = anchor + offset * (max_radius / norm)
        if not moved:
            break
    return adjusted


def _grounded_positions(
    *,
    nodes: list[int],
    node_series: np.ndarray,
    gt_series: np.ndarray,
    action_series: np.ndarray,
    action_names: list[str],
    node_obs_modes: dict[int, int],
    edge_summary: dict[tuple[int, int], dict[str, float]],
) -> dict[int, np.ndarray]:
    gt_positions = _anchor_positions_from_ground_truth(
        gt_series=gt_series,
        action_series=action_series,
        action_names=action_names,
    )
    if not gt_positions:
        return _spring_positions(nodes, edge_summary, seed=42)

    assignments, confidence = _node_to_ground_truth_assignments(
        node_series=node_series,
        gt_series=gt_series,
        node_obs_modes=node_obs_modes,
    )
    if len(gt_positions) > 1:
        gt_coords = np.stack(list(gt_positions.values()), axis=0)
        dists = []
        for idx in range(len(gt_coords)):
            for jdx in range(idx + 1, len(gt_coords)):
                dists.append(float(np.linalg.norm(gt_coords[idx] - gt_coords[jdx])))
        min_anchor_distance = min(dists) if dists else 1.0
    else:
        min_anchor_distance = 1.0
    base_radius = max(0.18, min_anchor_distance * 0.18)

    anchor_to_nodes: dict[int, list[int]] = {}
    for node in nodes:
        anchor_to_nodes.setdefault(int(assignments.get(node, 0)), []).append(int(node))

    pos: dict[int, np.ndarray] = {}
    for anchor, cluster_nodes in anchor_to_nodes.items():
        anchor_pos = gt_positions.get(anchor, np.asarray([0.0, 0.0], dtype=np.float32))
        offsets = _cluster_offsets(
            nodes=cluster_nodes,
            node_obs_modes=node_obs_modes,
            confidence=confidence,
            base_radius=base_radius,
        )
        for node in cluster_nodes:
            pos[int(node)] = np.asarray(anchor_pos + offsets.get(int(node), 0.0), dtype=np.float32)

    missing = [node for node in nodes if node not in pos]
    if missing:
        fallback = _spring_positions(missing, {}, seed=13)
        for node in missing:
            pos[int(node)] = fallback[int(node)]
    return _resolve_node_overlaps(
        pos=pos,
        nodes=nodes,
        node_obs_modes=node_obs_modes,
        assignments=assignments,
        gt_positions=gt_positions,
        confidence=confidence,
        base_radius=base_radius,
    )


def _action_names_from_count(count: int) -> list[str]:
    return [f"a{idx}" for idx in range(max(count, 1))]


def _draw_graph_on_axes(ax: plt.Axes, spec: GraphSpec) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(spec.nodes)
    for (u, v), info in spec.edge_summary.items():
        graph.add_edge(u, v, **info)

    obs_values = np.asarray([spec.node_obs_modes.get(node, 0) for node in spec.nodes], dtype=np.int32)
    n_obs = int(obs_values.max()) + 1 if obs_values.size else 1
    obs_cmap = plt.get_cmap("tab20", max(n_obs, 2))
    action_cmap = plt.get_cmap("tab10", max(len(spec.action_names), 2))

    node_colors = [obs_cmap(int(spec.node_obs_modes.get(node, 0))) for node in spec.nodes]
    edge_colors = [action_cmap(int(graph.edges[edge].get("action_mode", 0))) for edge in graph.edges()]
    edge_widths = [0.45 + min(1.6, 0.22 * np.log1p(float(graph.edges[edge].get("weight", 1.0)))) for edge in graph.edges()]

    nx.draw_networkx_nodes(graph, pos=spec.pos, node_color=node_colors, node_size=520, ax=ax, linewidths=0.4, edgecolors="k")
    nx.draw_networkx_labels(graph, pos=spec.pos, labels=spec.node_labels, font_size=8, ax=ax)
    if graph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            graph,
            pos=spec.pos,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )

    if spec.action_vectors is not None:
        for node in spec.nodes:
            x0, y0 = float(spec.pos[node][0]), float(spec.pos[node][1])
            for action_idx in range(spec.action_vectors.shape[0]):
                dx, dy = float(spec.action_vectors[action_idx, 0]), float(spec.action_vectors[action_idx, 1])
                ax.plot(
                    [x0, x0 + dx],
                    [y0, y0 + dy],
                    color=action_cmap(action_idx),
                    alpha=0.18,
                    linewidth=0.8,
                    zorder=0,
                )

    handles = [plt.Line2D([0], [0], color=action_cmap(idx), lw=2, label=f"action={name}") for idx, name in enumerate(spec.action_names)]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title(spec.title)
    ax.set_axis_off()


def _save_graph_plot(spec: GraphSpec, out_path: Path) -> Path:
    matplotlib.use("Agg", force=True)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=220)
    _draw_graph_on_axes(ax, spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _save_combined_plot(specs: list[GraphSpec], out_path: Path) -> Path:
    matplotlib.use("Agg", force=True)
    columns = max(len(specs), 1)
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 6), dpi=220)
    axes_list = [axes] if columns == 1 else list(np.ravel(axes))
    for ax, spec in zip(axes_list, specs):
        _draw_graph_on_axes(ax, spec)
    for ax in axes_list[len(specs) :]:
        ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _infer_mercury_current_timestep_observations(
    *, mercury_graph: Graph, n_observations: int, memory_length: int
) -> tuple[dict[int, int], dict[int, str]]:
    mem_adj = np.asarray(mercury_graph.node_features.get("mem_adj"), dtype=np.float32)
    nodes = list(range(int(mercury_graph.n)))
    if mem_adj.ndim != 2 or mem_adj.shape[0] < mercury_graph.n or memory_length <= 0:
        fallback_labels = {node: f"M{node}\nS?" for node in nodes}
        return ({node: 0 for node in nodes}, fallback_labels)

    mem_len = int(mem_adj.shape[1])
    if mem_len % memory_length != 0:
        fallback_labels = {node: f"M{node}\nS?" for node in nodes}
        return ({node: 0 for node in nodes}, fallback_labels)

    n_slots = mem_len // memory_length
    idx_now = np.arange(n_slots, dtype=np.int64) * memory_length
    mem_adj_now = mem_adj[: mercury_graph.n, idx_now]
    label_prefix = "O" if n_slots == n_observations and n_observations > 0 else "S"

    unknown_color = n_slots
    multi_color = n_slots + 1
    node_obs_modes: dict[int, int] = {}
    node_labels: dict[int, str] = {}
    for node in nodes:
        active = np.flatnonzero(mem_adj_now[node] > 0)
        if active.size == 1:
            slot_idx = int(active[0])
            node_obs_modes[node] = slot_idx
            node_labels[node] = f"M{node}\n{label_prefix}{slot_idx}"
            continue
        if active.size == 0:
            node_obs_modes[node] = unknown_color
            node_labels[node] = f"M{node}\n{label_prefix}?"
            continue
        node_obs_modes[node] = multi_color
        slot_labels = ",".join(f"{label_prefix}{int(slot_idx)}" for slot_idx in active.tolist())
        node_labels[node] = f"M{node}\n{slot_labels}"
    return node_obs_modes, node_labels


def _spec_mercury(
    *,
    mercury_states_path: Path,
    mercury_graph_path: Path,
    n_observations: int,
    memory_length: int,
    action_names: list[str],
    action_ids: np.ndarray,
) -> GraphSpec:
    mercury_graph = Graph.from_npz(str(mercury_graph_path))
    state_columns = _load_state_columns(mercury_states_path, "latent_bmu", "cartesian_proxy_bmu")
    latent_series = state_columns["latent_bmu"]
    gt_series = state_columns["cartesian_proxy_bmu"]
    nodes = list(range(int(mercury_graph.n)))
    edge_summary = _filter_edge_summary_to_nodes(_build_edge_summary_from_graph(mercury_graph), nodes)
    node_obs_modes, node_labels = _infer_mercury_current_timestep_observations(
        mercury_graph=mercury_graph,
        n_observations=n_observations,
        memory_length=memory_length,
    )
    return GraphSpec(
        method_name="mercury",
        title="Mercury Internal Latent Graph",
        nodes=nodes,
        node_obs_modes=node_obs_modes,
        edge_summary=edge_summary,
        action_names=action_names,
        pos=_grounded_positions(
            nodes=nodes,
            node_series=latent_series,
            gt_series=gt_series,
            action_series=action_ids,
            action_names=action_names,
            node_obs_modes=node_obs_modes,
            edge_summary=edge_summary,
        ),
        node_labels=node_labels,
    )


def _spec_pocml(
    *,
    pocml_embeddings_path: Path,
    dataset_obs_ids: np.ndarray,
    action_names: list[str],
    title_suffix: str = "",
) -> GraphSpec:
    embeddings = np.load(pocml_embeddings_path)
    q_matrix = np.nan_to_num(np.asarray(embeddings["Q"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v_matrix = np.nan_to_num(np.asarray(embeddings["V"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    state_sequence = np.asarray(embeddings["state_sequence"], dtype=np.int32)
    action_sequence = np.asarray(embeddings["action_sequence"], dtype=np.int32)
    state_obs_from_memory = np.asarray(embeddings["state_obs_from_memory"], dtype=np.int32)

    pca = PCA(n_components=2)
    q_pca = pca.fit_transform(q_matrix.T)
    v_pca = pca.transform(v_matrix.T)

    nodes = list(range(q_pca.shape[0]))
    length = min(len(state_sequence), len(dataset_obs_ids))
    node_obs_modes = {int(idx): int(state_obs_from_memory[idx]) for idx in range(len(state_obs_from_memory))}
    if length:
        node_obs_modes.update(_build_node_obs_modes(state_sequence[:length], dataset_obs_ids[:length]))
    edge_summary = _build_transition_summary(state_sequence, action_sequence)
    pos = {int(node): np.asarray(q_pca[int(node)], dtype=np.float32) for node in nodes}
    return GraphSpec(
        method_name="pocml",
        title=("POCML Internal State Graph" + title_suffix),
        nodes=nodes,
        node_obs_modes=node_obs_modes,
        edge_summary=edge_summary,
        action_names=action_names,
        pos=pos,
        node_labels=_label_nodes(nodes, node_obs_modes, "P"),
        action_vectors=np.asarray(v_pca, dtype=np.float32),
    )


def _resolve_pocml_capacity_specs(
    *,
    run_root: Path,
    resolved_artifacts: dict[str, str],
    dataset_obs_ids: np.ndarray,
    action_names: list[str],
) -> tuple[list[tuple[str, GraphSpec]], GraphSpec | None]:
    pocml_embeddings = resolved_artifacts.get("pocml_embeddings_npz")
    if not pocml_embeddings:
        return [], None

    primary_embeddings_path = Path(pocml_embeddings)
    primary_spec = _spec_pocml(
        pocml_embeddings_path=primary_embeddings_path,
        dataset_obs_ids=dataset_obs_ids,
        action_names=action_names,
    )

    pocml_states_raw = resolved_artifacts.get("pocml_states_parquet")
    if not pocml_states_raw:
        return [("pocml_plot", primary_spec)], primary_spec

    meta = read_bundle_meta(Path(pocml_states_raw))
    pocml_eval = ((meta.run_parameters or {}).get("pocml_eval") or {}) if meta is not None else {}
    capacities = [int(value) for value in (pocml_eval.get("precision_capacities") or [])]
    primary_capacity = pocml_eval.get("primary_precision_capacity")
    if not capacities or primary_capacity is None:
        return [("pocml_plot", primary_spec)], primary_spec

    primary_capacity = int(primary_capacity)
    specs: list[tuple[str, GraphSpec]] = [("pocml_plot", primary_spec)]
    for capacity in capacities:
        capacity_path = primary_embeddings_path.with_name(
            f"{capacity_run_id(base_run_id=meta.run_id, capacity=int(capacity), primary_capacity=primary_capacity)}_embeddings.npz"
        )
        if not capacity_path.exists():
            continue
        specs.append(
            (
                f"pocml_plot_capacity={int(capacity)}",
                _spec_pocml(
                    pocml_embeddings_path=capacity_path,
                    dataset_obs_ids=dataset_obs_ids,
                    action_names=action_names,
                    title_suffix=f" (K={int(capacity)})",
                ),
            )
        )
    return specs, primary_spec


def _cscg_clone_observation_labels(n_clones: np.ndarray) -> np.ndarray:
    return np.arange(n_clones.shape[0], dtype=np.int32).repeat(n_clones.astype(np.int64, copy=False))


def _spec_cscg(
    *,
    cscg_states_path: Path,
    cscg_model_path: Path,
    dataset_obs_ids: np.ndarray,
    action_names: list[str],
    action_ids: np.ndarray,
) -> GraphSpec:
    states_frame = pl.read_parquet(cscg_states_path).select("cscg_state_id", "cartesian_proxy_bmu")
    clone_state_sequence = states_frame.get_column("cscg_state_id").to_numpy().astype(np.int32, copy=False)
    gt_series = states_frame.get_column("cartesian_proxy_bmu").to_numpy().astype(np.int32, copy=False)
    model = np.load(cscg_model_path)
    n_clones = np.asarray(model["n_clones"], dtype=np.int32)

    used_states = np.unique(clone_state_sequence).astype(np.int32, copy=False)
    resolved_action_names = action_names or _action_names_from_count(int(np.max(action_ids)) + 1 if action_ids.size else 0)
    nodes = [int(state) for state in used_states.tolist()]
    edge_summary = _filter_edge_summary_to_nodes(_build_transition_summary(clone_state_sequence, action_ids), nodes)
    # CSCG clone states are tied to observation families by construction via n_clones.
    # Do not overwrite those labels with empirical observation modes from the decoded sequence.
    node_obs_modes = {int(state): int(_cscg_clone_observation_labels(n_clones)[int(state)]) for state in used_states.tolist()}
    return GraphSpec(
        method_name="cscg",
        title="CSCG Internal Clone Graph",
        nodes=nodes,
        node_obs_modes=node_obs_modes,
        edge_summary=edge_summary,
        action_names=resolved_action_names,
        pos=_grounded_positions(
            nodes=nodes,
            node_series=clone_state_sequence,
            gt_series=gt_series,
            action_series=action_ids,
            action_names=resolved_action_names,
            node_obs_modes=node_obs_modes,
            edge_summary=edge_summary,
        ),
        node_labels=_label_nodes(nodes, node_obs_modes, "C"),
    )


def generate_method_graph_plots(
    *,
    run_root: Path,
    datasets_root: Path,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Path]:
    resolved_artifacts = artifacts or _load_run_artifacts(run_root)
    mercury_states = resolved_artifacts.get("mercury_states_parquet")
    mercury_graph = resolved_artifacts.get("mercury_latent_graph_npz")
    if not mercury_states or not mercury_graph:
        raise RuntimeError(f"Missing Mercury artifacts in run status for {run_root}")

    mercury_states_path = Path(mercury_states)
    dataset_path = _resolve_dataset_parquet(mercury_states_path, datasets_root)
    observations, actions, action_names = _extract_dataset_sequences(dataset_path)
    dataset_obs_ids = _obs_ids(observations)
    action_ids = _action_ids(actions)
    n_observations = int(dataset_obs_ids.max()) + 1 if dataset_obs_ids.size else 0
    mercury_meta = read_bundle_meta(mercury_states_path)
    memory_length = int(mercury_meta.memory_length) if mercury_meta is not None else 0

    specs: list[GraphSpec] = [
        _spec_mercury(
            mercury_states_path=mercury_states_path,
            mercury_graph_path=Path(mercury_graph),
            n_observations=n_observations,
            memory_length=memory_length,
            action_names=action_names,
            action_ids=action_ids,
        )
    ]

    pocml_capacity_specs, pocml_primary_spec = _resolve_pocml_capacity_specs(
        run_root=run_root,
        resolved_artifacts=resolved_artifacts,
        dataset_obs_ids=dataset_obs_ids,
        action_names=action_names,
    )
    if pocml_primary_spec is not None:
        specs.append(pocml_primary_spec)

    cscg_states = resolved_artifacts.get("cscg_states_parquet")
    cscg_model = resolved_artifacts.get("cscg_model_npz")
    if cscg_states and cscg_model:
        specs.append(
            _spec_cscg(
                cscg_states_path=Path(cscg_states),
                cscg_model_path=Path(cscg_model),
                dataset_obs_ids=dataset_obs_ids,
                action_names=action_names,
                action_ids=action_ids,
            )
        )

    outputs: dict[str, Path] = {}
    for spec in specs:
        outputs[f"{spec.method_name}_plot"] = _save_graph_plot(spec, run_root / "plots" / f"{spec.method_name}_internal_graph.png")
    for output_key, spec in pocml_capacity_specs:
        if output_key == "pocml_plot":
            continue
        capacity_suffix = output_key.replace("pocml_plot_", "")
        outputs[output_key] = _save_graph_plot(spec, run_root / "plots" / f"pocml_internal_graph_{capacity_suffix}.png")
    outputs["combined_plot"] = _save_combined_plot(specs, run_root / "plots" / "internal_graphs_comparison.png")
    return outputs
