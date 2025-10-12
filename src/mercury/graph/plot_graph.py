# src/mercury/graph/plot_graph.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, Iterable
import re

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------

_key_re = re.compile(r"^(?P<name>[A-Za-z0-9_\-]+)(\[(?P<idx>\d+)\])?$")


def _parse_feat_key(key: str) -> Tuple[str, Optional[int]]:
    m = _key_re.match(key)
    if not m:
        return key, None
    name = m.group("name")
    idx = m.group("idx")
    return name, (int(idx) if idx is not None else None)


def _resolve_value(d: Dict[str, Any], key: Optional[str], default: float = 0.0) -> float:
    if not key:
        return default
    name, idx = _parse_feat_key(key)
    if name not in d:
        return default
    v = d[name]
    if idx is None:
        if isinstance(v, (np.ndarray, list, tuple)) and np.ndim(v) > 0:
            try:
                return float(np.asarray(v).reshape(-1)[0])
            except Exception:
                return default
        try:
            return float(v)
        except Exception:
            return default
    try:
        arr = np.asarray(v).reshape(-1)
        if 0 <= idx < arr.size:
            return float(arr[idx])
        return default
    except Exception:
        return default

# ---------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------

def to_networkx(g) -> nx.Graph:
    """NumPy Graph -> NetworkX with split scalar/vector attrs."""
    G = nx.DiGraph() if getattr(g, "directed", True) else nx.Graph()

    # nodes
    for i in range(g.n):
        attrs: Dict[str, Any] = {}
        for name, arr in g.node_features.items():
            a = arr[i]
            if np.ndim(a) == 0:
                attrs[name] = float(a)
            else:
                a1 = np.asarray(a).ravel()
                attrs[name] = a1
                for k, vk in enumerate(a1):
                    attrs[f"{name}[{k}]"] = float(vk)
        G.add_node(i, **attrs)

    # edges
    nz_u, nz_v = np.nonzero(g.adj != 0)
    for u, v in zip(nz_u.tolist(), nz_v.tolist()):
        attrs: Dict[str, Any] = {"weight": float(g.adj[u, v])}
        for name, ef in g.edge_features.items():
            vec = ef[u, v, :]
            if vec.shape[0] == 1:
                attrs[name] = float(vec[0])
            else:
                for k, vk in enumerate(np.asarray(vec).ravel()):
                    attrs[f"{name}[{k}]"] = float(vk)
        G.add_edge(u, v, **attrs)

    return G

# ---------------------------------------------------------------------
# Layout and styling helpers
# ---------------------------------------------------------------------

def _layout(G: nx.Graph, name: str):
    name = (name or "spring").lower()
    if name in ("spring", "fr", "fruchterman_reingold"):
        return nx.spring_layout(G, seed=0)
    if name in ("kamada_kawai", "kk"):
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=0)
    if name == "circular":
        return nx.circular_layout(G)
    if name == "spectral":
        try:
            return nx.spectral_layout(G)
        except Exception:
            return nx.spring_layout(G, seed=0)
    return nx.spring_layout(G, seed=0)


def _normalize_widths(widths: Iterable[float]):
    widths = list(widths)
    if not widths:
        return widths
    w = np.asarray(widths, dtype=float)
    if not np.isfinite(w).all():
        return widths
    rng = float(np.ptp(w))  # NumPy 2.0 compatible
    return (1.0 + 2.0 * (w - float(w.min())) / (rng + 1e-9)).tolist()


def _collect_edge_colors(G: nx.Graph, key: Optional[str], cmap: str):
    if not key:
        return None, None
    vals = []
    for u, v in G.edges:
        vals.append(_resolve_value(G[u][v], key, default=0.0))
    return vals, plt.get_cmap(cmap)


def _collect_node_colors(G: nx.Graph, key: Optional[str], cmap: str):
    if not key:
        return "tab:blue", None
    vals = [_resolve_value(G.nodes[i], key, default=0.0) for i in G.nodes]
    return vals, plt.get_cmap(cmap)


def _edge_connectionstyles(G: nx.Graph, base_rad: float) -> Dict[Tuple[int, int], str]:
    """
    Per-edge curvature. For antiparallel pairs (u,v) and (v,u), give +rad and -rad.
    Otherwise 0 (straight). Returns mapping (u,v) -> "arc3,rad=...".
    """
    if base_rad == 0.0:
        return {(u, v): "arc3,rad=0.0" for u, v in G.edges}

    es: Dict[Tuple[int, int], str] = {}
    present = set(G.edges)
    for u, v in G.edges:
        if (v, u) in present and u < v:
            es[(u, v)] = f"arc3,rad={abs(base_rad)}"
            es[(v, u)] = f"arc3,rad={-abs(base_rad)}"
        elif (v, u) in present and u > v:
            # already assigned above
            continue
        else:
            es[(u, v)] = "arc3,rad=0.0"
    return es

# ---------------------------------------------------------------------
# Memory grid helpers
# ---------------------------------------------------------------------

def memory_grid_pos(S: int, L: int, dx: float = 1.0, dy: float = 1.0) -> dict[int, tuple[float, float]]:
    """Row s, col t -> node id s*L + t. Row 0 at top."""
    pos: dict[int, tuple[float, float]] = {}
    for s in range(S):
        base = s * L
        for t in range(L):
            pos[base + t] = (t * dx, (S - 1 - s) * dy)
    return pos


def draw_memory_grid_on_axes(
    ax: plt.Axes,
    mem_state,           # mercury.memory.state.MemoryState
    *,
    S: Optional[int] = None,
    L: Optional[int] = None,
    dx: float = 1.0,
    dy: float = 1.0,
    node_color_key: str = "activation",
    arrows: bool = True,
    alpha: float = 0.9,
) -> None:
    """Render MemoryState as an S×L grid using fixed positions."""
    g = mem_state.gs
    if L is None:
        L = int(mem_state.length)
    if S is None:
        S = int(g.n // L)

    G = to_networkx(g)
    pos = memory_grid_pos(S, L, dx=dx, dy=dy)

    node_colors, node_cmap = _collect_node_colors(G, node_color_key, "viridis")
    widths = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges]
    widths = _normalize_widths(widths)

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color=node_colors, cmap=node_cmap, alpha=alpha, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        width=widths if widths else 1.0,
        edge_color="k",
        arrows=arrows and isinstance(G, nx.DiGraph),
        alpha=alpha,
        ax=ax,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

# ---------------------------------------------------------------------
# Position caching for live plotting
# ---------------------------------------------------------------------

class Positioner:
    """
    Cache node positions. Recompute when node count increases or when forced.
    layout: "spring" | "kamada_kawai" | "circular" | "spectral"
    """

    def __init__(self, layout: str = "spring"):
        self.layout = layout
        self._n_last: int = -1
        self._pos: Optional[dict[int, tuple[float, float]]] = None

    def reset(self) -> None:
        self._n_last = -1
        self._pos = None

    def maybe_update(self, g) -> dict[int, tuple[float, float]]:
        """Recompute positions if node count grew. Otherwise reuse."""
        n = int(getattr(g, "n", 0))
        if self._pos is None or n > self._n_last:
            G = to_networkx(g)
            self._pos = _layout(G, self.layout)
            self._n_last = n
        return self._pos

# ---------------------------------------------------------------------
# Public drawing API
# ---------------------------------------------------------------------

def draw_graph_on_axes(
    ax: plt.Axes,
    g,
    *,
    layout: str = "spring",
    node_size: int | np.ndarray = 400,
    node_color_key: Optional[str] = None,
    node_label_key: Optional[str] = None,
    edge_width_key: str = "weight",
    edge_color_key: Optional[str] = None,
    cmap: str = "viridis",
    arrows: bool = True,
    with_labels: bool = False,
    alpha: float = 0.9,
    pos: Optional[dict[int, tuple[float, float]]] = None,  # external positions
    positioner: Optional[Positioner] = None,               # optional cache
    edge_curvature: float = 0.15,                          # curvature magnitude
) -> None:
    """
    Draw a Graph on the provided Axes.
    - If `pos` is provided, use it.
    - Else if `positioner` is provided, reuse/recompute there.
    - Else compute layout on the fly.
    - Curved edges help show antiparallel overlaps.
    """
    G = to_networkx(g)
    if pos is None:
        pos = positioner.maybe_update(g) if positioner is not None else _layout(G, layout)

    node_colors, node_cmap = _collect_node_colors(G, node_color_key, cmap)
    widths = [_resolve_value(G[u][v], edge_width_key, default=1.0) for u, v in G.edges]
    widths = _normalize_widths(widths)
    edge_colors, edge_cmap = _collect_edge_colors(G, edge_color_key, cmap)

    # nodes
    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, cmap=node_cmap, alpha=alpha, ax=ax
    )

    # edges with per-edge curvature when needed
    if len(G.edges):
        connstyles = _edge_connectionstyles(G, base_rad=float(edge_curvature or 0.0))
        # draw per-edge to vary curvature
        for i, (u, v) in enumerate(G.edges):
            w = widths[i] if isinstance(widths, list) and i < len(widths) else 1.0
            ec = (edge_colors[i] if edge_colors is not None and i < len(edge_colors) else "k")
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=w,
                edge_color=[ec] if edge_colors is not None else ec,
                edge_cmap=edge_cmap,
                arrows=arrows and isinstance(G, nx.DiGraph),
                alpha=alpha,
                ax=ax,
                connectionstyle=connstyles.get((u, v), "arc3,rad=0.0"),
            )

    if with_labels:
        if node_label_key:
            labels = {i: str(_resolve_value(G.nodes[i], node_label_key, default="")) for i in G.nodes}
        else:
            labels = {i: str(i) for i in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)

    ax.axis("off")


def plot_graph(
    g,
    *,
    layout: str = "spring",
    node_size: int | np.ndarray = 400,
    node_color_key: Optional[str] = None,
    node_label_key: Optional[str] = None,
    edge_width_key: str = "weight",
    edge_color_key: Optional[str] = None,
    cmap: str = "viridis",
    arrows: bool = True,
    with_labels: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.9,
    edge_curvature: float = 0.15,
):
    """Create a new figure and plot the graph."""
    G = to_networkx(g)
    pos = _layout(G, layout)

    node_colors, node_cmap = _collect_node_colors(G, node_color_key, cmap)
    if with_labels:
        if node_label_key:
            labels = {i: str(_resolve_value(G.nodes[i], node_label_key, default="")) for i in G.nodes}
        else:
            labels = {i: str(i) for i in G.nodes}
    else:
        labels = None

    widths = [_resolve_value(G[u][v], edge_width_key, default=1.0) for u, v in G.edges]
    widths = _normalize_widths(widths)
    edge_colors, edge_cmap = _collect_edge_colors(G, edge_color_key, cmap)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, cmap=node_cmap, alpha=alpha)

    # edges with curvature
    if len(G.edges):
        connstyles = _edge_connectionstyles(G, base_rad=float(edge_curvature or 0.0))
        for i, (u, v) in enumerate(G.edges):
            w = widths[i] if isinstance(widths, list) and i < len(widths) else 1.0
            ec = (edge_colors[i] if edge_colors is not None and i < len(edge_colors) else "k")
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=w,
                edge_color=[ec] if edge_colors is not None else ec,
                edge_cmap=edge_cmap,
                arrows=arrows and isinstance(G, nx.DiGraph),
                alpha=alpha,
                connectionstyle=connstyles.get((u, v), "arc3,rad=0.0"),
            )

    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    return ax
