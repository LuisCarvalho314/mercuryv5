# src/mercury/graph/plot_graph.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import re

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Helpers
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
    """
    Get d[key] or d[name][idx] if key is like 'feat[0]'.
    Missing -> default. Arrays are assumed numpy arrays or sequences.
    """
    if not key:
        return default
    name, idx = _parse_feat_key(key)
    if name not in d:
        return default
    v = d[name]
    if idx is None:
        # scalar or vector: if vector, try first component
        if isinstance(v, (np.ndarray, list, tuple)) and np.ndim(v) > 0:
            try:
                return float(np.asarray(v).reshape(-1)[0])
            except Exception:
                return default
        try:
            return float(v)
        except Exception:
            return default
    # indexed
    try:
        arr = np.asarray(v).reshape(-1)
        if 0 <= idx < arr.size:
            return float(arr[idx])
        return default
    except Exception:
        return default

def to_networkx(g) -> nx.Graph:
    """Convert NumPy Graph -> NetworkX with split scalar/vector attrs."""
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
                attrs[name] = a1  # optional full vector
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
                attrs[name] = float(vec[0])          # scalar channel -> "name"
            else:
                # split vector channels -> "name[idx]"
                for k, vk in enumerate(np.asarray(vec).ravel()):
                    attrs[f"{name}[{k}]"] = float(vk)
        G.add_edge(u, v, **attrs)

    return G



def _layout(G: nx.Graph, name: str):
    name = (name or "spring").lower()
    if name in ("spring", "fr", "fruchterman_reingold"):
        return nx.spring_layout(G, seed=0)
    if name in ("kamada_kawai", "kk"):
        # Requires SciPy in NetworkX. Fall back if unavailable.
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


def _normalize_widths(widths):
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
        val = _resolve_value(G[u][v], key, default=0.0)
        vals.append(val)
    return vals, plt.get_cmap(cmap)


def _collect_node_colors(G: nx.Graph, key: Optional[str], cmap: str):
    if not key:
        return "tab:blue", None
    vals = [_resolve_value(G.nodes[i], key, default=0.0) for i in G.nodes]
    return vals, plt.get_cmap(cmap)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def plot_graph(
    g,
    *,
    layout: str = "spring",
    node_size: int | np.ndarray = 400,
    node_color_key: Optional[str] = None,     # e.g. "activation" or "state[0]"
    node_label_key: Optional[str] = None,     # text label per node from node feature key
    edge_width_key: str = "weight",           # edge attribute for width
    edge_color_key: Optional[str] = None,     # e.g. "age" or "age[0]"
    cmap: str = "viridis",
    arrows: bool = True,
    with_labels: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.9,
):
    """
    Visualize the graph with NetworkX + Matplotlib.

    Parameters
    ----------
    g : Graph
        Your NumPy Graph instance.
    layout : str
        'spring' | 'kamada_kawai' | 'circular' | 'spectral'.
    node_size : int or array
        Fixed size or per-node array-like.
    node_color_key : str or None
        Node feature key to color by. Example: 'activation' or 'state[0]'.
    node_label_key : str or None
        Node feature key to label nodes with. If None and with_labels=True, labels are node ids.
    edge_width_key : str
        Edge attribute to set widths. Default 'weight'.
    edge_color_key : str or None
        Edge attribute for color. Example: 'age' or 'age[0]'.
    """
    G = to_networkx(g)
    pos = _layout(G, layout)

    # node colors
    node_colors, node_cmap = _collect_node_colors(G, node_color_key, cmap)

    # node labels
    if with_labels:
        if node_label_key:
            labels = {i: str(_resolve_value(G.nodes[i], node_label_key, default="")) for i in G.nodes}
        else:
            labels = {i: str(i) for i in G.nodes}
    else:
        labels = None

    # edge widths
    widths = [_resolve_value(G[u][v], edge_width_key, default=1.0) for u, v in G.edges]
    widths = _normalize_widths(widths)

    # edge colors
    edge_colors, edge_cmap = _collect_edge_colors(G, edge_color_key, cmap)

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        cmap=node_cmap,
        alpha=alpha,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths if widths else 1.0,
        edge_color=edge_colors if edge_colors is not None else "k",
        edge_cmap=edge_cmap,
        arrows=arrows and isinstance(G, nx.DiGraph),
        alpha=alpha,
    )

    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.axis("off")
    plt.tight_layout()
    return plt.gca()
