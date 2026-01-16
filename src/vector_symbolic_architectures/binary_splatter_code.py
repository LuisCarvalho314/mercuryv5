import numpy as np



# -------------------------
# Primitives
# -------------------------
def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.logical_xor(a, b)

def cyclic_permute(vector: np.ndarray, shift: int) -> np.ndarray:
    shift = shift % vector.size
    return np.roll(vector, shift)

def bundle_deterministic(*vectors: np.ndarray) -> np.ndarray:
    """Majority bundle for {0,1} vectors with deterministic tie break."""
    stacked = np.stack(vectors, axis=0)  # (k, D)
    k, D = stacked.shape

    ones = np.sum(stacked, axis=0)
    zeros = k - ones

    out = np.empty(D, dtype=np.uint8)
    out[ones > zeros] = 1
    out[ones < zeros] = 0

    tie = (ones == zeros)
    if np.any(tie):
        # deterministic tie-break: use next bit of first and last vectors
        next_idx = (np.arange(D) + 1) % D
        tiebits = np.logical_xor(stacked[0, next_idx], stacked[-1, next_idx])
        out[tie] = tiebits[tie].astype(np.uint8)

    return out

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.logical_xor(a, b)))

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 1.0 means identical, 0.0 means all bits differ
    return 1.0 - hamming_distance(a, b) / a.size

# -------------------------
# Random HV helpers
# -------------------------
def random_hv(D: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=D, dtype=np.uint8)


# -------------------------
# Encoding
# -------------------------
def encode_labeled_directed_graph(
    edges: list[tuple[str, str, str]],  # (src, label, dst)
    item_memory_nodes: dict[str, np.ndarray],
    item_memory_labels: dict[str, np.ndarray],
    role_src: np.ndarray,
    role_rel: np.ndarray,
    role_dst: np.ndarray,
    dst_shift: int = 1,
) -> np.ndarray:
    triples = []
    for src, label, dst in edges:
        src_hv = item_memory_nodes[src]
        dst_hv = item_memory_nodes[dst]
        lbl_hv = item_memory_labels[label]

        triple = role_src ^ src_hv ^ role_rel ^ lbl_hv ^ role_dst ^ cyclic_permute(dst_hv, dst_shift)
        triples.append(triple.astype(np.uint8))

    return bundle_deterministic(*triples)

import numpy as np

def update_labelled_directed_graph(
    graph_hv: np.ndarray,
    src_hv: np.ndarray,
    rel_hv: np.ndarray,
    dst_hv: np.ndarray,
    role_src: np.ndarray,
    role_rel: np.ndarray,
    role_dst: np.ndarray,
    dst_shift: int = 1,
) -> np.ndarray:
    """
    Incrementally add one labeled directed edge (src --rel--> dst) into an
    existing BSC graph hypervector using the same triple encoding as
    encode_labeled_directed_graph.

    Notes
    -----
    - This recomputes a 2-vector majority bundle between the existing graph and
      the new triple. With k=2, ties are common; the deterministic tie-break in
      bundle_deterministic makes this reproducible.
    - If you want counts/strengths, add the same triple multiple times or keep
      a separate accumulator (e.g., integer votes) and threshold at the end.
    """
    edge_triple = (
        role_src
        ^ src_hv
        ^ role_rel
        ^ rel_hv
        ^ role_dst
        ^ cyclic_permute(dst_hv, dst_shift)
    ).astype(np.uint8)

    # Bundle existing graph with the new edge triple
    return bundle_deterministic(graph_hv.astype(np.uint8), edge_triple)


def update_labelled_directed_graph_many(
    graph_hv: np.ndarray,
    edges: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    role_src: np.ndarray,
    role_rel: np.ndarray,
    role_dst: np.ndarray,
    dst_shift: int = 1,
) -> np.ndarray:
    """
    Add multiple edges at once: edges is a list of (src_hv, rel_hv, dst_hv).
    """
    triples = []
    for src_hv, rel_hv, dst_hv in edges:
        triple = (
            role_src
            ^ src_hv
            ^ role_rel
            ^ rel_hv
            ^ role_dst
            ^ cyclic_permute(dst_hv, dst_shift)
        ).astype(np.uint8)
        triples.append(triple)

    return bundle_deterministic(graph_hv.astype(np.uint8), *triples)

# -------------------------
# Query: given (src,label) retrieve likely dst
# -------------------------
def query_dst_from_src_label(
    graph_hv: np.ndarray,
    src: str,
    label: str,
    item_memory_nodes: dict[str, np.ndarray],
    item_memory_labels: dict[str, np.ndarray],
    role_src: np.ndarray,
    role_rel: np.ndarray,
    role_dst: np.ndarray,
    dst_shift: int = 1,
) -> tuple[str, int]:
    src_hv = item_memory_nodes[src]
    lbl_hv = item_memory_labels[label]

    # Unbind everything except the (role_dst ^ permuted(dst)) term
    noisy = graph_hv ^ role_src ^ src_hv ^ role_rel ^ lbl_hv ^ role_dst

    # Undo permutation to get something close to dst_hv
    candidate = cyclic_permute(noisy, -dst_shift)

    # Cleanup by nearest neighbor in node item memory (min Hamming distance)
    best_name, best_dist = None, 10**9
    for name, hv in item_memory_nodes.items():
        dist = hamming_distance(candidate, hv)
        print(name, dist)
        if dist < best_dist:
            best_name, best_dist = name, dist

    return best_name, best_dist


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D = 10000

    nodes = ["a", "b", "c", "d"]
    labels = ["up", "down", "left", "right"]

    node_im = {n: random_hv(D, rng) for n in nodes}
    label_im = {l: random_hv(D, rng) for l in labels}

    R_SRC = random_hv(D, rng)
    R_REL = random_hv(D, rng)
    R_DST = random_hv(D, rng)

    edges = [("a", "right", "b"), ("b", "right", "b"), ("b", "left", "b"), ("b",
                                                                         "left",
                                                                         "a")]

    G1 = encode_labeled_directed_graph(edges, node_im, label_im, R_SRC,
                                       R_REL, R_DST, dst_shift=1)

    edges = [("a", "right", "b"), ("b", "left", "a")]

    G2 = encode_labeled_directed_graph(edges, node_im, label_im, R_SRC,
                                       R_REL, R_DST, dst_shift=1)

    #
    # print(query_dst_from_src_label(G, "b", "down", node_im, label_im, R_SRC,
    #                                R_REL, R_DST))

    print(hamming_similarity(G1, G2))





