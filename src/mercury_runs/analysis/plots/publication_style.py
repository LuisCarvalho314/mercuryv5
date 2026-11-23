"""Shared publication labels and colours for maintained analysis figures."""
from __future__ import annotations


METHOD_LABELS = {
    "mercury": "MERCURY",
    "tmgwr": "TMGWR",
    "mercury_initialised": "MERCURY initialised",
    "tmgwr_initialised": "TMGWR initialised",
    "standard_cscg": "Standard CSCG",
    "oracle_capacity": "Oracle capacity",
    "oracle_capacity_structural": "Oracle capacity + structure",
    "oracle_capacity_first_order": "Oracle capacity + first-order structure",
    "ground_truth_state_1": "GT states x1",
    "ground_truth_state_2": "GT states x2",
    "uniform_obs_4": "Obs clones x4",
    "uniform_obs_8": "Obs clones x8",
}

METHOD_COLORS = {
    "mercury": "#D55E00",
    "tmgwr": "#E69F00",
    "mercury_initialised": "#CC79A7",
    "tmgwr_initialised": "#7B61A8",
    "standard_cscg": "#264653",
    "oracle_capacity": "#0072B2",
    "oracle_capacity_structural": "#009E73",
    "oracle_capacity_first_order": "#56B4E9",
    "ground_truth_state_1": "#1D4E89",
    "ground_truth_state_2": "#468FAF",
    "uniform_obs_4": "#2A9D8F",
    "uniform_obs_8": "#76C893",
}

CSCG_VARIANT_ORDER = (
    "vanilla_cscg",
    "oracle_capacity",
    "oracle_capacity_first_order",
    "oracle_capacity_structural",
)

METHOD_ALIASES = {
    "mercury": "mercury",
    "MERCURY": "mercury",
    "Mercury": "mercury",
    "mercury_cscg_hybrid_mercury": "mercury",
    "tmgwr": "tmgwr",
    "TMGWR": "tmgwr",
    "tmgwr_cscg_hybrid_tmgwr": "tmgwr",
    "hybrid": "mercury_initialised",
    "mercury_cscg_hybrid": "mercury_initialised",
    "Hybrid": "mercury_initialised",
    "Hybrid CSCG": "mercury_initialised",
    "Mercury-CSCG": "mercury_initialised",
    "MERCURY-CSCG Hybrid": "mercury_initialised",
    "MERCURY–CSCG Hybrid": "mercury_initialised",
    "MERCURY initialised": "mercury_initialised",
    "tmgwr_hybrid": "tmgwr_initialised",
    "tmgwr_cscg_hybrid": "tmgwr_initialised",
    "TMGWR-CSCG": "tmgwr_initialised",
    "TMGWR-CSCG Hybrid": "tmgwr_initialised",
    "TMGWR hybrid": "tmgwr_initialised",
    "TMGWR initialised": "tmgwr_initialised",
    "vanilla_cscg": "standard_cscg",
    "Vanilla CSCG": "standard_cscg",
    "CSCG - vanilla": "standard_cscg",
    "Standard CSCG": "standard_cscg",
    "oracle_capacity": "oracle_capacity",
    "CSCG - oracle capacity": "oracle_capacity",
    "Oracle capacity": "oracle_capacity",
    "oracle_capacity_structural": "oracle_capacity_structural",
    "CSCG - oracle structural": "oracle_capacity_structural",
    "Oracle structural": "oracle_capacity_structural",
    "Oracle capacity + structure": "oracle_capacity_structural",
    "oracle_capacity_first_order": "oracle_capacity_first_order",
    "CSCG - oracle first-order": "oracle_capacity_first_order",
    "Oracle first-order": "oracle_capacity_first_order",
    "Oracle capacity + first-order structure": "oracle_capacity_first_order",
    "ground_truth_state_1": "ground_truth_state_1",
    "ground_truth_state_2": "ground_truth_state_2",
    "uniform_obs_4": "uniform_obs_4",
    "uniform_obs_8": "uniform_obs_8",
}

METRIC_LABELS = {
    "latent_precision": "Latent purity",
    "latent_recall": "Latent concentration",
}

PUBLICATION_COMPARISON_TITLES = {
    "checkpoint_2000": "After one training pass",
    "checkpoint_4000": "After two training passes",
    "final": "Final comparison",
}


def method_id(identifier: str, *, comparison: str | None = None) -> str:
    resolved = METHOD_ALIASES.get(str(identifier), str(identifier))
    if comparison == "checkpoint_2000" and resolved == "tmgwr_initialised":
        return "tmgwr"
    return resolved


def method_label(identifier: str, *, comparison: str | None = None) -> str:
    resolved = method_id(identifier, comparison=comparison)
    return METHOD_LABELS.get(resolved, str(identifier))


def method_color(identifier: str, *, comparison: str | None = None, default: str = "#4B5563") -> str:
    return METHOD_COLORS.get(method_id(identifier, comparison=comparison), default)


def metric_label(metric: str, fallback: str | None = None) -> str:
    return METRIC_LABELS.get(str(metric), fallback if fallback is not None else str(metric))
