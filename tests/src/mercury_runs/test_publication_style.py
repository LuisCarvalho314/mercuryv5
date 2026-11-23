from __future__ import annotations

from mercury_runs.analysis.plots import publication_style as style


def test_primary_method_labels_and_colours_are_canonical() -> None:
    expected = {
        "mercury": ("MERCURY", "#D55E00"),
        "tmgwr": ("TMGWR", "#E69F00"),
        "mercury_cscg_hybrid": ("MERCURY initialised", "#CC79A7"),
        "tmgwr_cscg_hybrid": ("TMGWR initialised", "#7B61A8"),
        "vanilla_cscg": ("Standard CSCG", "#264653"),
        "oracle_capacity": ("Oracle capacity", "#0072B2"),
        "oracle_capacity_structural": ("Oracle capacity + structure", "#009E73"),
        "oracle_capacity_first_order": ("Oracle capacity + first-order structure", "#56B4E9"),
    }

    for identifier, (label, colour) in expected.items():
        assert style.method_label(identifier) == label
        assert style.method_color(identifier) == colour


def test_cscg_variants_are_ordered_by_increasing_privileged_information() -> None:
    assert tuple(style.method_label(identifier) for identifier in style.CSCG_VARIANT_ORDER) == (
        "Standard CSCG",
        "Oracle capacity",
        "Oracle capacity + first-order structure",
        "Oracle capacity + structure",
    )


def test_historical_labels_resolve_to_publication_vocabulary() -> None:
    assert style.method_label("MERCURY-CSCG Hybrid") == "MERCURY initialised"
    assert style.method_label("TMGWR-CSCG Hybrid") == "TMGWR initialised"
    assert style.method_label("CSCG - vanilla") == "Standard CSCG"
    assert style.method_label("Oracle structural") == "Oracle capacity + structure"
    assert style.method_label("Oracle first-order") == "Oracle capacity + first-order structure"


def test_checkpoint_2000_uses_raw_tmgwr_identity() -> None:
    assert style.method_label("tmgwr_hybrid", comparison="checkpoint_2000") == "TMGWR"
    assert style.method_color("tmgwr_hybrid", comparison="checkpoint_2000") == "#E69F00"
    assert style.method_label("tmgwr_hybrid", comparison="checkpoint_4000") == "TMGWR initialised"
    assert style.method_color("tmgwr_hybrid", comparison="checkpoint_4000") == "#7B61A8"


def test_metric_labels_are_publication_terms() -> None:
    assert style.metric_label("latent_precision") == "Latent purity"
    assert style.metric_label("latent_recall") == "Latent concentration"
