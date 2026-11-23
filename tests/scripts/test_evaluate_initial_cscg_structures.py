from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analysis.evaluate_initial_cscg_structures import (
    _copy_shadow_metadata,
    _select_configs,
    counts_to_transition_tensor,
    coverage_failed,
)


@pytest.mark.parametrize(
    ("method", "preset", "has_structure"),
    [
        ("mercury_cscg_hybrid", "source_graph_clone_state", True),
        ("tmgwr_cscg_hybrid", "source_graph_first_order", True),
        ("cscg", "vanilla_cscg", False),
        ("cscg", "oracle_capacity", False),
        ("cscg", "oracle_capacity_first_order", True),
        ("cscg", "oracle_capacity_structural", True),
    ],
)
def test_every_initialisation_applies_pseudocount(
    method: str,
    preset: str,
    has_structure: bool,
) -> None:
    del method, preset
    counts = np.zeros((2, 3, 3), dtype=np.float32)
    if has_structure:
        counts[0, 0, 1] = 1.0
        counts[1, 2, 0] = 2.0

    transition = counts_to_transition_tensor(counts, pseudocount=0.002)

    assert np.all(transition > 0.0)
    assert np.allclose(transition.sum(axis=2), 1.0)
    if has_structure:
        assert transition[0, 0, 1] > transition[0, 0, 0]
    else:
        assert np.allclose(transition, np.full_like(transition, 1.0 / 3.0))


def test_empty_counts_require_positive_pseudocount() -> None:
    with pytest.raises(ValueError, match="empty row"):
        counts_to_transition_tensor(np.zeros((1, 2, 2), dtype=np.float32), pseudocount=0.0)


def test_select_configs_includes_the_six_main_study_arms() -> None:
    configs = []
    for index, preset in enumerate(
        (
            "vanilla_cscg",
            "oracle_capacity",
            "oracle_capacity_first_order",
            "oracle_capacity_structural",
        )
    ):
        configs.append(
            {
                "requested_run_id": f"cscg-{index}",
                "method": "cscg",
                "cscg_structural_preset": preset,
                "level": 13,
                "seed": 0,
                "rand_prob": 0.1,
                "cscg_batch_size": 40,
            }
        )
    configs.extend(
        {
            "requested_run_id": method,
            "method": method,
            "cscg_structural_preset": preset,
            "level": 13,
            "seed": 0,
            "rand_prob": 0.1,
            "cscg_batch_size": 40,
        }
        for method, preset in (
            ("mercury_cscg_hybrid", "source_graph_clone_state"),
            ("tmgwr_cscg_hybrid", "source_graph_first_order"),
        )
    )

    selected = _select_configs(
        configs,
        levels={13},
        seeds={0},
        methods={"cscg", "mercury_cscg_hybrid", "tmgwr_cscg_hybrid"},
        rand_prob=0.1,
        batch_size=40,
    )

    assert len(selected) == 6
    assert {row["cscg_structural_preset"] for row in selected} == {
        "source_graph_clone_state",
        "source_graph_first_order",
        "vanilla_cscg",
        "oracle_capacity",
        "oracle_capacity_first_order",
        "oracle_capacity_structural",
    }


def test_coverage_fails_after_incomplete_decode_or_nll() -> None:
    complete = {
        "status": "complete",
        "requested_walks": 100,
        "decoded_walks": 100,
        "decode_skipped_walks": 0,
        "decode_unsupported_walks": 0,
        "nll_skipped_walks": 0,
        "nll_unsupported_walks": 0,
    }
    assert coverage_failed(complete) is False
    assert coverage_failed({**complete, "decoded_walks": 99}) is True
    assert coverage_failed({**complete, "nll_unsupported_walks": 1}) is True
    assert coverage_failed({**complete, "status": "error"}) is True


def test_shadow_metadata_copy_does_not_change_source_study(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    run = source / "run-1"
    run.mkdir(parents=True)
    config = {"requested_run_id": "run-1", "method": "cscg"}
    manifest_text = json.dumps({"config": config}) + "\n"
    (source / "study_manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    (run / "run_config.json").write_text('{"value": 1}\n', encoding="utf-8")
    (run / "run_status.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    before = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }

    _copy_shadow_metadata(source_study=source, output_study=output, configs=[config])

    after = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert (output / "study_manifest.jsonl").read_text(encoding="utf-8") == manifest_text
    assert (output / "run-1" / "run_config.json").exists()
    assert json.loads((output / "source_study.json").read_text(encoding="utf-8"))["source_is_read_only"] is True
