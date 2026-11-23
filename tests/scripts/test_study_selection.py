from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis import (
    mercury_source_initialisation_paired_contrasts,
    planned_paired_contrasts,
    plot_combined_metric_profiles,
    plot_cscg_batch_size_sensitivity,
    plot_pass_2_to_1000_metric_delta,
    postprocess_evaluation_v5,
)
from scripts.analysis.evaluate_v5 import EvaluationV5MatchedBudgetProtocol
from scripts.analysis.study_selection import (
    BATCH_SWEEP_ROOT,
    MAIN_STUDY_ROOT,
    PROTOCOL_ID,
    STUDIES_ROOT,
    load_study_protocol,
)


def test_selected_studies_use_the_recorded_batch_protocol() -> None:
    for root in (MAIN_STUDY_ROOT, BATCH_SWEEP_ROOT):
        assert load_study_protocol(root, PROTOCOL_ID).protocol_id() == PROTOCOL_ID


def test_loading_a_mismatched_protocol_fails(tmp_path: Path) -> None:
    payload = json.loads((MAIN_STUDY_ROOT / "protocol.json").read_text(encoding="utf-8"))
    payload["batch_overlap_samples"] = 0
    (tmp_path / "protocol.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_study_protocol(tmp_path).protocol_id() != PROTOCOL_ID
    with pytest.raises(ValueError, match="Protocol mismatch"):
        load_study_protocol(tmp_path, PROTOCOL_ID)


def test_study_analysis_defaults_and_overrides() -> None:
    for analysis in (
        planned_paired_contrasts,
        plot_pass_2_to_1000_metric_delta,
        mercury_source_initialisation_paired_contrasts,
    ):
        assert analysis.DEFAULT_STUDY_ROOT == MAIN_STUDY_ROOT
        assert analysis.DEFAULT_PROTOCOL_ID == PROTOCOL_ID
        args = analysis.build_parser().parse_args([])
        assert args.study_root == MAIN_STUDY_ROOT
        assert args.protocol_id == PROTOCOL_ID
    args = postprocess_evaluation_v5._build_parser().parse_args([])
    assert args.evaluation_studies_root == STUDIES_ROOT
    assert args.protocol_id is None
    custom = postprocess_evaluation_v5._build_parser().parse_args(
        ["--evaluation-studies-root", "other/studies", "--protocol-id", "other-protocol"]
    )
    assert custom.evaluation_studies_root == Path("other/studies")
    assert custom.protocol_id == "other-protocol"
    batch_args = plot_cscg_batch_size_sensitivity.build_parser().parse_args(
        ["--output-dir", "plots"]
    )
    assert batch_args.study_root == BATCH_SWEEP_ROOT
    assert batch_args.protocol_id == PROTOCOL_ID
    batch_custom = plot_cscg_batch_size_sensitivity.build_parser().parse_args(
        ["--study-root", "other/study", "--protocol-id", "other-id", "--output-dir", "plots"]
    )
    assert batch_custom.study_root == Path("other/study")
    assert batch_custom.protocol_id == "other-id"
    profile_args = plot_combined_metric_profiles.build_parser().parse_args([])
    assert profile_args.study_root == MAIN_STUDY_ROOT
    assert profile_args.protocol_id == PROTOCOL_ID


def test_postprocessor_accepts_matched_main_and_keeps_v2_default_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(postprocess_evaluation_v5, "STUDIES_ROOT", tmp_path)
    main_root = tmp_path / "main_study"
    batch_root = tmp_path / "cscg_batch_sweep"
    main_root.mkdir()
    batch_root.mkdir()
    (main_root / "protocol.json").write_text(
        json.dumps({"schema_version": EvaluationV5MatchedBudgetProtocol.schema_version}),
        encoding="utf-8",
    )
    (batch_root / "protocol.json").write_text(
        json.dumps({"schema_version": "evaluation_v5_conditional_nonself_v2"}),
        encoding="utf-8",
    )
    assert postprocess_evaluation_v5._expected_protocol_id(main_root, tmp_path, "main_study", None) is None
    assert postprocess_evaluation_v5._expected_protocol_id(batch_root, tmp_path, "cscg_batch_sweep", None) == PROTOCOL_ID
    assert postprocess_evaluation_v5._expected_protocol_id(main_root, tmp_path, "main_study", "explicit") == "explicit"
