"""Authoritative evaluation-v5 records for the main and batch-sweep studies."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.analysis.evaluate_v5 import EvaluationV5Protocol, evaluation_v5_protocol_from_dict


STUDIES_ROOT = Path("results/current/action_alignment_rerun/evaluation_v5/studies")
PROTOCOL_ID = "evaluation-v5-nonself-v2-835996d8e9ad60d3"
MAIN_STUDY_ROOT = STUDIES_ROOT / "main_study"
BATCH_SWEEP_ROOT = STUDIES_ROOT / "cscg_batch_sweep"


def load_study_protocol(
    study_root: Path, expected_protocol_id: str | None = None
) -> EvaluationV5Protocol:
    """Load a study protocol and check its ID when one is selected or supplied."""
    protocol = evaluation_v5_protocol_from_dict(
        json.loads((Path(study_root) / "protocol.json").read_text(encoding="utf-8"))
    )
    actual = protocol.protocol_id()
    if expected_protocol_id is not None and actual != expected_protocol_id:
        raise ValueError(
            f"Protocol mismatch for {study_root}: expected {expected_protocol_id}, got {actual}"
        )
    return protocol
