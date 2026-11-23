"""Canonical evaluation protocol schema, hashing, and serialisation.

The protocol captures every parameter that determines evaluation semantics.
A change to any of the listed fields must create a new protocol identifier.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class EvaluationProtocol:
    """Versioned evaluation protocol.  Hash uniquely identifies the protocol."""

    schema_version: str = "canonical_v5_conditional_nonself"
    protocol_name: str = "canonical-v5"

    # Decoder
    decoder: str = "method_standard_online"
    decoder_configuration_source: str = "snapshot"
    online_inference_mode: str = "frozen_recursive"

    # Inference state
    inference_state_initialisation: str = "method_standard_reset"
    inference_state_reset_policy: str = "reset_per_walk"
    initial_timestep_policy: str = "include_all"
    burn_in_steps: int = 0

    # Evaluation walks
    evaluation_seed: int = 0
    number_of_walks: int = 100
    walk_length: int = 10_000
    start_state_policy: str = "uniform_open_positions"

    # Environment
    environment: str = "maze"
    environment_implementation_version: str = "v3"
    sensor: str = "cartesian"
    sensor_range: int | None = None
    rand_prob: float = 0.3
    valid_trajectories_only: bool = False

    # Ground truth
    ground_truth_source: str = "empirical_held_out_counts"

    # Aggregation policies
    primary_tv_aggregation: str = "evaluation_occupancy"
    secondary_tv_aggregations: list[str] = field(default_factory=lambda: ["macro"])
    primary_self_loop_policy: str = "include"
    diagnostic_self_loop_policy: str = "conditional_non_self_loop"

    # Collapsed action marginalisation
    collapsed_action_marginalisation: str = "evaluation_policy"
    collapsed_state_aggregation: str = "evaluation_occupancy"
    optional_collapsed_action_marginalisations: list[str] = field(default_factory=list)

    # Edge metrics
    edge_thresholds: list[float] = field(default_factory=lambda: [1e-6])

    # Alignment
    unmatched_state_policy: str = "explicit_sink"
    alignment_policy: str = "many_to_one_wta"
    alignment_tie_breaking: str = "smallest_sorted_state_id"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def protocol_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def protocol_id(self) -> str:
        return f"{self.protocol_name}-{self.protocol_hash()}"

    def walk_cache_key(self) -> str:
        """Key identifying the walk-generating parameters (model-independent)."""
        cache_params = {
            "environment": self.environment,
            "environment_implementation_version": self.environment_implementation_version,
            "sensor": self.sensor,
            "sensor_range": self.sensor_range,
            "rand_prob": self.rand_prob,
            "valid_trajectories_only": self.valid_trajectories_only,
            "evaluation_seed": self.evaluation_seed,
            "number_of_walks": self.number_of_walks,
            "walk_length": self.walk_length,
            "start_state_policy": self.start_state_policy,
        }
        canonical = json.dumps(cache_params, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def default_canonical_v4_protocol(
    *,
    level: int,
    sensor: str = "cartesian",
    sensor_range: int | None = None,
    rand_prob: float = 0.3,
    evaluation_seed: int = 0,
    number_of_walks: int = 100,
    walk_length: int = 10_000,
    valid_trajectories_only: bool = False,
    edge_thresholds: list[float] | None = None,
) -> EvaluationProtocol:
    """Construct the current canonical protocol.

    The historical function name is retained for API compatibility.  Protocol
    identity is v5 because conditional no-self-loop metrics now aggregate only
    over rows with ground-truth movement support.
    """
    return EvaluationProtocol(
        sensor=sensor,
        sensor_range=sensor_range,
        rand_prob=rand_prob,
        evaluation_seed=evaluation_seed,
        number_of_walks=number_of_walks,
        walk_length=walk_length,
        valid_trajectories_only=valid_trajectories_only,
        edge_thresholds=edge_thresholds if edge_thresholds is not None else [1e-6],
    )


def protocol_from_dict(d: dict[str, Any]) -> EvaluationProtocol:
    known = set(EvaluationProtocol.__dataclass_fields__.keys())
    filtered = {k: v for k, v in d.items() if k in known}
    return EvaluationProtocol(**filtered)


def protocol_from_yaml(path: Any) -> EvaluationProtocol:
    import yaml  # type: ignore
    from pathlib import Path
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return protocol_from_dict(data)
