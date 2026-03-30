from dataclasses import dataclass

@dataclass(frozen=True)
class LatentParams:
    # ------------------------------------------------------------------
    # Graph / structure
    # ------------------------------------------------------------------
    allow_self_loops: bool = True
    max_neurons: int = 300
    max_age: int = 18  # satisfies HasMaxAge

    # ------------------------------------------------------------------
    # Learning / adaptation
    # ------------------------------------------------------------------
    action_lr: float = 0.5
    gaussian_shape: int = 2
    ambiguity_threshold: int = 10
    ambiguity_decay: float = 0.99

    # ------------------------------------------------------------------
    # Trace dynamics
    # ------------------------------------------------------------------
    trace_decay: float = 0.99
    lambda_trace: float = 0.05

    # ------------------------------------------------------------------
    # Energy weights (4-term decomposition)
    #
    # E_i = I_invalid(i)
    #       - w_mem * s_mem(i)
    #       - w_undir * s_undir(i)
    #       - w_base * s_base(i)
    #       - w_action * s_action(i)
    #
    # These are automatically normalised to sum to 1.
    # ------------------------------------------------------------------
    weight_memory: float = 0.4
    weight_undirected: float = 0.2
    weight_base: float = 0.2
    weight_action: float = 0.2

    # ------------------------------------------------------------------
    # Legacy mixture parameters (kept for compatibility)
    # These can be safely deprecated once all code is migrated.
    # ------------------------------------------------------------------
    mixture_alpha: float = 0.2
    mixture_beta: float = 0.2

    # ------------------------------------------------------------------
    # Memory behaviour
    # ------------------------------------------------------------------
    memory_replay: bool = True
    memory_disambiguation: bool = True

    # ------------------------------------------------------------------
    # Helper: normalized energy weights
    # ------------------------------------------------------------------
    def normalized_energy_weights(self) -> tuple[float, float, float, float]:
        raw_weights = (
            self.weight_memory,
            self.weight_undirected,
            self.weight_base,
            self.weight_action,
        )
        weight_sum = sum(raw_weights)

        if weight_sum <= 0.0:
            raise ValueError("Energy weights must sum to a positive value.")

        return tuple(weight / weight_sum for weight in raw_weights)
