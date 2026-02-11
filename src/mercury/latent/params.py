from dataclasses import dataclass

@dataclass(frozen=True)
class LatentParams:

    allow_self_loops: bool = True
    max_neurons: int = 300
    action_lr : float = 0.5
    gaussian_shape: int = 2
    max_age: int = 18          # satisfies HasMaxAge
    ambiguity_threshold: int = 10
    trace_decay: float = 0.99
    mixture_alpha: float = 0.2
    mixture_beta: float = 0.2
    memory_replay: bool = True
    memory_disambiguation: bool = True
