from dataclasses import dataclass

@dataclass(frozen=True)
class LatentParams:
    allow_self_loops: bool = True
    max_neurons: int = 50
    action_lr : float = 0.5
    gaussian_shape: int = 2
    max_age: int = 18          # satisfies HasMaxAge
