from dataclasses import dataclass

@dataclass(frozen=True)
class LatentParams:
    allow_self_loops: bool = False
    max_neurons: int = 50
    action_lr : float = 0.5
    global_context_lr: float = 0.9
    max_age: int = 18          # satisfies HasMaxAge
