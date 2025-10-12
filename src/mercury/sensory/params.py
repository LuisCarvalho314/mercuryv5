from dataclasses import dataclass

@dataclass(frozen=True)
class SensoryParams:
    allow_self_loops: bool = False
    activation_threshold: float = 0.95
    topological_neighbourhood_threshold: float = 0.6
    max_neurons: int = 50
    sensory_weighting: float = 0.8
    winning_node_lr: float = 0.55
    topological_neighbourhood_lr: float = 0.9
    action_lr : float = 0.5
    global_context_lr: float = 0.9
    max_age: int = 18          # satisfies HasMaxAge
    gaussian_shape: int = 2
