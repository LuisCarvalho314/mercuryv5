from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryParams:
    length : int = 5