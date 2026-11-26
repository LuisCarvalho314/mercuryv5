"""Colour-tile mazes: every cell carries a floor colour and the agent senses only that colour.

The outer ring of each grid is the wall; its value is never observed (level0 draws it as 0,
level1 as 1, and both values also occur as interior colours). Everything inside the ring is
open floor. ``ColourTileLevel`` exposes the wall mask and the colour grid separately, so the
maze environment can use the usual "non-zero = wall" collision rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR_SENSOR = "floor"


@dataclass(frozen=True)
class ColourTileLevel:
    grid: tuple[tuple[int, ...], ...]

    @property
    def walls(self) -> np.ndarray:
        walls = np.ones(np.shape(self.grid), dtype=np.uint8)
        walls[1:-1, 1:-1] = 0
        return walls

    @property
    def colours(self) -> np.ndarray:
        return np.asarray(self.grid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.grid)

    def __getitem__(self, row):
        return self.grid[row]


def is_colour_tile_level(level) -> bool:
    # Duck-typed: the package is importable as both `src.maze_environment` and `maze_environment`.
    return type(level).__name__ == "ColourTileLevel" and hasattr(level, "walls") and hasattr(level, "colours")


level0 = [
    [0,0,0,0,0,0,],
    [0,1,2,3,4,0,],
    [0,2,3,4,1,0,],
    [0,3,4,1,2,0,],
    [0,4,1,2,3,0,],
    [0,0,0,0,0,0,],
]

level1 = [
    [1,1,1,1,1,1,],
    [1,0,2,3,4,1,],
    [1,5,6,7,8,1,],
    [1,9,10,2,0,1,],
    [1,3,4,5,6,1,],
    [1,1,1,1,1,1,],
]


colour_levels = [level0, level1]
colour_tile_levels = [ColourTileLevel(tuple(tuple(row) for row in level)) for level in colour_levels]
