from __future__ import annotations

from maze_environment.LEVEL import level_0, level_13, level_22, levels


def test_levels_index_matches_level_suffix() -> None:
    assert levels[0] == level_0
    assert levels[13] == level_13
    assert levels[22] == level_22
