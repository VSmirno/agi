"""CurriculumManager: progressive difficulty scaling (Stage 36).

Manages grid-size curriculum: 5x5 → 6x6 → 8x8 → 12x12.
Advances when success_rate >= threshold over a window of episodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class LevelStats:
    """Statistics for one curriculum level."""
    grid_size: int
    episodes: int = 0
    successes: int = 0
    total_steps: int = 0
    causal_links_learned: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(self.episodes, 1)


class CurriculumManager:
    """Progressive difficulty curriculum for grid environments.

    Starts from smallest grid, advances when success criterion is met.
    """

    def __init__(
        self,
        levels: list[int] | None = None,
        advance_threshold: float = 0.5,
        window_size: int = 20,
        min_episodes_per_level: int = 10,
    ) -> None:
        self.levels = levels or [5, 6, 8, 12]
        self.advance_threshold = advance_threshold
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level
        self._current_idx = 0
        self._windows: dict[int, deque[bool]] = {
            s: deque(maxlen=window_size) for s in self.levels
        }
        self._stats: dict[int, LevelStats] = {
            s: LevelStats(grid_size=s) for s in self.levels
        }

    @property
    def current_grid_size(self) -> int:
        return self.levels[self._current_idx]

    @property
    def current_level_idx(self) -> int:
        return self._current_idx

    @property
    def is_final_level(self) -> bool:
        return self._current_idx >= len(self.levels) - 1

    def record_episode(self, success: bool, steps: int = 0) -> None:
        """Record an episode result at current level."""
        size = self.current_grid_size
        self._windows[size].append(success)
        self._stats[size].episodes += 1
        self._stats[size].total_steps += steps
        if success:
            self._stats[size].successes += 1

    def should_advance(self) -> bool:
        """Check if agent should advance to next difficulty level."""
        if self.is_final_level:
            return False
        size = self.current_grid_size
        stats = self._stats[size]
        if stats.episodes < self.min_episodes_per_level:
            return False
        window = self._windows[size]
        if len(window) < min(self.window_size, self.min_episodes_per_level):
            return False
        window_rate = sum(window) / len(window)
        return window_rate >= self.advance_threshold

    def advance(self) -> int:
        """Advance to next level. Returns new grid_size."""
        if not self.is_final_level:
            self._current_idx += 1
        return self.current_grid_size

    def get_stats(self, grid_size: int | None = None) -> LevelStats:
        """Get stats for a specific level or current."""
        size = grid_size or self.current_grid_size
        return self._stats.get(size, LevelStats(grid_size=size))

    def all_stats(self) -> dict[int, LevelStats]:
        """Get stats for all levels."""
        return dict(self._stats)

    def max_steps_for_level(self, grid_size: int | None = None) -> int:
        """Suggested max steps per episode based on grid size."""
        size = grid_size or self.current_grid_size
        # Rough heuristic: grid_area * 3
        return size * size * 3

    def episodes_budget_for_level(self, grid_size: int | None = None) -> int:
        """Suggested episode budget based on grid size."""
        size = grid_size or self.current_grid_size
        if size <= 5:
            return 50
        elif size <= 6:
            return 50
        elif size <= 8:
            return 100
        else:
            return 200
