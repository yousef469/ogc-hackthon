from __future__ import annotations
import math
from typing import Optional

from utils import _bounding_box


class SpatialGrid:
    def __init__(self, bay_width: float, bay_height: float, cell_size: float = 20.0):
        self.cell_size = cell_size
        self.cols = max(1, math.ceil(bay_width / cell_size))
        self.rows = max(1, math.ceil(bay_height / cell_size))
        self.cells: list[list[set[int]]] = [
            [set() for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.block_cells: dict[int, set[tuple[int, int]]] = {}

    def _cell_coords(self, x: float, y: float) -> tuple[int, int]:
        c = min(self.cols - 1, max(0, int(x // self.cell_size)))
        r = min(self.rows - 1, max(0, int(y // self.cell_size)))
        return (c, r)

    def add(self, block_id: int, bbox: tuple[float, float, float, float]):
        min_x, min_y, max_x, max_y = bbox
        c1, r1 = self._cell_coords(min_x, min_y)
        c2, r2 = self._cell_coords(max_x, max_y)
        occupied: set[tuple[int, int]] = set()
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.cells[r][c].add(block_id)
                occupied.add((c, r))
        self.block_cells[block_id] = occupied

    def remove(self, block_id: int):
        if block_id in self.block_cells:
            for c, r in self.block_cells[block_id]:
                self.cells[r][c].discard(block_id)
            del self.block_cells[block_id]

    def get_overlapping(self, block_id: int, bbox: tuple) -> set[int]:
        min_x, min_y, max_x, max_y = bbox
        c1, r1 = self._cell_coords(min_x, min_y)
        c2, r2 = self._cell_coords(max_x, max_y)
        nearby: set[int] = set()
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                nearby.update(self.cells[r][c])
        nearby.discard(block_id)
        return nearby
