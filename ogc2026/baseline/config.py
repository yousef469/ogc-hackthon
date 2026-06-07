from __future__ import annotations
import math


class Config:
    def __init__(self, prob_info: dict | None = None):
        self.construction_time_ratio: float = 0.05
        self.num_construction_strategies: int = 5
        self.repair_max_passes: int = 10

        self.sa_initial_temperature: float = 100.0
        self.sa_cooling_rate: float = 0.9995

        self.num_workers: int = 4
        self.grid_cell_size: int = 20
        self.use_parallel: bool = True

        if prob_info is not None:
            self.auto_tune(prob_info)

    def auto_tune(self, prob_info: dict):
        weights = prob_info.get("weights", {})
        w1 = weights.get("w1", 1)
        n_blocks = len(prob_info.get("blocks", []))

        self.sa_initial_temperature = 50000.0 * w1 / 2667.0
        self.sa_cooling_rate = 0.99995
        self.sa_cooling_power = 3.0
        base_workers = min(4, max(2, n_blocks // 60 + 1))
        self.num_workers = min(4, base_workers)

    def get_construction_time(self, timelimit: float) -> float:
        return min(30.0, max(5.0, timelimit * 0.08))
