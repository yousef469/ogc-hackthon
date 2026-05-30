from __future__ import annotations


class Config:
    def __init__(self, prob_info: dict | None = None):
        self.construction_time_ratio: float = 0.05
        self.num_construction_strategies: int = 5
        self.repair_max_passes: int = 10

        self.sa_initial_temperature: float = 100.0
        self.sa_cooling_rate: float = 0.999

        self.num_workers: int = 4
        self.grid_cell_size: int = 20
        self.use_parallel: bool = False

        if prob_info is not None:
            self.auto_tune(prob_info)

    def auto_tune(self, prob_info: dict):
        weights = prob_info.get("weights", {})
        w1 = weights.get("w1", 1)

        self.sa_initial_temperature = 50000.0 * w1 / 2667.0

    def get_construction_time(self, timelimit: float) -> float:
        return min(120.0, max(10.0, timelimit * 0.15))
