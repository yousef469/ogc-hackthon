from __future__ import annotations
import math
import random


class SimulatedAnnealing:
    def __init__(self, initial_temp: float = 100.0, cooling_rate: float = 0.995):
        self.temperature = initial_temp
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = 0

    def accept(self, current_obj: float, new_obj: float, rng: random.Random) -> bool:
        self.iterations += 1
        if new_obj < current_obj:
            return True
        delta = new_obj - current_obj
        prob = math.exp(-delta / max(self.temperature, 1e-10))
        return rng.random() < prob

    def cool(self):
        self.temperature *= self.cooling_rate

    def reset(self):
        self.temperature = self.initial_temp
        self.iterations = 0
