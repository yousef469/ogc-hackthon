from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from utils import Block


@dataclass
class Assignment:
    block_id: int
    bay_id: int
    x: int
    y: int
    orient_idx: int
    entry_time: int
    exit_time: int

    def to_dict(self) -> dict:
        return {
            "block_id": self.block_id,
            "bay_id": self.bay_id,
            "x": self.x,
            "y": self.y,
            "orient_idx": self.orient_idx,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
        }


class Solution:
    def __init__(self, prob_info: dict):
        self.prob_info = prob_info
        n_bays = len(prob_info["bays"])
        self.assignments: dict[int, Assignment] = {}
        self.bay_blocks: list[list[Block]] = [[] for _ in range(n_bays)]
        self.bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
        self.bay_loads: list[float] = [0.0] * n_bays
        self._feasibility_result: dict | None = None

    def copy(self) -> Solution:
        new = Solution(self.prob_info)
        new.assignments = {
            bid: Assignment(**a.__dict__)
            for bid, a in self.assignments.items()
        }
        new.bay_blocks = [list(b) for b in self.bay_blocks]
        new.bay_schedule = [list(s) for s in self.bay_schedule]
        new.bay_loads = list(self.bay_loads)
        new._feasibility_result = self._feasibility_result
        return new

    def add_from_assignment(self, asgn: Assignment, block_data: dict):
        bay_id = asgn.bay_id
        blk = Block(
            block_id=asgn.block_id,
            block_data=block_data,
            x=asgn.x,
            y=asgn.y,
            orient_idx=asgn.orient_idx,
        )
        self.assignments[asgn.block_id] = asgn
        self.bay_blocks[bay_id].append(blk)
        self.bay_schedule[bay_id].append((asgn.entry_time, asgn.exit_time))
        self.bay_loads[bay_id] += block_data["workload"]
        self._feasibility_result = None

    def remove_block(self, block_id: int):
        if block_id not in self.assignments:
            return
        asgn = self.assignments[block_id]
        bay_id = asgn.bay_id
        for i, b in enumerate(self.bay_blocks[bay_id]):
            if b.block_id == block_id:
                self.bay_blocks[bay_id].pop(i)
                break
        for i, (entry, et) in enumerate(self.bay_schedule[bay_id]):
            if entry == asgn.entry_time and et == asgn.exit_time:
                self.bay_schedule[bay_id].pop(i)
                break
        self.bay_loads[bay_id] -= self.prob_info["blocks"][block_id]["workload"]
        del self.assignments[block_id]
        self._feasibility_result = None

    def evaluate(self) -> dict:
        from utils import check_feasibility
        if self._feasibility_result is None:
            ops = self._build_operations()
            self._feasibility_result = check_feasibility(
                self.prob_info, {"operations": ops}
            )
        return self._feasibility_result

    def is_feasible(self) -> bool:
        return self.evaluate().get("feasible", False)

    def objective_value(self) -> float | None:
        r = self.evaluate()
        return r.get("objective") if r.get("feasible") else None

    def obj_components(self) -> dict:
        r = self.evaluate()
        return {
            "obj1": r.get("obj1"),
            "obj2": r.get("obj2"),
            "obj3": r.get("obj3"),
        }

    def _build_operations(self) -> dict:
        from construction.helpers import build_operations
        return build_operations([
            a.to_dict() for a in self.assignments.values()
        ])

    def to_solution_dict(self) -> dict:
        return {"operations": self._build_operations()}
