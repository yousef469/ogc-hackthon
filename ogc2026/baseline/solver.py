from __future__ import annotations
import math
import random
import time
from typing import Optional

from utils import Bay, Block, check_feasibility
from config import Config
from construction.strategies import ALL_STRATEGIES
from construction.greedy import run_greedy
from construction.repair import repair_simple
from construction.helpers import build_operations
from improvement.lns import run_lns
from improvement.parallel import run_parallel_lns
from improvement.refine import refine_solution
from core.objective import fast_objective


def solve(prob_info: dict, timelimit: float = 60.0) -> dict:
    t_start = time.time()
    config = Config(prob_info)

    bays_data = prob_info["bays"]
    blocks_data = prob_info["blocks"]
    weights = prob_info.get("weights", {})
    w1 = weights.get("w1", 1.0)
    w2 = weights.get("w2", 1.0)
    w3 = weights.get("w3", 1.0)

    bays = [Bay.from_dict(d, i) for i, d in enumerate(bays_data)]

    best_assignments = None
    best_objective = float("inf")

    strategy_time = config.get_construction_time(timelimit)
    strategies_to_try = list(ALL_STRATEGIES.keys())

    for strategy in strategies_to_try:
        if time.time() - t_start > strategy_time:
            break

        strat_start = time.time()
        assignments, bay_placed, bay_schedule, bay_loads = run_greedy(
            blocks_data, bays, w1, w2, w3,
            strategy=strategy,
        )

        sol = {"operations": build_operations(list(assignments.values()))}

        repaired = repair_simple(
            prob_info, assignments, bays, blocks_data,
        )
        repaired_sol = {"operations": build_operations(list(repaired.values()))}

        result = check_feasibility(prob_info, repaired_sol)
        if result["feasible"]:
            obj = result["objective"]
            if obj is not None and obj < best_objective:
                best_objective = obj
                best_assignments = repaired
                print(f"[Solver] {strategy} -> objective {obj:.0f}  "
                      f"elapsed={time.time()-strat_start:.1f}s")

    if best_assignments is None:
        print(f"[Solver] No feasible construction, falling back to EDD")
        assignments, *_ = run_greedy(blocks_data, bays, w1, w2, w3, strategy="edd")
        sol = {"operations": build_operations(list(assignments.values()))}
        best_assignments = repair_simple(
            prob_info, assignments, bays, blocks_data,
        )
        best_objective = check_feasibility(
            prob_info, {"operations": build_operations(list(best_assignments.values()))}
        ).get("objective", float("inf"))

    lns_time_remaining = max(1.0, timelimit - (time.time() - t_start) - 1.0)
    lns_budget = lns_time_remaining * 0.85

    lns_result = run_parallel_lns(
        prob_info, bays, blocks_data, w1, w2, w3,
        best_assignments, t_start, lns_budget,
        config, num_workers=min(config.num_workers, 2),
        verbose=True,
    )

    refine_budget = max(1.0, timelimit - (time.time() - t_start) - 0.5)

    if refine_budget > 2.0:
        n_bays = len(bays)
        bay_placed_r: list[list[Block]] = [[] for _ in range(n_bays)]
        bay_schedule_r: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
        bay_loads_r: list[float] = [0.0] * n_bays

        for bid, a in lns_result.items():
            bay_id = a["bay_id"]
            bay_placed_r[bay_id].append(Block(
                block_id=bid, block_data=blocks_data[bid],
                x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
            ))
            bay_schedule_r[bay_id].append((a["entry_time"], a["exit_time"]))
            bay_loads_r[bay_id] += blocks_data[bid]["workload"]

        final_assignments = refine_solution(
            lns_result, blocks_data, bays,
            bay_placed_r, bay_schedule_r, bay_loads_r,
            bays_data, weights,
            t_start, refine_budget,
            rng=random.Random(99), verbose=True,
        )
    else:
        final_assignments = lns_result

    final_sol = {"operations": build_operations(list(final_assignments.values()))}
    result = check_feasibility(prob_info, final_sol)
    elapsed = time.time() - t_start

    if not result.get("feasible", False):
        fallback_obj = best_objective if best_objective is not None else 0
        print(f"[Solver] LNS best not feasible, falling back to construction ({fallback_obj:.0f})")
        final_assignments = {bid: dict(a) for bid, a in best_assignments.items()}
        final_sol = {"operations": build_operations(list(final_assignments.values()))}
        result = check_feasibility(prob_info, final_sol)
        obj = result.get("objective") or fallback_obj
    else:
        obj = result.get("objective") or 0

    print(f"[Solver] Final objective: {obj:.0f}  "
          f"(obj1={result.get('obj1', 0) or 0:.1f}  "
          f"obj2={result.get('obj2', 0) or 0:.1f}  "
          f"obj3={result.get('obj3', 0) or 0:.1f})  "
          f"elapsed={elapsed:.1f}s")
    return final_sol
