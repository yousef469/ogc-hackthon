from __future__ import annotations
import math
import random
import time
from typing import Optional

from utils import Bay, Block, check_feasibility
from config import Config
from construction.strategies import ALL_STRATEGIES
from construction.helpers import build_operations, empty_bay_entry, block_bbox
from construction.repair import repair_simple
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
    n_bays = len(bays)

    best_assignments = None
    best_objective = float("inf")

    def fast_construct(block_order: list[int]) -> dict[int, dict]:
        assigns: dict[int, dict] = {}
        bay_sched: list[list] = [[] for _ in range(n_bays)]
        for bi in block_order:
            blk = blocks_data[bi]
            r_time = blk["release_time"]
            proc = blk["processing_time"]
            prefs = blk["bay_preferences"]
            n_o = len(blk["shape"])
            best = None
            best_score = float("inf")
            for bj in sorted(range(n_bays), key=lambda j: prefs[j], reverse=True):
                bay = bays[bj]
                for oi in range(n_o):
                    bb = block_bbox(blk, oi)
                    bw = bb[2] - bb[0]
                    bh = bb[3] - bb[1]
                    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                        continue
                    px = math.ceil(max(0.0, -bb[0] + 1e-9))
                    py = math.ceil(max(0.0, -bb[1] + 1e-9))
                    if px + bw > bay.width + 1e-6 or py + bh > bay.height + 1e-6:
                        continue
                    entry = empty_bay_entry(bay_sched[bj], r_time, proc)
                    if entry is not None:
                        tardy = max(0, entry + proc - blk["due_date"])
                        score = tardy * w1 + (max(prefs) - prefs[bj]) * w3
                        if score < best_score:
                            best_score = score
                            best = (bj, px, py, oi, int(entry), int(entry + proc))
                            break
                if best and best_score == 0:
                    break
            if not best:
                bj = max(range(n_bays), key=lambda j: prefs[j])
                bay = bays[bj]
                bb = block_bbox(blk, 0)
                bw = bb[2] - bb[0]
                bh = bb[3] - bb[1]
                px = math.ceil(max(0.0, -bb[0] + 1e-9))
                py = math.ceil(max(0.0, -bb[1] + 1e-9))
                if px + bw > bay.width + 1e-6 or py + bh > bay.height + 1e-6:
                    px, py = 0, 0
                entry = empty_bay_entry(bay_sched[bj], r_time, proc)
                best = (bj, px, py, 0, int(entry), int(entry + proc))
            bj, px, py, oi, entry, exit_t = best
            bay_sched[bj].append((entry, exit_t))
            assigns[bi] = {"block_id": bi, "bay_id": bj,
                           "x": px, "y": py, "orient_idx": oi,
                           "entry_time": entry, "exit_time": exit_t}
        return assigns

    strategy_time = config.get_construction_time(timelimit)
    strategies_to_try = list(ALL_STRATEGIES.keys())

    for strategy in strategies_to_try:
        if time.time() - t_start > strategy_time:
            break

        strat_start = time.time()
        block_order = ALL_STRATEGIES[strategy](blocks_data)
        assignments = fast_construct(block_order)

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
        edd_order = ALL_STRATEGIES["edd"](blocks_data)
        best_assignments = fast_construct(edd_order)
        best_assignments = repair_simple(prob_info, best_assignments, bays, blocks_data)
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
