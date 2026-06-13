from __future__ import annotations
import math
import random
import time
from typing import Optional

from utils import Bay, Block, check_feasibility, check_entry, check_exit, check_collisions
from config import Config
from construction.strategies import ALL_STRATEGIES
from construction.helpers import build_operations, empty_bay_entry, block_bbox, find_earliest_slot
from construction.repair import repair_simple, _valid_x_range, _valid_y_range, _candidate_positions
from core.objective import fast_objective
from improvement.parallel import run_parallel_lns, run_multi_start_lns
from improvement.refine import refine_solution, escape_tardiness

def refine_positions(
    assignments: dict[int, dict],
    prob_info: dict,
    bays: list[Bay],
) -> dict[int, dict]:
    blocks_data = prob_info["blocks"]
    n_bays = len(bays)
    bay_placed: list[list[Block]] = [[] for _ in range(n_bays)]
    bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]

    for a in assignments.values():
        bid = a["block_id"]
        bay_id = a["bay_id"]
        bay_placed[bay_id].append(Block(
            block_id=bid, block_data=blocks_data[bid],
            x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
        ))
        bay_schedule[bay_id].append((a["entry_time"], a["exit_time"]))

    current = {bid: dict(a) for bid, a in assignments.items()}

    for bay_id in range(n_bays):
        bay = bays[bay_id]
        placed = bay_placed[bay_id]
        sched = bay_schedule[bay_id]

        order = sorted(range(len(placed)), key=lambda i: sched[i][0])
        for idx in order:
            blk = placed[idx]
            bid = blk.block_id
            blk_data = blocks_data[bid]
            old_entry, old_exit = sched[idx]
            old_x, old_y = blk.x, blk.y
            old_oi = blk.orient_idx

            placed.pop(idx)
            sched.pop(idx)

            best_pos = None

            for oi in range(len(blk_data["shape"])):
                bb = block_bbox(blk_data, oi)
                xr = _valid_x_range(bay.width, bb)
                yr = _valid_y_range(bay.height, bb)
                if xr[0] > xr[1] or yr[0] > yr[1]:
                    continue

                pos_list = []
                if oi == old_oi:
                    pos_list.append((old_x, old_y))
                pos_list.append((xr[0], yr[0]))
                if oi == old_oi:
                    pos_list.append((xr[1], yr[1]))
                else:
                    mid_x = xr[0] + (xr[1] - xr[0]) // 2
                    mid_y = yr[0] + (yr[1] - yr[0]) // 2
                    pos_list.append((mid_x, mid_y))

                for px, py in pos_list:
                    new_blk = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=oi)
                    if not bay.contains_block(new_blk):
                        continue
                    present_entry = [placed[k] for k in range(len(placed))
                                     if sched[k][0] <= old_entry < sched[k][1]]
                    if check_entry(bay, present_entry, new_blk):
                        continue
                    present_exit = [new_blk] + [placed[k] for k in range(len(placed))
                                                if sched[k][0] < old_exit < sched[k][1]]
                    if check_exit(bay, present_exit, new_blk):
                        continue
                    collides = any(
                        check_collisions(bay, [new_blk, placed[k]])
                        for k in range(len(placed))
                        if sched[k][0] < old_exit and old_entry < sched[k][1]
                    )
                    if collides:
                        continue
                    best_pos = (px, py, oi)
                    break
                if best_pos is not None:
                    break

            if best_pos is not None:
                px, py, oi = best_pos
            else:
                px, py, oi = old_x, old_y, old_oi

            final_blk = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=oi)
            placed.insert(idx, final_blk)
            sched.insert(idx, (old_entry, old_exit))
            current[bid] = {
                "block_id": bid, "bay_id": bay_id,
                "x": px, "y": py, "orient_idx": oi,
                "entry_time": int(old_entry), "exit_time": int(old_exit),
            }

    return current


def refine_timing(
    assignments: dict[int, dict],
    prob_info: dict,
    bays: list[Bay],
    blocks_data: list[dict],
    w1: float, w2: float, w3: float,
    t_start: float,
    timelimit: float = 5.0,
) -> dict[int, dict]:
    n_bays = len(bays)
    bay_placed: list[list[Block]] = [[] for _ in range(n_bays)]
    bay_sched: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]

    for a in assignments.values():
        bid = a["block_id"]
        bay_id = a["bay_id"]
        bay_placed[bay_id].append(Block(
            block_id=bid, block_data=blocks_data[bid],
            x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
        ))
        bay_sched[bay_id].append((a["entry_time"], a["exit_time"]))

    current = {bid: dict(a) for bid, a in assignments.items()}

    bbox_cache: dict[tuple[int, int], tuple] = {}
    for bid, blk in enumerate(blocks_data):
        for oi in range(len(blk["shape"])):
            bbox_cache[(bid, oi)] = block_bbox(blk, oi)

    for _pass in range(20):
        if time.time() - t_start >= timelimit:
            break
        improved = False

        late = sorted(
            [(bid, a) for bid, a in current.items()
             if a["exit_time"] - blocks_data[bid]["due_date"] > 0],
            key=lambda x: -(x[1]["exit_time"] - blocks_data[x[0]]["due_date"]),
        )[:20]

        for bid, old_a in late:
            if time.time() - t_start >= timelimit:
                break
            old_bay_id = old_a["bay_id"]
            for i, b in enumerate(bay_placed[old_bay_id]):
                if b.block_id == bid:
                    bay_placed[old_bay_id].pop(i)
                    break
            for i, (entry, et) in enumerate(bay_sched[old_bay_id]):
                if entry == old_a["entry_time"] and et == old_a["exit_time"]:
                    bay_sched[old_bay_id].pop(i)
                    break

            blk_data = blocks_data[bid]
            prefs = blk_data["bay_preferences"]
            r_time = blk_data["release_time"]
            proc = blk_data["processing_time"]
            old_tardy = old_a["exit_time"] - blk_data["due_date"]

            best_fit = None
            best_score = float("inf")
            n_o = len(blk_data["shape"])

            bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
            for bay_id in bay_order:
                bay = bays[bay_id]
                for oi in range(n_o):
                    bb = bbox_cache[(bid, oi)]
                    bw = bb[2] - bb[0]
                    bh = bb[3] - bb[1]
                    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                        continue
                    xr = _valid_x_range(bay.width, bb)
                    yr = _valid_y_range(bay.height, bb)
                    if xr[0] > xr[1] or yr[0] > yr[1]:
                        continue
                    candidates = _candidate_positions(xr, yr)
                    for px, py in candidates:
                        cand_blk = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=oi)
                        if not bay.contains_block(cand_blk):
                            continue
                        slot = find_earliest_slot(
                            cand_blk, bay,
                            bay_placed[bay_id], bay_sched[bay_id],
                            r_time, proc,
                        )
                        if slot[0] is None:
                            continue
                        entry, exit_t = slot
                        tardy = max(0, exit_t - blk_data["due_date"])
                        pref_pen = max(prefs) - prefs[bay_id]
                        score = tardy * w1 + pref_pen * w3
                        if score < best_score:
                            best_score = score
                            best_fit = (bay_id, px, py, oi, entry, exit_t)
                            if score == 0:
                                break
                    if best_fit is not None and best_score == 0:
                        break
                if best_fit is not None and best_score == 0:
                    break

            old_score = max(0, old_tardy) * w1 + (max(prefs) - prefs[old_bay_id]) * w3
            if best_fit is not None:
                bay_id, px, py, oi, entry, exit_t = best_fit
                new_score = max(0, entry + proc - blk_data["due_date"]) * w1 + (max(prefs) - prefs[bay_id]) * w3
                if new_score < old_score:
                    new_blk = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=oi)
                    bay_placed[bay_id].append(new_blk)
                    bay_sched[bay_id].append((entry, exit_t))
                    current[bid] = {
                        "block_id": bid, "bay_id": bay_id,
                        "x": px, "y": py, "orient_idx": oi,
                        "entry_time": entry, "exit_time": exit_t,
                    }
                    improved = True
                    continue

            restored_blk = Block(block_id=bid, block_data=blk_data, x=old_a["x"], y=old_a["y"], orient_idx=old_a["orient_idx"])
            bay_placed[old_bay_id].append(restored_blk)
            bay_sched[old_bay_id].append((old_a["entry_time"], old_a["exit_time"]))

        if not improved:
            break

    return current


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
    feasible_starts: list[tuple[str, dict[int, dict]]] = []

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
            feasible_starts.append((strategy, repaired))
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
        feasible_starts = [("edd_fallback", best_assignments)]

    lns_time_remaining = max(2.0, timelimit - (time.time() - t_start) - 1.0)
    lns_budget = lns_time_remaining * 0.92
    escape_budget = max(3.0, min(8.0, lns_time_remaining * 0.06))

    if timelimit >= 60.0:
        lns_result = run_multi_start_lns(
            prob_info, bays, blocks_data, w1, w2, w3,
            best_assignments, time.time(), lns_budget,
            config, verbose=True,
        )
    else:
        lns_result = run_parallel_lns(
            prob_info, bays, blocks_data, w1, w2, w3,
            best_assignments, time.time(), lns_budget,
            config, num_workers=min(config.num_workers, 2),
            verbose=True,
        )

    final_assignments = lns_result
    lns_safe = {bid: dict(a) for bid, a in lns_result.items()}

    final_assignments = refine_positions(final_assignments, prob_info, bays)
    refine_budget = max(3.0, min(15.0, lns_budget * 0.08))
    final_assignments = refine_timing(
        final_assignments, prob_info, bays, blocks_data, w1, w2, w3,
        t_start, refine_budget,
    )
    final_assignments = refine_positions(final_assignments, prob_info, bays)
    final_assignments = refine_timing(
        final_assignments, prob_info, bays, blocks_data, w1, w2, w3,
        t_start, min(3.0, refine_budget * 0.3),
    )
    # Neighborhood search (swap/move/rotate/time_shift/reassign)
    n_bays = len(bays)
    bay_placed_final = [[] for _ in range(n_bays)]
    bay_sched_final = [[] for _ in range(n_bays)]
    bay_loads_final = [0.0] * n_bays
    for a in final_assignments.values():
        bid = a["block_id"]
        bay_id = a["bay_id"]
        bay_placed_final[bay_id].append(Block(
            block_id=bid, block_data=blocks_data[bid],
            x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
        ))
        bay_sched_final[bay_id].append((a["entry_time"], a["exit_time"]))
        bay_loads_final[bay_id] += blocks_data[bid]["workload"]
    weights_dict = prob_info.get("weights", {})
    final_before_refine = {bid: dict(a) for bid, a in final_assignments.items()}
    final_assignments = refine_solution(
        final_assignments, blocks_data, bays,
        bay_placed_final, bay_sched_final, bay_loads_final,
        bays_data, weights_dict,
        t_start, max(1.0, refine_budget * 0.5),
        random.Random(999),
    )
    check_sol = {"operations": build_operations(list(final_assignments.values()))}
    final_result = check_feasibility(prob_info, check_sol)
    if not final_result.get("feasible", False):
        final_assignments = final_before_refine
    final_sol = {"operations": build_operations(list(final_assignments.values()))}
    result = check_feasibility(prob_info, final_sol)
    elapsed = time.time() - t_start

    if not result.get("feasible", False):
        fallback_obj = best_objective if best_objective is not None else 0
        print(f"[Solver] LNS best not feasible ({result.get('feasible', False)}), "
              f"falling back to construction ({fallback_obj:.0f})")
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
