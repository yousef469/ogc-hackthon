from __future__ import annotations
import math
import random
import time
from typing import Optional

from utils import Bay, Block
from construction.helpers import block_bbox, find_earliest_slot, empty_bay_entry
from construction.repair import _valid_x_range, _valid_y_range
from improvement.local_search import (
    try_swap_blocks, try_move_block, try_rotate_block,
    try_time_shift, try_reassign_bay,
)
from core.objective import fast_objective


def refine_solution(
    assignments: dict[int, dict],
    blocks_data: list[dict],
    bays: list[Bay],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    bays_data: list[dict],
    weights_dict: dict,
    t_start: float,
    timelimit: float,
    rng: random.Random,
    verbose: bool = False,
) -> dict[int, dict]:
    n_bays = len(bays)
    n_blocks = len(blocks_data)

    current = {bid: dict(a) for bid, a in assignments.items()}
    obj_result = fast_objective(current, blocks_data, bays_data, weights_dict)
    best_obj = obj_result["objective"]
    best = current

    n_improved = 0
    n_attempted = 0
    n_iterations = 0
    deadline = time.time() + timelimit

    bay_block_ids: dict[int, list[int]] = {j: [] for j in range(n_bays)}
    for bid, a in current.items():
        bay_block_ids[a["bay_id"]].append(bid)

    while time.time() < deadline:
        n_iterations += 1

        if not any(bay_block_ids.values()):
            break

        bay_id = rng.choice([j for j, ids in bay_block_ids.items() if ids])
        ids_in_bay = bay_block_ids[bay_id]
        if len(ids_in_bay) < 2:
            continue

        bid_a = rng.choice(ids_in_bay)
        bid_b = rng.choice([b for b in ids_in_bay if b != bid_a])

        n_attempted += 1
        old_obj = fast_objective(current, blocks_data, bays_data, weights_dict)["objective"]

        success = False
        op_type = rng.choice(["time_shift", "reassign", "swap", "move", "rotate"])

        if op_type == "time_shift":
            a = current[bid_a]
            blk_data = blocks_data[bid_a]
            r_proc = blk_data["processing_time"]
            r_time = blk_data["release_time"]
            bay = bays[a["bay_id"]]
            cand_blk = Block(
                block_id=bid_a, block_data=blk_data,
                x=a["x"], y=a["y"], orient_idx=a["orient_idx"],
            )
            slot = find_earliest_slot(
                cand_blk, bay, bay_placed[a["bay_id"]],
                bay_schedule[a["bay_id"]], r_time, r_proc,
            )
            if slot[0] is not None and slot[0] < a["entry_time"]:
                success = try_time_shift(
                    bid_a, slot[0], current, bay_placed,
                    bay_schedule, bay_loads, blocks_data, bays,
                )
        elif op_type == "reassign":
            a = current[bid_a]
            blk_data = blocks_data[bid_a]
            prefs = blk_data["bay_preferences"]
            n_o = len(blk_data["shape"])
            best_bay = None
            best_oi = None
            best_slot = None
            for bj in sorted(range(n_bays), key=lambda j: prefs[j], reverse=True):
                if bj == a["bay_id"]:
                    continue
                bay = bays[bj]
                for oi in range(n_o):
                    bb = block_bbox(blk_data, oi)
                    bw = bb[2] - bb[0]
                    bh = bb[3] - bb[1]
                    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                        continue
                    xr = _valid_x_range(bay.width, bb)
                    yr = _valid_y_range(bay.height, bb)
                    if xr[0] > xr[1] or yr[0] > yr[1]:
                        continue
                    px = xr[0]
                    py = yr[0]
                    cand_blk = Block(block_id=bid_a, block_data=blk_data, x=px, y=py, orient_idx=oi)
                    slot = find_earliest_slot(
                        cand_blk, bay, bay_placed[bj],
                        bay_schedule[bj], blk_data["release_time"],
                        blk_data["processing_time"],
                    )
                    if slot[0] is not None:
                        if best_slot is None or slot[0] < best_slot[0]:
                            best_slot = (slot[0], slot[1])
                            best_bay = bj
                            best_oi = oi
                            best_x, best_y = px, py
            if best_bay is not None:
                success = try_reassign_bay(
                    bid_a, best_bay, best_x, best_y, best_oi,
                    best_slot[0], current, bay_placed,
                    bay_schedule, bay_loads, blocks_data, bays,
                )
        elif op_type == "swap":
            success = try_swap_blocks(
                bid_a, bid_b, current, bay_placed, bay_schedule, bay_loads,
                blocks_data, bays,
            )
        elif op_type == "move":
            a = current[bid_a]
            bb = block_bbox(blocks_data[bid_a], a["orient_idx"])
            dx = rng.randint(-5, 5)
            dy = rng.randint(-5, 5)
            new_x = max(0, a["x"] + dx)
            new_y = max(0, a["y"] + dy)
            if (new_x, new_y) != (a["x"], a["y"]):
                success = try_move_block(
                    bid_a, new_x, new_y, current, bay_placed, bay_schedule,
                    bay_loads, blocks_data, bays,
                )
        elif op_type == "rotate":
            a = current[bid_a]
            blk_data = blocks_data[bid_a]
            n_o = len(blk_data["shape"])
            if n_o > 1:
                new_o = rng.choice([oi for oi in range(n_o) if oi != a["orient_idx"]])
                success = try_rotate_block(
                    bid_a, new_o, current, bay_placed, bay_schedule,
                    bay_loads, blocks_data, bays,
                )

        if success:
            new_obj = fast_objective(current, blocks_data, bays_data, weights_dict)["objective"]
            if new_obj < old_obj:
                n_improved += 1
                if new_obj < best_obj:
                    best_obj = new_obj
                    best = {bid: dict(a) for bid, a in current.items()}
                    if verbose:
                        print(f"[Refine] improved to {best_obj:.0f}  ({op_type})")
            else:
                if op_type == "swap":
                    try_swap_blocks(
                        bid_a, bid_b, current, bay_placed, bay_schedule,
                        bay_loads, blocks_data, bays,
                    )
                elif op_type == "move":
                    a = current[bid_a]
                    try_move_block(
                        bid_a, a["x"] - dx, a["y"] - dy,
                        current, bay_placed, bay_schedule,
                        bay_loads, blocks_data, bays,
                    )
                elif op_type == "rotate":
                    a = current[bid_a]
                    try_rotate_block(
                        bid_a, a["orient_idx"], current, bay_placed,
                        bay_schedule, bay_loads, blocks_data, bays,
                    )

        if n_iterations % 500 == 0 and verbose:
            elapsed = time.time() - (deadline - timelimit)
            print(f"[Refine] iter={n_iterations}  best={best_obj:.0f}  "
                  f"improved={n_improved}  elapsed={elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"[Refine] Done  iterations={n_iterations}  "
              f"improved={n_improved}  best={best_obj:.0f}  "
              f"elapsed={elapsed:.1f}s")

    return best
