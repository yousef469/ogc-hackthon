from __future__ import annotations
import time
from typing import Optional

from utils import Bay, Block
from construction.helpers import (
    block_bbox, block_size,
    find_earliest_slot, force_place, empty_bay_entry,
    candidate_positions, placement_score, build_operations,
)
from construction.strategies import ALL_STRATEGIES


def greedy_place_blocks(
    block_ids: list[int],
    blocks_data: list[dict],
    bays: list[Bay],
    bay_placed: list[list[Block]],
    bay_schedule: list[list[tuple[int, int]]],
    bay_loads: list[float],
    w1: float, w2: float, w3: float,
    forced_ids: set[int] | None = None,
    prev_assignments: dict[int, dict] | None = None,
    t_start: float | None = None,
    log_interval: int = 0,
) -> dict[int, dict]:
    if forced_ids is None:
        forced_ids = set()

    n_bays = len(bays)
    n_total = len(block_ids)
    result: dict[int, dict] = {}
    n_forced = 0
    n_fallback = 0

    _bay_areas = [bay.width * bay.height for bay in bays]
    _avg_area = sum(_bay_areas) / n_bays
    bay_weights = [_avg_area / a for a in _bay_areas]

    for rank, bi in enumerate(block_ids):
        blk_data = blocks_data[bi]
        r_time = blk_data["release_time"]
        due = blk_data["due_date"]
        proc = blk_data["processing_time"]
        workload = blk_data["workload"]
        prefs = blk_data["bay_preferences"]
        s_max = max(prefs)
        n_orient = len(blk_data["shape"])

        best_score = float("inf")
        best_placement = None
        used_forced = bi in forced_ids

        if not used_forced:
            if prev_assignments and bi in prev_assignments:
                pa = prev_assignments[bi]
                pb_id = pa["bay_id"]
                px, py, poi = int(pa["x"]), int(pa["y"]), pa["orient_idx"]
                prev_blk = Block(
                    block_id=bi, block_data=blk_data,
                    x=px, y=py, orient_idx=poi,
                )
                if bays[pb_id].contains_block(prev_blk):
                    entry, exit_t = find_earliest_slot(
                        prev_blk, bays[pb_id],
                        bay_placed[pb_id], bay_schedule[pb_id],
                        r_time, proc,
                    )
                    if entry is not None:
                        tardiness = max(0.0, exit_t - due)
                        p_bb = block_bbox(blk_data, poi)
                        best_score = placement_score(
                            tardiness, workload, bay_loads, pb_id,
                            s_max - prefs[pb_id], bay_weights, w1, w2, w3,
                            top_y=py + p_bb[3],
                        )
                        best_placement = (pb_id, px, py, poi, entry, exit_t)

            bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
            for bay_id in bay_order:
                bay = bays[bay_id]
                placed_in_bay = bay_placed[bay_id]
                schedule_in_bay = bay_schedule[bay_id]

                orient_order = sorted(
                    range(n_orient),
                    key=lambda oi: (
                        block_bbox(blk_data, oi)[2] - block_bbox(blk_data, oi)[0]
                    ) * (block_bbox(blk_data, oi)[3] - block_bbox(blk_data, oi)[1])
                )[:4]

                for oi in orient_order:
                    blk_bb = block_bbox(blk_data, oi)
                    bw = blk_bb[2] - blk_bb[0]
                    bh = blk_bb[3] - blk_bb[1]
                    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                        continue

                    candidates = candidate_positions(
                        bay.width, bay.height, placed_in_bay, blk_bb,
                        max_candidates=15,
                    )
                    for cx, cy in candidates:
                        new_blk = Block(
                            block_id=bi, block_data=blk_data,
                            x=cx, y=cy, orient_idx=oi,
                        )
                        if not bay.contains_block(new_blk):
                            continue

                        entry, exit_t = find_earliest_slot(
                            new_blk, bay, placed_in_bay, schedule_in_bay,
                            r_time, proc,
                        )
                        if entry is None:
                            continue

                        tardiness = max(0.0, exit_t - due)
                        score = placement_score(
                            tardiness, workload, bay_loads, bay_id,
                            s_max - prefs[bay_id], bay_weights, w1, w2, w3,
                            top_y=cy + blk_bb[3],
                        )
                        if score < best_score:
                            best_score = score
                            best_placement = (bay_id, cx, cy, oi, entry, exit_t)

        if best_placement is None:
            best_placement = force_place(bi, blocks_data, bays, bay_schedule, prefs)
            n_fallback += 1

        if used_forced:
            n_forced += 1

        bay_id, cx, cy, oi, entry, exit_t = best_placement
        final_blk = Block(
            block_id=bi, block_data=blk_data,
            x=cx, y=cy, orient_idx=oi,
        )
        bay_placed[bay_id].append(final_blk)
        bay_schedule[bay_id].append((entry, exit_t))
        bay_loads[bay_id] += workload

        result[bi] = {
            "block_id": bi,
            "bay_id": bay_id,
            "x": int(round(cx)),
            "y": int(round(cy)),
            "orient_idx": oi,
            "entry_time": int(round(entry)),
            "exit_time": int(round(exit_t)),
        }

    return result


def run_greedy(
    blocks_data: list[dict],
    bays: list[Bay],
    w1: float, w2: float, w3: float,
    strategy: str = "edd",
    forced_ids: set[int] | None = None,
) -> tuple[dict[int, dict], list[list[Block]], list[list[tuple[int, int]]], list[float]]:
    n_bays = len(bays)
    bay_placed: list[list[Block]] = [[] for _ in range(n_bays)]
    bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
    bay_loads: list[float] = [0.0] * n_bays

    order_fn = ALL_STRATEGIES.get(strategy, ALL_STRATEGIES["edd"])
    block_ids = order_fn(blocks_data)

    assignments = greedy_place_blocks(
        block_ids, blocks_data, bays,
        bay_placed, bay_schedule, bay_loads,
        w1, w2, w3,
        forced_ids=forced_ids or set(),
    )
    return assignments, bay_placed, bay_schedule, bay_loads
