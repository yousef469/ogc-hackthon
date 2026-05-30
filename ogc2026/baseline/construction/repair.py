from __future__ import annotations
import time
from typing import Optional

from utils import Bay, Block, check_feasibility
from construction.helpers import build_operations, empty_bay_entry
from construction.greedy import greedy_place_blocks


def repair_greedy(
    prob_info: dict,
    sol: dict,
    assignments: dict[int, dict],
    bays: list[Bay],
    blocks_data: list[dict],
    w1: float, w2: float, w3: float,
    t_start: float,
    timelimit: float,
    max_passes: int = 10,
) -> dict[int, dict]:
    repaired_counts: dict[int, int] = {}
    forced_ids: set[int] = set()

    for pass_idx in range(max_passes):
        if time.time() - t_start > timelimit * 0.98:
            break

        result = check_feasibility(prob_info, sol)
        if result["feasible"]:
            break

        viols = result["violations"]

        to_repair: list[int] = []
        seen: set[int] = set()
        for v in viols:
            try:
                bid = int(v.split("block ")[1].split()[0])
                if bid not in seen:
                    seen.add(bid)
                    to_repair.append(bid)
            except (IndexError, ValueError):
                pass

        if not to_repair:
            break

        to_repair.sort(key=lambda b: (
            blocks_data[b]["due_date"],
            blocks_data[b]["processing_time"],
        ))

        for bid in to_repair:
            repaired_counts[bid] = repaired_counts.get(bid, 0) + 1
            if repaired_counts[bid] > 1:
                forced_ids.add(bid)

        for bid in to_repair:
            assignments.pop(bid, None)

        n_bays = len(bays)
        bay_placed: list[list[Block]] = [[] for _ in range(n_bays)]
        bay_schedule2: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
        bay_loads: list[float] = [0.0] * n_bays

        for a in assignments.values():
            bid_a = a["block_id"]
            bay_id = a["bay_id"]
            blk = Block(
                block_id=bid_a, block_data=blocks_data[bid_a],
                x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
            )
            bay_placed[bay_id].append(blk)
            bay_schedule2[bay_id].append((a["entry_time"], a["exit_time"]))
            bay_loads[bay_id] += blocks_data[bid_a]["workload"]

        for bi in to_repair:
            if time.time() - t_start > timelimit * 0.90:
                forced_ids.add(bi)
            partial = greedy_place_blocks(
                [bi], blocks_data, bays,
                bay_placed, bay_schedule2, bay_loads,
                w1, w2, w3, forced_ids,
                prev_assignments=assignments,
            )
            assignments.update(partial)
            new_a = partial[bi]
            bay_placed[new_a["bay_id"]].append(
                Block(
                    block_id=bi, block_data=blocks_data[bi],
                    x=int(new_a["x"]), y=int(new_a["y"]),
                    orient_idx=new_a["orient_idx"],
                )
            )
            bay_schedule2[new_a["bay_id"]].append(
                (new_a["entry_time"], new_a["exit_time"])
            )
            bay_loads[new_a["bay_id"]] += blocks_data[bi]["workload"]

        sol = {"operations": build_operations(list(assignments.values()))}

    return assignments


def repair_simple(
    prob_info: dict,
    assignments: dict[int, dict],
    bays: list[Bay],
    blocks_data: list[dict],
) -> dict[int, dict]:
    from utils import check_feasibility

    sol = {"operations": build_operations(list(assignments.values()))}
    result = check_feasibility(prob_info, sol)
    if result["feasible"]:
        return assignments

    viols = result["violations"]
    to_repair: list[int] = []
    seen: set[int] = set()
    for v in viols:
        try:
            bid = int(v.split("block ")[1].split()[0])
            if bid not in seen:
                seen.add(bid)
                to_repair.append(bid)
        except (IndexError, ValueError):
            pass

    n_bays = len(bays)
    bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
    for a in assignments.values():
        bay_schedule[a["bay_id"]].append((a["entry_time"], a["exit_time"]))

    for bid in to_repair:
        a = assignments[bid]
        bay_id = a["bay_id"]
        r_time = blocks_data[bid]["release_time"]
        proc = blocks_data[bid]["processing_time"]

        old_slot = (a["entry_time"], a["exit_time"])
        if old_slot in bay_schedule[bay_id]:
            bay_schedule[bay_id].remove(old_slot)

        entry = empty_bay_entry(bay_schedule[bay_id], r_time, proc)
        exit_t = entry + proc

        x, y, oi = a["x"], a["y"], a["orient_idx"]
        if result["stage"] == 4:
            x, y = 0, 0

        assignments[bid] = dict(
            a, x=x, y=y, orient_idx=oi,
            entry_time=int(round(entry)),
            exit_time=int(round(exit_t)),
        )
        bay_schedule[bay_id].append((entry, exit_t))

    return assignments
