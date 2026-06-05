import math
import time

from utils import Bay, Block, check_feasibility
from construction.helpers import build_operations, empty_bay_entry, block_bbox


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
    from construction.greedy import greedy_place_blocks

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


def _valid_x_range(bay_width: float, bb: tuple) -> tuple[int, int]:
    local_min_x, _, local_max_x, _ = bb
    min_x = math.ceil(max(0.0, -local_min_x + 1e-9))
    max_x = math.floor(bay_width - local_max_x - 1e-9)
    if min_x > max_x:
        return (0, -1)
    return (min_x, max_x)


def _valid_y_range(bay_height: float, bb: tuple) -> tuple[int, int]:
    _, local_min_y, _, local_max_y = bb
    min_y = math.ceil(max(0.0, -local_min_y + 1e-9))
    max_y = math.floor(bay_height - local_max_y - 1e-9)
    if min_y > max_y:
        return (0, -1)
    return (min_y, max_y)


def _candidate_positions(xr: tuple[int, int], yr: tuple[int, int]) -> list[tuple[int, int]]:
    xmin, xmax = xr
    ymin, ymax = yr
    pos = [(xmin, ymin)]
    if xmax > xmin:
        pos.append((xmax, ymin))
    if ymax > ymin:
        pos.append((xmin, ymax))
    if xmax > xmin and ymax > ymin:
        pos.append((xmin + (xmax - xmin) // 2, ymin + (ymax - ymin) // 2))
    return pos


def repair_simple(
    prob_info: dict,
    assignments: dict[int, dict],
    bays: list[Bay],
    blocks_data: list[dict],
    max_passes: int = 20,
) -> dict[int, dict]:
    for pass_idx in range(max_passes):
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

        if not to_repair:
            return assignments

        n_bays = len(bays)
        bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
        for a in assignments.values():
            bay_schedule[a["bay_id"]].append((a["entry_time"], a["exit_time"]))

        for bid in to_repair:
            a = assignments[bid]
            r_time = blocks_data[bid]["release_time"]
            proc = blocks_data[bid]["processing_time"]
            prefs = blocks_data[bid]["bay_preferences"]
            n_o = len(blocks_data[bid]["shape"])

            old_slot = (a["entry_time"], a["exit_time"])
            for bj_list in bay_schedule:
                if old_slot in bj_list:
                    bj_list.remove(old_slot)
                    break

            best = None
            for bj in sorted(range(n_bays), key=lambda j: prefs[j], reverse=True):
                bay = bays[bj]
                for oi in range(n_o):
                    bb = block_bbox(blocks_data[bid], oi)
                    xr = _valid_x_range(bay.width, bb)
                    yr = _valid_y_range(bay.height, bb)
                    if xr[0] > xr[1] or yr[0] > yr[1]:
                        continue
                    for px, py in _candidate_positions(xr, yr):
                        entry = empty_bay_entry(bay_schedule[bj], r_time, proc)
                        if entry is not None:
                            best = (bj, px, py, oi, int(entry), int(entry + proc))
                            break
                    if best:
                        break
                if best:
                    break

            if not best:
                bj = max(range(n_bays), key=lambda j: prefs[j])
                bay = bays[bj]
                found = False
                for oi in range(n_o):
                    bb = block_bbox(blocks_data[bid], oi)
                    xr = _valid_x_range(bay.width, bb)
                    yr = _valid_y_range(bay.height, bb)
                    if xr[0] > xr[1] or yr[0] > yr[1]:
                        continue
                    step = max(5, int(min(xr[1] - xr[0], yr[1] - yr[0]) ** 0.5) + 1)
                    for px in range(xr[0], xr[1] + 1, step):
                        for py in range(yr[0], yr[1] + 1, step):
                            entry = empty_bay_entry(bay_schedule[bj], r_time, proc)
                            if entry is not None:
                                best = (bj, px, py, oi,
                                        int(entry), int(entry + proc))
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    entry = empty_bay_entry(bay_schedule[bj], r_time, proc)
                    best = (bj, 0, 0, 0, int(entry), int(entry + proc))

            bj, px, py, oi, entry, exit_t = best
            assignments[bid] = dict(
                a, bay_id=bj, x=px, y=py, orient_idx=oi,
                entry_time=entry, exit_time=exit_t,
            )
            bay_schedule[bj].append((entry, exit_t))

    return assignments
