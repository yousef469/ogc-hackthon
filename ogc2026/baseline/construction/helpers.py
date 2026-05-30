import math
from utils import Bay, Block, check_entry, check_exit, check_collisions, _resolve_layers, _bounding_box


def block_bbox(block_data: dict, orient_idx: int) -> tuple[float, float, float, float]:
    raw_layers = block_data["shape"][orient_idx]["layers"]
    layers = _resolve_layers(raw_layers)
    if not layers:
        return (0.0, 0.0, 1.0, 1.0)
    all_verts = [v for l in layers for v in l]
    return _bounding_box(all_verts)


def block_size(block_data: dict, orient_idx: int) -> tuple[float, float]:
    bb = block_bbox(block_data, orient_idx)
    return (bb[2] - bb[0], bb[3] - bb[1])


def time_overlaps(a_entry: int, a_exit: int,
                  b_entry: int, b_exit: int) -> bool:
    return a_entry < b_exit and b_entry < a_exit


def candidate_positions(bay_w: float, bay_h: float,
                        placed_blocks: list[Block],
                        blk_bb: tuple[float, float, float, float],
                        max_candidates: int = 30) -> list[tuple[int, int]]:
    lx0, ly0, lx1, ly1 = blk_bb
    xs = {max(0, math.ceil(-lx0))}
    ys = {max(0, math.ceil(-ly0))}
    for b in placed_blocks:
        bb = b.bounding_rect()
        xs.add(math.ceil(bb[2] - lx0))
        ys.add(math.ceil(bb[3] - ly0))
    candidates = []
    for x in sorted(xs):
        for y in sorted(ys):
            if x + lx1 <= bay_w + 1e-6 and y + ly1 <= bay_h + 1e-6:
                candidates.append((int(x), int(y)))
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    return candidates


def placement_score(tardiness: float, workload: float,
                    bay_loads: list[float], bay_id: int,
                    pref_penalty: float,
                    bay_weights: list[float],
                    w1: float, w2: float, w3: float,
                    top_y: float = 0.0, w4: float = 1e-4) -> float:
    new_load = bay_loads[bay_id] + workload
    new_obj2 = max(
        (abs(bay_weights[bay_id] * new_load - bay_weights[j] * bay_loads[j])
         for j in range(len(bay_loads)) if j != bay_id),
        default=0.0
    )
    return w1 * tardiness + w2 * new_obj2 + w3 * pref_penalty + w4 * top_y


def find_earliest_slot(new_blk: Block,
                       bay: Bay,
                       placed_in_bay: list[Block],
                       schedule_in_bay: list[tuple[int, int]],
                       r_time: int,
                       proc: int) -> tuple[int | None, int | None]:
    def _overlaps(ae, ee, ao, eo):
        return ae < eo and ao < ee

    candidate_entries = sorted({r_time} | {e for _, e in schedule_in_bay})

    for entry_candidate in candidate_entries:
        entry = max(r_time, entry_candidate)
        exit_t = entry + proc

        present_at_entry = [
            b for b, (a, e) in zip(placed_in_bay, schedule_in_bay)
            if a <= entry < e
        ]
        if check_entry(bay, present_at_entry, new_blk, fast=True):
            continue

        present_at_exit = [new_blk] + [
            b for b, (a, e) in zip(placed_in_bay, schedule_in_bay)
            if a < exit_t < e
        ]
        if check_exit(bay, present_at_exit, new_blk, fast=True):
            continue

        s4_blocked = False
        for b_other, (a_other, e_other) in zip(placed_in_bay, schedule_in_bay):
            if a_other <= entry or e_other >= exit_t:
                continue
            if not _overlaps(entry, exit_t, a_other, e_other):
                continue
            if check_collisions(bay, [new_blk, b_other]):
                s4_blocked = True
                break
        if s4_blocked:
            continue

        return entry, exit_t

    return None, None


def empty_bay_entry(schedule_in_bay: list[tuple[int, int]],
                    r_time: int, proc: int) -> int:
    entry = int(r_time)
    if not schedule_in_bay:
        return entry
    sorted_sched = sorted(schedule_in_bay)
    for a, e in sorted_sched:
        if entry >= e:
            continue
        if entry + proc <= a:
            return entry
        entry = e
    return entry


def force_place(bi: int,
                blocks_data: list[dict],
                bays: list[Bay],
                bay_schedule: list[list[tuple[int, int]]],
                prefs: list[float]) -> tuple:
    blk_data = blocks_data[bi]
    r_time = blk_data["release_time"]
    proc = blk_data["processing_time"]
    n_bays = len(bays)

    for bay_id in sorted(range(n_bays), key=lambda j: prefs[j], reverse=True):
        bay = bays[bay_id]
        for oi in range(len(blk_data["shape"])):
            bw, bh = block_size(blk_data, oi)
            if bw <= bay.width + 1e-6 and bh <= bay.height + 1e-6:
                bb = block_bbox(blk_data, oi)
                px = max(0, math.ceil(-bb[0]))
                py = max(0, math.ceil(-bb[1]))
                entry = empty_bay_entry(bay_schedule[bay_id], r_time, proc)
                return (bay_id, px, py, oi, entry, entry + proc)

    bay_id = max(range(n_bays), key=lambda j: prefs[j])
    bb = block_bbox(blk_data, 0)
    px = max(0, math.ceil(-bb[0]))
    py = max(0, math.ceil(-bb[1]))
    entry = empty_bay_entry(bay_schedule[bay_id], r_time, proc)
    return (bay_id, px, py, 0, entry, entry + proc)


def build_operations(assignments: list[dict]) -> dict:
    buckets: dict[int, list[tuple]] = {}
    for a in assignments:
        t_entry = int(a["entry_time"])
        t_exit = int(a["exit_time"])
        bid = a["block_id"]
        bay = a["bay_id"]
        buckets.setdefault(t_exit, []).append(
            (0, "EXIT", bid, bay, None, None, None)
        )
        buckets.setdefault(t_entry, []).append(
            (1, "ENTRY", bid, bay, a["x"], a["y"], a["orient_idx"])
        )
    operations: dict[str, list[dict]] = {}
    for t in sorted(buckets):
        ops = sorted(buckets[t], key=lambda x: (x[0], x[2]))
        result = []
        for _, kind, bid, bay, x, y, orient_idx in ops:
            op: dict = {"type": kind, "block_id": bid, "bay_id": bay}
            if kind == "ENTRY":
                op["x"] = x
                op["y"] = y
                op["orient_idx"] = orient_idx
            result.append(op)
        operations[str(t)] = result
    return operations
