from __future__ import annotations
import math
import random
from typing import Optional


def random_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    block_ids = list(assignments.keys())
    return rng.sample(block_ids, min(k, len(block_ids)))


def worst_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    scored = []
    for bid, a in assignments.items():
        tardiness = max(0, a["exit_time"] - blocks_data[bid]["due_date"])
        scored.append((-tardiness, bid))
    scored.sort()
    return [bid for _, bid in scored[:k]]


def related_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    seed = rng.choice(list(assignments.keys()))
    removed = {seed}
    seed_a = assignments[seed]

    others = []
    for bid, a in assignments.items():
        if bid in removed:
            continue
        same_bay = 1 if a["bay_id"] == seed_a["bay_id"] else 0
        time_dist = abs(a["entry_time"] - seed_a["entry_time"])
        others.append((1 - same_bay, time_dist, bid))

    others.sort()
    for _, _, bid in others[: k - 1]:
        removed.add(bid)

    return list(removed)


def bay_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    by_bay: dict[int, list[int]] = {}
    for bid, a in assignments.items():
        by_bay.setdefault(a["bay_id"], []).append(bid)

    target_bay = rng.choice(list(by_bay.keys()))
    candidates = by_bay[target_bay]
    return rng.sample(candidates, min(k, len(candidates)))


def cross_bay_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    by_bay: dict[int, list[int]] = {}
    for bid, a in assignments.items():
        by_bay.setdefault(a["bay_id"], []).append(bid)
    if len(by_bay) < 2:
        return random_removal(assignments, blocks_data, k, rng)
    bay_a, bay_b = rng.sample(list(by_bay.keys()), 2)
    candidates = by_bay[bay_a] + by_bay[bay_b]
    return rng.sample(candidates, min(k, len(candidates)))


def time_window_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    all_times = [a["entry_time"] for a in assignments.values()]
    t_min = min(all_times)
    t_max = max(a["exit_time"] for a in assignments.values())
    if t_max <= t_min:
        return random_removal(assignments, blocks_data, k, rng)

    window_start = rng.randint(t_min, t_max)
    window_end = window_start + max(1, (t_max - t_min) // 4)

    candidates = [
        bid for bid, a in assignments.items()
        if a["entry_time"] < window_end and a["exit_time"] > window_start
    ]
    if len(candidates) <= 1:
        return random_removal(assignments, blocks_data, k, rng)
    return rng.sample(candidates, min(k, len(candidates)))


def critical_path_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    scored = []
    for bid, a in assignments.items():
        blk = blocks_data[bid]
        tardy = max(0, a["exit_time"] - blk["due_date"])
        proc = max(blk["processing_time"], 1)
        criticality = tardy * (blk.get("workload", 1) / proc)
        scored.append((criticality, bid))
    scored.sort(reverse=True)
    return [bid for _, bid in scored[:k]]


def spatial_cluster_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    seed = rng.choice(list(assignments.keys()))
    seed_a = assignments[seed]
    removed = {seed}

    by_bay: dict[int, list[tuple[int, int, int]]] = {}
    for bid, a in assignments.items():
        by_bay.setdefault(a["bay_id"], []).append((a["x"], a["y"], bid))

    candidates = by_bay.get(seed_a["bay_id"], [])
    if len(candidates) <= 1:
        return worst_removal(assignments, blocks_data, k, rng)

    sx, sy = seed_a["x"], seed_a["y"]
    candidates.sort(key=lambda t: (t[0] - sx) ** 2 + (t[1] - sy) ** 2)
    for _, _, bid in candidates:
        removed.add(bid)
        if len(removed) >= k:
            break
    return list(removed)[:k]


def due_date_window_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    due_dates = [blk["due_date"] for blk in blocks_data]
    dd_min = min(due_dates)
    dd_max = max(due_dates)
    if dd_max <= dd_min:
        return random_removal(assignments, blocks_data, k, rng)

    pivot = rng.randint(dd_min, dd_max)
    half_width = max(1, (dd_max - dd_min) // 6)

    candidates = [
        bid for bid in assignments
        if abs(blocks_data[bid]["due_date"] - pivot) <= half_width
    ]
    if len(candidates) <= 1:
        return random_removal(assignments, blocks_data, k, rng)
    return rng.sample(candidates, min(k, len(candidates)))


def tardy_blaster_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    tardy = sorted(
        [(bid, a) for bid, a in assignments.items()
         if a["exit_time"] - blocks_data[bid]["due_date"] > 0],
        key=lambda x: -(x[1]["exit_time"] - blocks_data[x[0]]["due_date"]),
    )
    removed: set[int] = set()
    blocked_by: dict[int, set[int]] = {}

    for bid, a in tardy:
        removed.add(bid)
        if len(removed) >= k:
            break
        bay_id = a["bay_id"]
        entry, exit_t = a["entry_time"], a["exit_time"]
        for obid, ob in assignments.items():
            if obid == bid or obid in removed:
                continue
            if ob["bay_id"] != bay_id:
                continue
            if ob["entry_time"] < exit_t and entry < ob["exit_time"]:
                blocked_by.setdefault(bid, set()).add(obid)

    for bid in list(removed):
        blockers = blocked_by.get(bid, set())
        for blk_id in blockers:
            if len(removed) >= k:
                break
            removed.add(blk_id)

    remaining = [b for b in assignments if b not in removed]
    if remaining and len(removed) < k:
        removed.update(rng.sample(remaining, min(k - len(removed), len(remaining))))

    return list(removed)[:k]





def precedence_chain_removal(
    assignments: dict,
    blocks_data: list[dict],
    k: int,
    rng: random.Random,
) -> list[int]:
    if not assignments:
        return []
    by_bay_sched: dict[int, list[tuple[int, int, int]]] = {}
    for bid, a in assignments.items():
        by_bay_sched.setdefault(a["bay_id"], []).append(
            (a["entry_time"], a["exit_time"], bid)
        )

    all_chains: list[int] = []
    for bay_id, entries in by_bay_sched.items():
        if len(entries) < 3:
            continue
        entries.sort()
        for i in range(len(entries) - 1):
            gap = entries[i + 1][0] - entries[i][1]
            if gap < 5:
                all_chains.append(entries[i][2])
                all_chains.append(entries[i + 1][2])

    if not all_chains:
        all_chains = list(assignments.keys())
    uniq = list(set(all_chains))
    if len(uniq) <= k:
        return uniq[:k]
    return rng.sample(uniq, k)


ALL_DESTROY_OPERATORS = [
    ("random", random_removal),
    ("worst", worst_removal),
    ("related", related_removal),
    ("bay", bay_removal),
    ("cross_bay", cross_bay_removal),
    ("time_window", time_window_removal),
    ("critical_path", critical_path_removal),
    ("spatial_cluster", spatial_cluster_removal),
    ("due_date_window", due_date_window_removal),
    ("precedence_chain", precedence_chain_removal),
    ("tardy_blaster", tardy_blaster_removal),
]
