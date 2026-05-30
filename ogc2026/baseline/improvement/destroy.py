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


ALL_DESTROY_OPERATORS = [
    ("random", random_removal),
    ("worst", worst_removal),
    ("related", related_removal),
    ("bay", bay_removal),
    ("time_window", time_window_removal),
]
