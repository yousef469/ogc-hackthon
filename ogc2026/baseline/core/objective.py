from __future__ import annotations
from typing import Optional


def fast_objective(
    assignments: dict[int, dict],
    blocks_data: list[dict],
    bays_data: list[dict],
    weights: dict,
) -> dict:
    w1 = weights.get("w1", 1.0)
    w2 = weights.get("w2", 1.0)
    w3 = weights.get("w3", 1.0)

    n_bays = len(bays_data)
    bay_areas = [b["width"] * b["height"] for b in bays_data]
    avg_area = sum(bay_areas) / n_bays if n_bays else 1.0
    bay_weights = [avg_area / max(a, 1) for a in bay_areas]

    bay_loads = [0.0] * n_bays
    obj1 = 0.0
    obj3 = 0.0
    unassigned_penalty = 0.0

    for bid, blk in enumerate(blocks_data):
        a = assignments.get(bid)
        if a is None:
            unassigned_penalty += w1 * blk["due_date"] + w3 * 100
            continue
        obj1 += max(0, a["exit_time"] - blk["due_date"])
        bay_loads[a["bay_id"]] += blk["workload"]

        s_max = max(blk["bay_preferences"])
        obj3 += s_max - blk["bay_preferences"][a["bay_id"]]

    obj2 = 0.0
    if n_bays > 1:
        weighted_loads = [bay_weights[j] * bay_loads[j] for j in range(n_bays)]
        obj2 = max(weighted_loads) - min(weighted_loads)

    objective = w1 * obj1 + w2 * obj2 + w3 * obj3 + unassigned_penalty

    return {
        "feasible": True,
        "objective": objective,
        "obj1": obj1,
        "obj2": obj2,
        "obj3": obj3,
    }
