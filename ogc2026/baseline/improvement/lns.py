from __future__ import annotations
import math
import random
import time

from utils import Bay, Block, check_feasibility, check_entry, check_exit, check_collisions, _bb_overlap
from config import Config
from construction.helpers import build_operations, block_bbox, empty_bay_entry, find_earliest_slot
from construction.repair import _valid_x_range, _valid_y_range, _candidate_positions
from improvement.destroy import ALL_DESTROY_OPERATORS
from improvement.acceptance import SimulatedAnnealing
from core.objective import fast_objective


def beam_search_rebuild(
    destroyed_order: list[int],
    old_of_removed: dict[int, dict],
    base_assignments: dict[int, dict],
    base_bay_placed: list[list[Block]],
    base_bay_schedule: list[list[tuple[int, int]]],
    base_bay_loads: list[float],
    blocks_data: list[dict],
    bays: list[Bay],
    bays_data: list[dict],
    weights_dict: dict,
    n_bays: int,
    bbox_cache: dict,
    w1: float, w2: float, w3: float,
    bay_weights: list[float],
    beam_width: int,
    options_per_block: int,
    rng: random.Random,
) -> dict[int, dict]:
    beam: list[tuple[dict[int, dict], list[float], list[list[tuple[int, int]]], float]] = [
        (dict(base_assignments), list(base_bay_loads),
         [list(s) for s in base_bay_schedule], 0.0)
    ]

    for bid in destroyed_order:
        blk_data = blocks_data[bid]
        old_a = old_of_removed.get(bid)
        prefs = blk_data["bay_preferences"]
        r_time = blk_data["release_time"]
        proc = blk_data["processing_time"]
        workload = blk_data["workload"]
        n_o = len(blk_data["shape"])

        new_beam: list = []
        for assigns, loads, schedule, score in beam:
            best_options: list[tuple] = []
            for bay_id in range(n_bays):
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

                    positions = [(xr[0], yr[0])]
                    if old_a and bay_id == old_a["bay_id"] and oi == old_a["orient_idx"]:
                        ox, oy = int(old_a["x"]), int(old_a["y"])
                        if xr[0] <= ox <= xr[1] and yr[0] <= oy <= yr[1]:
                            positions.insert(0, (ox, oy))
                    if xr[1] > xr[0] and yr[1] > yr[0]:
                        positions.append(((xr[0] + xr[1]) // 2, (yr[0] + yr[1]) // 2))

                    for px, py in positions:
                        entry = empty_bay_entry(schedule[bay_id], r_time, proc)
                        if entry is None:
                            continue
                        exit_t = entry + proc
                        tardy = max(0, exit_t - blk_data["due_date"])
                        pref_pen = max(prefs) - prefs[bay_id]
                        delta = 0.0
                        if n_bays > 1 and w2 > 0:
                            new_loads = list(loads)
                            new_loads[bay_id] += workload
                            old_w = [bay_weights[j] * loads[j] for j in range(n_bays)]
                            new_w = [bay_weights[j] * new_loads[j] for j in range(n_bays)]
                            delta = (max(new_w) - min(new_w)) - (max(old_w) - min(old_w))
                        opt_score = tardy * w1 + delta * w2 + pref_pen * w3
                        best_options.append((opt_score, bay_id, px, py, oi, entry, exit_t))
            best_options.sort(key=lambda x: x[0])
            for opt_score, bay_id, px, py, oi, entry, exit_t in best_options[:options_per_block]:
                new_assigns = dict(assigns)
                new_assigns[bid] = {
                    "block_id": bid, "bay_id": bay_id,
                    "x": px, "y": py, "orient_idx": oi,
                    "entry_time": entry, "exit_time": exit_t,
                }
                new_loads = list(loads)
                new_loads[bay_id] += workload
                new_schedule = [list(s) for s in schedule]
                new_schedule[bay_id].append((entry, exit_t))
                new_beam.append((new_assigns, new_loads, new_schedule, score + opt_score))

        if not new_beam:
            continue
        new_beam.sort(key=lambda x: x[3])
        beam = new_beam[:beam_width]

    return beam[0][0] if beam else base_assignments


def run_lns(
    prob_info: dict,
    bays: list[Bay],
    blocks_data: list[dict],
    w1: float, w2: float, w3: float,
    initial_assignments: dict[int, dict],
    t_start: float,
    timelimit: float,
    config: Config,
    seed: int = 0,
    verbose: bool = False,
) -> dict[int, dict]:
    rng = random.Random(seed)
    lns_start = time.time()
    n_bays = len(bays)
    n_blocks = len(blocks_data)
    bays_data = prob_info["bays"]
    weights_dict = prob_info.get("weights", {})

    bay_areas = [b["width"] * b["height"] for b in bays_data]
    avg_area = sum(bay_areas) / n_bays if n_bays else 1.0
    bay_weights = [avg_area / max(a, 1) for a in bay_areas]

    bay_placed: list[list[Block]] = [[] for _ in range(n_bays)]
    bay_schedule: list[list[tuple[int, int]]] = [[] for _ in range(n_bays)]
    bay_loads: list[float] = [0.0] * n_bays

    bbox_cache: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for bid, blk in enumerate(blocks_data):
        for oi in range(len(blk["shape"])):
            bbox_cache[(bid, oi)] = block_bbox(blk, oi)

    for a in initial_assignments.values():
        bid = a["block_id"]
        bay_id = a["bay_id"]
        bay_placed[bay_id].append(Block(
            block_id=bid, block_data=blocks_data[bid],
            x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
        ))
        bay_schedule[bay_id].append((a["entry_time"], a["exit_time"]))
        bay_loads[bay_id] += blocks_data[bid]["workload"]

    current_assignments = {bid: dict(a) for bid, a in initial_assignments.items()}
    best_assignments = {bid: dict(a) for bid, a in initial_assignments.items()}

    obj_result = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)
    current_obj = obj_result["objective"]
    best_obj = current_obj

    sa = SimulatedAnnealing(
        initial_temp=config.sa_initial_temperature,
        cooling_rate=config.sa_cooling_rate,
    )
    cooling_power = config.sa_cooling_power

    if verbose:
        print(f"[LNS] seed={seed}  initial={current_obj:.0f}  "
              f"(tardy={obj_result['obj1']:.0f}  "
              f"bal={obj_result['obj2']:.1f}  "
              f"pref={obj_result['obj3']:.0f})")

    n_iterations = 0
    n_accepted = 0
    n_improved = 0
    stale = 0
    max_stale = max(100000, int(timelimit * 2000))
    deadline = time.time() + timelimit
    log_interval = max(100, 500 // (config.num_workers if config.use_parallel else 1))
    full_check_interval = 50

    n_operators = len(ALL_DESTROY_OPERATORS)
    op_weights = [1.0] * n_operators
    op_scores = [0.0] * n_operators
    op_counts = [0] * n_operators
    weight_update_interval = max(20, 100 // (config.num_workers if config.use_parallel else 1))

    def _select_operator() -> tuple[str, callable, int]:
        if rng.random() < 0.1 or sum(op_weights) < 1e-10:
            idx = rng.randint(0, n_operators - 1)
        else:
            r = rng.random() * sum(op_weights)
            cum = 0.0
            for i, w in enumerate(op_weights):
                cum += w
                if r <= cum:
                    idx = i
                    break
            else:
                idx = n_operators - 1
        name, fn = ALL_DESTROY_OPERATORS[idx]
        return name, fn, idx

    destroy_ratio_start = 0.25
    destroy_ratio_end = 0.08
    min_destroy = 5
    rescue_interval = max(50, int(timelimit * 5))
    shake_interval = max(25, int(timelimit * 2.5))
    last_shake_iter = 0
    shake_min_gap = max(10, int(timelimit * 1))
    reheat_threshold = max(200, int(timelimit * 20))
    reheat_count = 0
    last_shake_iter = 0
    shake_min_gap = 500

    def _remove_blocks(ids: list[int]) -> dict[int, dict]:
        saved = {}
        for bid in ids:
            saved[bid] = dict(current_assignments[bid])
            a = saved[bid]
            bay_id = a["bay_id"]
            for i, b in enumerate(bay_placed[bay_id]):
                if b.block_id == bid:
                    bay_placed[bay_id].pop(i)
                    break
            for i, (entry, et) in enumerate(bay_schedule[bay_id]):
                if entry == a["entry_time"] and et == a["exit_time"]:
                    bay_schedule[bay_id].pop(i)
                    break
            bay_loads[bay_id] -= blocks_data[bid]["workload"]
            del current_assignments[bid]
        return saved

    def _restore_blocks(saved: dict[int, dict]):
        for bid, a in saved.items():
            bay_id = a["bay_id"]
            blk = Block(
                block_id=bid, block_data=blocks_data[bid],
                x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
            )
            bay_placed[bay_id].append(blk)
            bay_schedule[bay_id].append((a["entry_time"], a["exit_time"]))
            bay_loads[bay_id] += blocks_data[bid]["workload"]
            current_assignments[bid] = dict(a)

    while time.time() < deadline and stale < max_stale:
        n_iterations += 1
        stale += 1

        progress = min(1.0, (time.time() - lns_start) / max(timelimit, 0.1))
        destroy_ratio = destroy_ratio_start + (destroy_ratio_end - destroy_ratio_start) * progress
        k = max(min_destroy, int(n_blocks * destroy_ratio))

        if stale > rescue_interval:
            k = max(min_destroy, int(n_blocks * 0.50))
            op_name, op_fn, _ = "rescue_random", ALL_DESTROY_OPERATORS[0][1], 0
            to_remove = op_fn(current_assignments, blocks_data, k, rng)
            op_idx = -1
        elif stale > shake_interval and n_iterations - last_shake_iter > shake_min_gap:
            shake_prob = min(0.4, (stale - shake_interval) / max(rescue_interval - shake_interval, 1))
            if rng.random() < shake_prob:
                k = rng.randint(int(n_blocks * 0.40), int(n_blocks * 0.55))
                tardy_bids = sorted(
                    current_assignments.keys(),
                    key=lambda b: -(current_assignments[b]["exit_time"] - blocks_data[b]["due_date"]),
                )
                n_tardy = max(k, min(k, len(tardy_bids)))
                shake_ids = set(tardy_bids[:n_tardy])
                remaining = [b for b in current_assignments if b not in shake_ids]
                if remaining:
                    shake_ids.update(rng.sample(remaining, min(k - n_tardy, len(remaining))))
                to_remove = list(shake_ids)[:k]
                op_name = "massive_shake"
                op_idx = -1
                last_shake_iter = n_iterations
            else:
                op_name, op_fn, op_idx = _select_operator()
                to_remove = op_fn(current_assignments, blocks_data, k, rng)
        elif stale > shake_interval * 0.5:
            frac = (stale - shake_interval * 0.5) / max(rescue_interval - shake_interval * 0.5, 1)
            adaptive_ratio = destroy_ratio + (0.50 - destroy_ratio) * min(frac, 1.0)
            k = max(min_destroy, int(n_blocks * adaptive_ratio))
            op_name, op_fn, op_idx = _select_operator()
            to_remove = op_fn(current_assignments, blocks_data, k, rng)
        else:
            op_name, op_fn, op_idx = _select_operator()
            to_remove = op_fn(current_assignments, blocks_data, k, rng)
        if len(to_remove) < 1:
            continue

        old_of_removed = _remove_blocks(to_remove)

        is_shake = op_name == "massive_shake"

        beam_width = config.beam_width
        options_per_block = config.beam_options_per_block
        destroyed_order = sorted(to_remove, key=lambda bi: (
            blocks_data[bi]["due_date"],
            blocks_data[bi]["processing_time"],
        ))

        new_assignments = beam_search_rebuild(
            destroyed_order, old_of_removed,
            current_assignments, bay_placed, bay_schedule, bay_loads,
            blocks_data, bays, bays_data, weights_dict,
            n_bays, bbox_cache, w1, w2, w3, bay_weights,
            beam_width, options_per_block, rng,
        )

        bay_placed = [[] for _ in range(n_bays)]
        bay_schedule = [[] for _ in range(n_bays)]
        bay_loads = [0.0] * n_bays
        for bid, a in new_assignments.items():
            bay_id = a["bay_id"]
            bay_placed[bay_id].append(Block(
                block_id=bid, block_data=blocks_data[bid],
                x=int(a["x"]), y=int(a["y"]), orient_idx=a["orient_idx"],
            ))
            bay_schedule[bay_id].append((a["entry_time"], a["exit_time"]))
            bay_loads[bay_id] += blocks_data[bid]["workload"]
        current_assignments = {bid: dict(a) for bid, a in new_assignments.items()}

        for bay_id in range(n_bays):
            placed = bay_placed[bay_id]
            sched = bay_schedule[bay_id]
            order = sorted(range(len(placed)), key=lambda i: sched[i][0])
            tardy_fix_limit = max(5, int(n_blocks * 0.05))
            tardy_fixed = 0
            for idx in order:
                blk = placed[idx]
                entry, exit_t = sched[idx]
                bid = blk.block_id
                blk_data = blocks_data[bid]
                br = blk.bounding_rect()
                collision_found = False
                for k in range(len(placed)):
                    if k == idx:
                        continue
                    if not (sched[k][0] < exit_t and entry < sched[k][1]):
                        continue
                    if _bb_overlap(br, placed[k].bounding_rect()):
                        collision_found = True
                        break

                if collision_found:
                    bb = bbox_cache[(bid, blk.orient_idx)]
                    xr = _valid_x_range(bays[bay_id].width, bb)
                    yr = _valid_y_range(bays[bay_id].height, bb)
                    placed.pop(idx)
                    sched.pop(idx)
                    for px, py in [(xr[0], yr[0])]:
                        if xr[0] + (bb[2]-bb[0]) > bays[bay_id].width + 1e-6 or yr[0] + (bb[3]-bb[1]) > bays[bay_id].height + 1e-6:
                            continue
                        cand = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=blk.orient_idx)
                        slot = find_earliest_slot(cand, bays[bay_id], placed, sched, blk_data["release_time"], blk_data["processing_time"])
                        if slot[0] is not None:
                            entry, exit_t = slot
                            placed.insert(idx, cand)
                            sched.insert(idx, (entry, exit_t))
                            current_assignments[bid] = {
                                "block_id": bid, "bay_id": bay_id,
                                "x": px, "y": py, "orient_idx": blk.orient_idx,
                                "entry_time": entry, "exit_time": exit_t,
                            }
                            break
                    else:
                        placed.insert(idx, blk)
                        sched.insert(idx, (entry, exit_t))
                elif exit_t > blk_data["due_date"] and tardy_fixed < tardy_fix_limit:
                    tardy_fixed += 1
                    placed.pop(idx)
                    sched.pop(idx)
                    slot = find_earliest_slot(blk, bays[bay_id], placed, sched, blk_data["release_time"], blk_data["processing_time"])
                    if slot[0] is not None and slot[0] < entry:
                        entry, exit_t = slot
                    placed.insert(idx, blk)
                    sched.insert(idx, (entry, exit_t))
                    current_assignments[bid]["entry_time"] = entry
                    current_assignments[bid]["exit_time"] = exit_t

        new_obj = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)["objective"]

        if n_iterations % full_check_interval == 0:
            check_sol = {"operations": build_operations(list(current_assignments.values()))}
            full_result = check_feasibility(prob_info, check_sol)
            if not full_result["feasible"]:
                for bid in to_remove:
                    a_new = current_assignments[bid]
                    bay_id = a_new["bay_id"]
                    for i, b in enumerate(bay_placed[bay_id]):
                        if b.block_id == bid:
                            bay_placed[bay_id].pop(i)
                            break
                    for i, (entry, et) in enumerate(bay_schedule[bay_id]):
                        if entry == a_new["entry_time"] and et == a_new["exit_time"]:
                            bay_schedule[bay_id].pop(i)
                            break
                    bay_loads[bay_id] -= blocks_data[bid]["workload"]
                    del current_assignments[bid]
                _restore_blocks(old_of_removed)
                old_of_removed.clear()
                continue

        accepted = sa.accept(
            current_obj if current_obj != float("inf") else 0,
            new_obj, rng,
        )

        if accepted:
            n_accepted += 1
            if new_obj < best_obj:
                check_sol = {"operations": build_operations(list(current_assignments.values()))}
                full_result = check_feasibility(prob_info, check_sol)
                if full_result.get("feasible", False):
                    current_obj = new_obj
                    best_obj = new_obj
                    best_assignments = {
                        bid_: dict(a_) for bid_, a_ in current_assignments.items()
                    }
                    n_improved += 1
                    stale = 0
                    if verbose:
                        elapsed = time.time() - lns_start
                        print(f"[LNS] seed={seed}  iter={n_iterations}  "
                              f"best={best_obj:.0f}  "
                              f"elapsed={elapsed:.1f}s")
                else:
                    for bid in to_remove:
                        a_new = current_assignments[bid]
                        bay_id = a_new["bay_id"]
                        for i, b in enumerate(bay_placed[bay_id]):
                            if b.block_id == bid:
                                bay_placed[bay_id].pop(i)
                                break
                        for i, (entry, et) in enumerate(bay_schedule[bay_id]):
                            if entry == a_new["entry_time"] and et == a_new["exit_time"]:
                                bay_schedule[bay_id].pop(i)
                                break
                        bay_loads[bay_id] -= blocks_data[bid]["workload"]
                        del current_assignments[bid]
                    _restore_blocks(old_of_removed)
            else:
                current_obj = new_obj
        else:
            for bid in to_remove:
                a_new = current_assignments[bid]
                bay_id = a_new["bay_id"]
                for i, b in enumerate(bay_placed[bay_id]):
                    if b.block_id == bid:
                        bay_placed[bay_id].pop(i)
                        break
                for i, (entry, et) in enumerate(bay_schedule[bay_id]):
                    if entry == a_new["entry_time"] and et == a_new["exit_time"]:
                        bay_schedule[bay_id].pop(i)
                        break
                bay_loads[bay_id] -= blocks_data[bid]["workload"]
                del current_assignments[bid]
            _restore_blocks(old_of_removed)

        if op_idx >= 0:
            delta = current_obj - new_obj if accepted else 0.0
            op_scores[op_idx] += max(0.0, delta)
            op_counts[op_idx] += 1

        if n_iterations % weight_update_interval == 0 and op_idx >= 0:
            for oi in range(n_operators):
                if op_counts[oi] > 0:
                    avg = op_scores[oi] / op_counts[oi]
                    op_weights[oi] = max(0.1, op_weights[oi] * 0.8 + avg * 0.2)
            if sum(op_weights) > 0:
                inv = 1.0 / sum(op_weights)
                for oi in range(n_operators):
                    op_weights[oi] *= inv * n_operators

        if is_shake:
            sa.temperature = sa.initial_temp
            stale = 0
        elif stale > reheat_threshold and reheat_count < 3:
            sa.temperature = sa.initial_temp * 0.3
            reheat_count += 1
            stale = 0
        else:
            elapsed_frac = (time.time() - lns_start) / max(timelimit, 0.1)
            sa.temperature = sa.initial_temp * max(1e-6, (1.0 - elapsed_frac) ** cooling_power)

        if n_iterations % log_interval == 0 and verbose:
            elapsed = time.time() - lns_start
            reheat_tag = " REHEAT" if stale == 0 and reheat_count > 0 and n_iterations % log_interval < 5 else ""
            print(f"[LNS] seed={seed}  iter={n_iterations}  "
                  f"current={current_obj:.0f}  best={best_obj:.0f}  "
                  f"temp={sa.temperature:.1f}  stale={stale}  "
                  f"destroy={op_name} k={len(to_remove)}  elapsed={elapsed:.1f}s{reheat_tag}")

    if verbose:
        elapsed = time.time() - lns_start
        reason = "stale" if stale >= max_stale else "timelimit"
        print(f"[LNS] seed={seed}  iterations={n_iterations}  "
              f"accepted={n_accepted}  improved={n_improved}  "
              f"best={best_obj:.0f}  stop={reason}  elapsed={elapsed:.1f}s")

    return best_assignments
