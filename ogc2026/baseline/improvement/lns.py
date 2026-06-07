from __future__ import annotations
import math
import random
import time

from utils import Bay, Block, check_feasibility, check_entry, check_exit, check_collisions
from config import Config
from construction.helpers import build_operations, block_bbox, empty_bay_entry, find_earliest_slot
from construction.repair import _valid_x_range, _valid_y_range, _candidate_positions
from improvement.destroy import ALL_DESTROY_OPERATORS
from improvement.acceptance import SimulatedAnnealing
from core.objective import fast_objective


def _aabb_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


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

    destroy_ratio_start = 0.25
    destroy_ratio_end = 0.08
    min_destroy = 5
    rescue_interval = 5000
    reheat_threshold = 20000
    reheat_count = 0

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
            op_name, op_fn = "rescue_random", ALL_DESTROY_OPERATORS[0][1]
            to_remove = op_fn(current_assignments, blocks_data, k, rng)
        else:
            op_name, op_fn = rng.choice(ALL_DESTROY_OPERATORS)
            to_remove = op_fn(current_assignments, blocks_data, k, rng)
        if len(to_remove) < 1:
            continue

        old_of_removed = _remove_blocks(to_remove)

        destroyed_order = sorted(to_remove, key=lambda bi: (
            blocks_data[bi]["due_date"],
            blocks_data[bi]["processing_time"],
        ))

        explore_positions = rng.random() < 0.50
        stochastic_rebuild = rng.random() < 0.60

        for bid in destroyed_order:
            blk_data = blocks_data[bid]
            old_a = old_of_removed[bid]
            r_time = blk_data["release_time"]
            proc = blk_data["processing_time"]
            prefs = blk_data["bay_preferences"]
            n_o = len(blk_data["shape"])

            best_fit = None
            best_score = float("inf")
            all_options: list[tuple[float, tuple]] = []

            old_bay_id = old_a["bay_id"]
            old_x = int(old_a["x"])
            old_y = int(old_a["y"])
            old_oi = old_a["orient_idx"]
            old_entry = old_a["entry_time"]
            old_exit = old_a["exit_time"]
            was_tardy = max(0, old_exit - blk_data["due_date"])

            bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
            if rng.random() < 0.20 and len(bay_order) > 1:
                pick = rng.randint(1, len(bay_order) - 1)
                bay_order[0], bay_order[pick] = bay_order[pick], bay_order[0]
            for bay_id in bay_order:
                bay = bays[bay_id]
                if bay_id == old_bay_id:
                    oi_list = [old_oi]
                else:
                    oi_list = sorted(
                        range(n_o),
                        key=lambda oi_: (
                            bbox_cache[(bid, oi_)][2] - bbox_cache[(bid, oi_)][0]
                        ) * (bbox_cache[(bid, oi_)][3] - bbox_cache[(bid, oi_)][1])
                    )
                for oi in oi_list:
                    bb = bbox_cache[(bid, oi)]
                    bw = bb[2] - bb[0]
                    bh = bb[3] - bb[1]
                    if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                        continue
                    if bay_id == old_bay_id and oi == old_oi:
                        px, py = old_x, old_y
                    elif explore_positions:
                        xr = _valid_x_range(bay.width, bb)
                        yr = _valid_y_range(bay.height, bb)
                        if xr[0] > xr[1] or yr[0] > yr[1]:
                            continue
                        for _ in range(5):
                            px = rng.randint(xr[0], xr[1])
                            py = rng.randint(yr[0], yr[1])
                            new_blk = Block(block_id=bid, block_data=blk_data, x=px, y=py, orient_idx=oi)
                            nbr = new_blk.bounding_rect()
                            if not any(
                                _aabb_overlap(nbr, b.bounding_rect())
                                for b in bay_placed[bay_id]
                            ):
                                break
                        else:
                            xr = _valid_x_range(bay.width, bb)
                            yr = _valid_y_range(bay.height, bb)
                            if xr[0] > xr[1] or yr[0] > yr[1]:
                                continue
                            candidates = _candidate_positions(xr, yr)
                            px, py = candidates[0]
                            if px + bw > bay.width + 1e-6 or py + bh > bay.height + 1e-6:
                                continue
                    else:
                        xr = _valid_x_range(bay.width, bb)
                        yr = _valid_y_range(bay.height, bb)
                        if xr[0] > xr[1] or yr[0] > yr[1]:
                            continue
                        candidates = _candidate_positions(xr, yr)
                        px, py = candidates[0]
                        if px + bw > bay.width + 1e-6 or py + bh > bay.height + 1e-6:
                            continue
                    entry = empty_bay_entry(bay_schedule[bay_id], r_time, proc)
                    if entry is not None:
                        exit_t = entry + proc
                        # Position-aware timing for high-tardiness blocks (probabilistic, expensive)
                        if n_iterations % 5 == 0 and was_tardy > 0 and bay_id == old_bay_id and oi == old_oi and (px, py) == (old_x, old_y):
                            cand_blk = Block(block_id=bid, block_data=blk_data, x=old_x, y=old_y, orient_idx=old_oi)
                            slot = find_earliest_slot(cand_blk, bay, bay_placed[bay_id], bay_schedule[bay_id], r_time, proc)
                            if slot[0] is not None and slot[0] < entry:
                                entry, exit_t = slot
                        tardy = max(0, exit_t - blk_data["due_date"])
                        pref_penalty = max(prefs) - prefs[bay_id]
                        score = tardy * w1 + pref_penalty * w3
                        fit = (bay_id, px, py, oi, int(entry), int(exit_t))
                        all_options.append((score, fit))
                        if score < best_score:
                            best_score = score
                            best_fit = fit
                            if score == 0:
                                break
                if best_fit and best_score == 0:
                    break

            if best_fit is None:
                bj = max(range(n_bays), key=lambda j: prefs[j])
                bay = bays[bj]
                bb = bbox_cache[(bid, 0)]
                xr = _valid_x_range(bay.width, bb)
                yr = _valid_y_range(bay.height, bb)
                px = max(0, xr[0]) if xr[0] <= xr[1] else 0
                py = max(0, yr[0]) if yr[0] <= yr[1] else 0
                entry = empty_bay_entry(bay_schedule[bj], r_time, proc)
                best_fit = (bj, px, py, 0, int(entry), int(entry + proc))

            if stochastic_rebuild and len(all_options) > 1:
                all_options.sort(key=lambda x: x[0])
                top_k = min(3, len(all_options))
                if rng.random() < 0.5:
                    best_fit = rng.choice(all_options[:top_k])[1]

            bay_id, cx, cy, oi, entry, exit_t = best_fit
            blk = Block(
                block_id=bid, block_data=blk_data,
                x=cx, y=cy, orient_idx=oi,
            )
            bay_placed[bay_id].append(blk)
            bay_schedule[bay_id].append((entry, exit_t))
            bay_loads[bay_id] += blk_data["workload"]
            current_assignments[bid] = {
                "block_id": bid, "bay_id": bay_id,
                "x": cx, "y": cy, "orient_idx": oi,
                "entry_time": entry, "exit_time": exit_t,
            }

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
                current_obj = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)["objective"]
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
                    _restore_blocks(old_of_removed)
                    current_obj = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)["objective"]
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
            _restore_blocks(old_of_removed)
            current_obj = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)["objective"]

        if stale > reheat_threshold and reheat_count < 3:
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
