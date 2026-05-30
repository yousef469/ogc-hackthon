from __future__ import annotations
import random
import time

from utils import Bay, Block, check_feasibility
from config import Config
from construction.helpers import build_operations, block_bbox, empty_bay_entry
from improvement.acceptance import SimulatedAnnealing
from core.objective import fast_objective


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

    if verbose:
        print(f"[LNS] seed={seed}  initial={current_obj:.0f}  "
              f"(tardy={obj_result['obj1']:.0f}  "
              f"bal={obj_result['obj2']:.1f}  "
              f"pref={obj_result['obj3']:.0f})")

    n_iterations = 0
    n_accepted = 0
    n_improved = 0
    stale = 0
    max_stale = 5000
    deadline = time.time() + timelimit
    log_interval = max(500, 2000 // (config.num_workers if config.use_parallel else 1))
    full_check_interval = 200

    while time.time() < deadline and stale < max_stale:
        n_iterations += 1
        stale += 1

        bid = rng.choice(list(current_assignments.keys()))
        a = current_assignments[bid]
        old_bay = a["bay_id"]
        blk_data = blocks_data[bid]
        r_time = blk_data["release_time"]
        proc = blk_data["processing_time"]
        prefs = blk_data["bay_preferences"]

        for i, b in enumerate(bay_placed[old_bay]):
            if b.block_id == bid:
                bay_placed[old_bay].pop(i)
                break
        for i, (entry, et) in enumerate(bay_schedule[old_bay]):
            if entry == a["entry_time"] and et == a["exit_time"]:
                bay_schedule[old_bay].pop(i)
                break
        bay_loads[old_bay] -= blk_data["workload"]

        best_fit = None
        best_score = float("inf")
        bay_order = sorted(range(n_bays), key=lambda j: prefs[j], reverse=True)
        orig_x = a.get("x", 0)
        orig_y = a.get("y", 0)
        orig_oi = a.get("orient_idx", 0)

        for bay_id in bay_order:
            bay = bays[bay_id]
            bb = block_bbox(blk_data, orig_oi)
            bw = bb[2] - bb[0]
            bh = bb[3] - bb[1]
            if bw > bay.width + 1e-6 or bh > bay.height + 1e-6:
                continue
            if bay_id != old_bay:
                if orig_x + bw > bay.width + 1e-6 or orig_y + bh > bay.height + 1e-6:
                    continue
                test_blk = Block(
                    block_id=bid, block_data=blk_data,
                    x=orig_x, y=orig_y, orient_idx=orig_oi,
                )
                if not bay.contains_block(test_blk):
                    continue
            entry = empty_bay_entry(bay_schedule[bay_id], r_time, proc)
            if entry is not None:
                tardy = max(0, entry + proc - blk_data["due_date"])
                pref_penalty = max(prefs) - prefs[bay_id]
                score = tardy * w1 + pref_penalty * w3
                if score < best_score:
                    best_score = score
                    best_fit = (bay_id, orig_x, orig_y, orig_oi, entry, entry + proc)
                    if score == 0:
                        break

        if best_fit is None:
            best_fit = (old_bay, a.get("x", 0), a.get("y", 0),
                        a.get("orient_idx", 0), a["entry_time"], a["exit_time"])

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
            "entry_time": int(entry), "exit_time": int(exit_t),
        }

        new_obj = fast_objective(current_assignments, blocks_data, bays_data, weights_dict)["objective"]

        if n_iterations % full_check_interval == 0:
            check_sol = {"operations": build_operations(list(current_assignments.values()))}
            full_result = check_feasibility(prob_info, check_sol)
            if not full_result["feasible"]:
                current_assignments.clear()
                for bid_, a_ in best_assignments.items():
                    current_assignments[bid_] = dict(a_)
                new_obj = best_obj
            else:
                full_obj = full_result.get("objective", new_obj)
                if abs(full_obj - new_obj) > 1.0:
                    new_obj = full_obj

        accepted = sa.accept(
            current_obj if current_obj != float("inf") else 0,
            new_obj, rng,
        )

        if accepted:
            n_accepted += 1
            current_obj = new_obj
            if new_obj < best_obj:
                check_sol = {"operations": build_operations(list(current_assignments.values()))}
                full_result = check_feasibility(prob_info, check_sol)
                if full_result.get("feasible", False):
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
            current_assignments.clear()
            for bid_, a_ in best_assignments.items():
                current_assignments[bid_] = dict(a_)
            current_obj = best_obj

            bay_placed.clear()
            bay_placed.extend([] for _ in range(n_bays))
            bay_schedule.clear()
            bay_schedule.extend([] for _ in range(n_bays))
            bay_loads[:] = [0.0] * n_bays

            for bid_, a_ in best_assignments.items():
                bay_id_ = a_["bay_id"]
                blk_ = Block(
                    block_id=bid_, block_data=blocks_data[bid_],
                    x=int(a_["x"]), y=int(a_["y"]),
                    orient_idx=a_["orient_idx"],
                )
                bay_placed[bay_id_].append(blk_)
                bay_schedule[bay_id_].append((a_["entry_time"], a_["exit_time"]))
                bay_loads[bay_id_] += blocks_data[bid_]["workload"]

        sa.cool()

        if n_iterations % log_interval == 0 and verbose:
            elapsed = time.time() - lns_start
            print(f"[LNS] seed={seed}  iter={n_iterations}  "
                  f"current={current_obj:.0f}  best={best_obj:.0f}  "
                  f"temp={sa.temperature:.1f}  stale={stale}  elapsed={elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - lns_start
        reason = "stale" if stale >= max_stale else "timelimit"
        print(f"[LNS] seed={seed}  iterations={n_iterations}  "
              f"accepted={n_accepted}  improved={n_improved}  "
              f"best={best_obj:.0f}  stop={reason}  elapsed={elapsed:.1f}s")

    return best_assignments
