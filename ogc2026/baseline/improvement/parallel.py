from __future__ import annotations
import concurrent.futures
import multiprocessing
import time
from typing import Optional

from config import Config
from improvement.lns import run_lns
from construction.helpers import build_operations
from utils import check_feasibility


def run_parallel_lns(
    prob_info: dict,
    bays,
    blocks_data,
    w1, w2, w3,
    initial_assignments: dict[int, dict],
    t_start: float,
    timelimit: float,
    config: Config,
    num_workers: int = 4,
    verbose: bool = False,
) -> dict[int, dict]:
    if num_workers <= 1 or not config.use_parallel:
        return run_lns(
            prob_info, bays, blocks_data, w1, w2, w3,
            initial_assignments, t_start, timelimit,
            config, seed=42, verbose=verbose,
        )

    worker_time = timelimit / num_workers

    if verbose:
        print(f"[Parallel] Starting {num_workers} workers, {worker_time:.1f}s each")

    results = [None] * num_workers

    def worker_fn(worker_id: int) -> dict:
        seed = 42 + worker_id
        return run_lns(
            prob_info, bays, blocks_data, w1, w2, w3,
            initial_assignments, t_start, worker_time,
            config, seed=seed, verbose=verbose,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(worker_fn, i): i
            for i in range(num_workers)
        }
        for future in concurrent.futures.as_completed(futures):
            wid = futures[future]
            try:
                results[wid] = future.result()
            except Exception as e:
                if verbose:
                    print(f"[Parallel] Worker {wid} failed: {e}")

    best = initial_assignments
    best_obj = float("inf")

    from core.objective import fast_objective
    for r in results:
        if r is not None:
            check_sol = {"operations": build_operations(list(r.values()))}
            full_result = check_feasibility(prob_info, check_sol)
            if not full_result.get("feasible", False):
                continue
            obj_r = full_result["objective"]
            if obj_r < best_obj:
                best_obj = obj_r
                best = r

    if verbose:
        print(f"[Parallel] Best across workers: {best_obj:.0f}")

    return best


def run_multi_start_lns(
    prob_info: dict,
    bays,
    blocks_data,
    w1, w2, w3,
    initial_assignments: dict[int, dict],
    t_start: float,
    timelimit: float,
    config: Config,
    verbose: bool = False,
) -> dict[int, dict]:
    each_run = max(15.0, timelimit / 90)
    max_runs = max(1, int(timelimit / each_run))

    best_assignments = {bid: dict(a) for bid, a in initial_assignments.items()}
    best_obj = float("inf")
    n_done = 0

    if verbose:
        print(f"[MultiStart] Up to {max_runs} independent runs x {each_run:.1f}s each")

    for run in range(max_runs):
        if time.time() - t_start > timelimit - each_run * 0.5:
            break

        seed = 42 + run
        result = run_parallel_lns(
            prob_info, bays, blocks_data, w1, w2, w3,
            initial_assignments, t_start, each_run,
            config, num_workers=2, verbose=False,
        )
        n_done += 1

        check_sol = {"operations": build_operations(list(result.values()))}
        full_result = check_feasibility(prob_info, check_sol)
        if not full_result.get("feasible", False):
            continue

        obj_r = full_result["objective"]
        if obj_r < best_obj:
            best_obj = obj_r
            best_assignments = {bid: dict(a) for bid, a in result.items()}
            if verbose:
                elapsed = time.time() - t_start
                print(f"[MultiStart] run {run+1}/{max_runs}  "
                      f"new best={best_obj:.0f}  elapsed={elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"[MultiStart] Final: {best_obj:.0f}  after {n_done} runs  elapsed={elapsed:.1f}s")

    return best_assignments
