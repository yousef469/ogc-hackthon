from __future__ import annotations
import concurrent.futures
import multiprocessing
import time
from typing import Optional

from config import Config
from improvement.lns import run_lns


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
            config, seed=seed, verbose=False,
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
            obj_r = fast_objective(r, blocks_data, prob_info["bays"], prob_info.get("weights", {}))
            if obj_r["objective"] < best_obj:
                best_obj = obj_r["objective"]
                best = r

    if verbose:
        print(f"[Parallel] Best across workers: {best_obj:.0f}")

    return best
