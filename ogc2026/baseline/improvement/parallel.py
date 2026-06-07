from __future__ import annotations
import concurrent.futures
import multiprocessing
import time

from config import Config
from improvement.lns import run_lns
from construction.helpers import build_operations
from utils import check_feasibility
from core.objective import fast_objective


def _lns_worker(
    prob_info, bays, blocks_data, w1, w2, w3,
    initial_assignments, t_start, worker_time, config, seed,
):
    return run_lns(
        prob_info, bays, blocks_data, w1, w2, w3,
        initial_assignments, t_start, worker_time,
        config, seed=seed,
    )


def _evaluate(prob_info, assignments, best_obj):
    check_sol = {"operations": build_operations(list(assignments.values()))}
    full_result = check_feasibility(prob_info, check_sol)
    if not full_result.get("feasible", False):
        return None
    obj_r = full_result["objective"]
    if obj_r < best_obj[0]:
        best_obj[0] = obj_r
        return {bid: dict(a) for bid, a in assignments.items()}
    return None


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

    best = initial_assignments
    best_obj = [float("inf")]

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, mp_context=ctx,
    ) as pool:
        futures = [
            pool.submit(
                _lns_worker,
                prob_info, bays, blocks_data, w1, w2, w3,
                initial_assignments, t_start, worker_time, config, 42 + i,
            )
            for i in range(num_workers)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                if verbose:
                    print(f"[Parallel] Worker failed: {e}")
                continue
            best_candidate = _evaluate(prob_info, result, best_obj)
            if best_candidate is not None:
                best = best_candidate

    if verbose:
        print(f"[Parallel] Best across workers: {best_obj[0]:.0f}")

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

    best_assignments = {bid: dict(a) for bid, a in initial_assignments.items()}
    best_obj = [float("inf")]
    n_done = 0

    if verbose:
        n_workers = config.num_workers
        print(f"[MultiStart] ProcessPoolExecutor({n_workers})  "
              f"{each_run:.1f}s per run")

    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.num_workers, mp_context=ctx,
    ) as pool:
        run_idx = 0
        while time.time() - t_start < timelimit - each_run * 1.1:
            batch = []
            for _ in range(config.num_workers):
                if time.time() - t_start > timelimit - each_run * 1.1:
                    break
                seed = 42 + run_idx
                run_idx += 1
                batch.append(pool.submit(
                    _lns_worker,
                    prob_info, bays, blocks_data, w1, w2, w3,
                    initial_assignments, t_start, each_run, config, seed,
                ))

            for future in concurrent.futures.as_completed(batch):
                try:
                    result = future.result()
                except Exception:
                    continue
                n_done += 1
                best_candidate = _evaluate(prob_info, result, best_obj)
                if best_candidate is not None:
                    best_assignments = best_candidate
                    if verbose:
                        elapsed = time.time() - t_start
                        print(f"[MultiStart] run {run_idx}  "
                              f"new best={best_obj[0]:.0f}  elapsed={elapsed:.1f}s")

    if verbose:
        elapsed = time.time() - t_start
        print(f"[MultiStart] Final: {best_obj[0]:.0f}  after {n_done} runs  elapsed={elapsed:.1f}s")

    return best_assignments
