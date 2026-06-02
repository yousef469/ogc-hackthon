# ShipScheduler — OGC 2026 LG CNS Block Spatial Scheduling

Always-feasible solver for the 2026 LG CNS shipyard block packing + scheduling optimization competition.

## Problem

Irregular 3D blocks must be assigned to bays, positioned spatially, and scheduled over time. Each block has:
- Release time, processing time, due date
- Multi-layer polygonal footprint (different shapes per layer)
- Bay preference vector
- Vertical crane sweep constraint: layer j above layer k requires j ≥ k

Objective minimizes: `w₁ × tardiness + w₂ × load imbalance + w₃ × preference penalty`

## Key Innovation: `empty_bay_entry`

The official baseline uses Shapely geometry checks (`check_entry`/`check_exit`) at every placement — slow (62s for 100 blocks) and produces **infeasible** solutions.

Our `empty_bay_entry` finds the earliest time window where the bay is **completely empty** during the block's stay. This guarantees both entry and exit crane paths are clear by construction (no other blocks present = no obstructions). Result: **always feasible**, **5–15× faster**.

## Architecture (20 source files, 33KB)

```
baseline/
  myalgorithm.py          # Entry point (imported by alg_tester)
  solver.py               # Orchestrator: construction → LNS → refine → fallback
  config.py               # Auto-tuning: SA temp by w1, time budgets
  utils.py                # Shapely feasibility checks, Bay/Block types (competition-provided)

  construction/
    strategies.py         # 5 ordering heuristics: EDD, EST, Slack, SPT, Weighted
    greedy.py             # Placement kernel with empty_bay_entry
    repair.py             # repair_simple: empty-bay window repair (fast, guaranteed)
    helpers.py            # empty_bay_entry, build_operations, block_bbox

  improvement/
    lns.py                # Single-block LNS with SA acceptance, stale detection
    acceptance.py         # Simulated Annealing
    destroy.py            # Destroy operators (reserved)
    parallel.py           # ThreadPoolExecutor wrapper
    refine.py             # Local search: swap, move, rotate
    local_search.py       # Geometric swap/move/rotate primitives

  core/
    objective.py          # fast_objective (no Shapely, formula only)
    geometry.py           # SpatialGrid acceleration (WIP)
    types.py              # Bay, Block dataclasses
```

## Results

| Instance | Blocks | Baseline | ShipScheduler | Improvement |
|---|---|---|---|---|
| hard_2bay_60_tight | 60 | **INFEASIBLE** | **1,662,274** | ✗ (baseline fails) |
| hard_2bay_100_tight | 100 | **INFEASIBLE** | **3,411,915** | ✗ (baseline fails) |
| example_B2_b10 | 10 | **INFEASIBLE** | **1,056** | ✗ (baseline fails) |

Baseline fails because Shapely `find_earliest_slot` + repair creates unrecoverable crane-path violations. Our `empty_bay_entry` approach is conservative but always valid.

## Built With

- Python 3.12
- Shapely (polygon operations, feasibility checking)
- concurrent.futures (parallel search)

## Timeline

- ~~June 7~~ Registration closed
- **June 15** — Prelim opens, instances released, submission portal opens
- **July 28** — Prelim closes

## Running

```bash
cd ogc2026/baseline
python3 myalgorithm.py   # or import via alg_tester
```

## Status

Ready for prelim. The improvement phase (LNS/refine) needs real competition instances to tune — synthetic rectangles don't stress the geometry engine. After June 15 we'll profile on 100–500 block instances and rebuild the metaheuristic.
