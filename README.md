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

## Results — 20/20 training instances feasible (8s timelimit each)

| Inst | Objective | Time | Inst | Objective | Time |
|------|-----------|------|------|-----------|------|
|  1 |  395,047,686 | 7.6s | 11 |  637,844,731 | 6.2s |
|  2 |  246,841,612 | 7.6s | 12 |  590,059,510 | 6.2s |
|  3 |  240,358,922 | 7.5s | 13 |  819,908,459 | 6.3s |
|  4 |  305,685,968 | 7.5s | 14 |  761,721,777 | 6.4s |
|  5 |  316,593,270 | 7.6s | 15 |  606,563,665 | 6.3s |
|  6 |  642,585,506 | 6.1s | 16 |  364,103,075 | 6.3s |
|  7 |  365,829,277 | 6.1s | 17 |  557,390,898 | 6.4s |
|  8 |  273,794,137 | 6.1s | 18 |  820,564,493 | 6.4s |
|  9 |  467,884,070 | 6.3s | 19 |  608,110,220 | 6.4s |
| 10 |  372,789,048 | 6.3s | 20 | 1,358,367,398 | 6.4s |
| **Total** |         | **131.8s** | | | |

All 20 training instances (100–300 blocks, 2–5 bays with complex polygon shapes) produce feasible solutions. Average wall time ~6.6s/instance.

Key fixes:
- **Bounding-box-aware positioning**: blocks whose shape extends left/down from the reference point are shifted so the bounding rect starts at (0,0) using `ceil(max(0, -local_min))`.
- **LNS position perturbation**: 15% of iterations try random positions/orientations, not just bay/time swaps.

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

**20/20 training instances feasible** in under 8s each (total 132s). Submitted zip verified with `alg_tester`. Tuning LNS/refine for better objective improvement on larger instances after June 15 instance release.
