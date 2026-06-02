# ShipScheduler

OGS 2026 LG CNS Block Spatial Scheduling solver with guaranteed feasibility.

Always-feasible shipyard block scheduling — `empty_bay_entry` window placement guarantees crane clearance, multi-strategy heuristics optimize the build sequence.

## Quick Start

```bash
python3 myalgorithm.py  # or import from alg_tester
```

## Architecture

- **construction/** — 5 ordering heuristics (EDD, EST, Slack, SPT, Weighted) + empty-bay placement kernel
- **improvement/** — single-block LNS with Simulated Annealing, parallel workers, local search
- **core/** — objective function, geometry utilities, SpatialGrid (WIP)

## Built With

- Python 3.12
- Shapely
- concurrent.futures
