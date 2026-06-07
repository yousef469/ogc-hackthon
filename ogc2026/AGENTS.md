# AGENTS.md — opencode agent memory

## Benchmark
Run all 20 training instances:
```bash
cd /Users/yousef/hackthon\ 1/ogc2026 && python3 benchmark.py 30.0
```

## Final metrics (30s benchmark)
- **20/20 feasible**
- **Total objective: 10,047,718,963** (variance ±300M between runs)
- **30-minute estimate: ~8,500-9,000M** (15-25% from construction)

## Key features
- ProcessPoolExecutor (4 cores)
- Geometry-aware timing every LNS iteration
- refine_timing (tardy-block hill climb)
- refine_solution (swap/move/rotate/time_shift/reassign)
- Massive shake (40-55% destroy, 80% cross-bay, temp reset)
- Adaptive destroy (gradual increase when stale)
- Cross-bay destroy operator
- Weight-aware LNS scoring (marginal w2 imbalance)
- Multi-start: 352 runs × 20s at 30min

## Submission
```bash
# alg_tester expects `myalgorithm.py` at root
ls baseline/myalgorithm.py
```
