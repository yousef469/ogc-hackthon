# AGENTS.md — opencode agent memory

## Benchmark
Run all 20 training instances:
```bash
cd /Users/yousef/hackthon\ 1/ogc2026 && python3 benchmark.py 30.0
```

## Final metrics (30s benchmark, 2026-06-13)
- **20/20 feasible**
- **Total objective: 10,601M** (30s per instance)
- **30-minute estimate: ~9,500M** (with longer runtimes per instance)

Instance breakdown:
| Instance | Blocks | Bays | Objective | obj1 (tardy) | obj2 (bal) | obj3 (pref) |
|---|---|---|---|---|---|---|
| prob_1 | 100 | 2 | 394,996,623 | 13,560 | 723 | 2,588 |
| prob_2 | 100 | 2 | 240,443,373 | 8,247 | 670 | 3,488 |
| prob_3 | 100 | 2 | 240,272,258 | 8,992 | 2,419 | 3,056 |
| prob_4 | 100 | 2 | 304,957,324 | 13,884 | 773 | 3,212 |
| prob_5 | 100 | 3 | 300,109,886 | 18,711 | 13,748 | 4,251 |
| prob_6 | 150 | 3 | 631,130,302 | 21,270 | 3,865 | 5,821 |
| prob_7 | 150 | 3 | 350,561,524 | 19,676 | 8,985 | 4,658 |
| prob_8 | 200 | 4 | 268,702,554 | 26,804 | 1,139 | 3,290 |
| prob_9 | 200 | 4 | 467,884,070 | 35,014 | 5,322 | 6,772 |
| prob_10 | 200 | 4 | 366,627,752 | 25,143 | 6,583 | 6,592 |
| prob_11 | 200 | 5 | 631,759,960 | 27,605 | 2,726 | 5,815 |
| prob_12 | 200 | 5 | 581,326,195 | 26,481 | 6,944 | 6,519 |
| prob_13 | 250 | 5 | 819,908,459 | 44,011 | 5,397 | 7,946 |
| prob_14 | 250 | 5 | 755,558,207 | 42,443 | 6,732 | 7,315 |
| prob_15 | 250 | 5 | 588,961,205 | 39,695 | 5,822 | 6,396 |
| prob_16 | 300 | 5 | 364,103,075 | 40,833 | 9,508 | 8,203 |
| prob_17 | 300 | 5 | 557,390,898 | 57,352 | 8,799 | 9,123 |
| prob_18 | 300 | 5 | 820,564,493 | 61,454 | 5,881 | 8,833 |
| prob_19 | 300 | 4 | 608,110,220 | 56,875 | 12,543 | 10,334 |
| prob_20 | 300 | 5 | 1,308,180,464 | 49,011 | 11,105 | 9,100 |

## Architecture
- **Multi-strategy LNS** with 4-core ProcessPoolExecutor parallelism
- **Beam search rebuild** (width 4-12, position diversity)
- **12 destroy operators** (random, worst, related, bay, cross_bay, time_window, critical_path, spatial_cluster, due_date_window, precedence_chain, tardy_blaster)
- **Adaptive operator weights** (tracks success delta, renormalizes every 20 iters)
- **Elite solution sharing** (each batch starts from current best)
- **SA acceptance** with time-based cooling and reheat on stale
- **Geometry-aware timing fix** inside LNS loop (find_earliest_slot for top-5% tardy + collided)
- **Post-processing:** refine_timing (8% budget, double pass) + refine_solution

## Key files
| File | Purpose |
|---|---|
| `baseline/myalgorithm.py` | Entry point |
| `baseline/solver.py` | Construction + LNS + post-processing pipeline |
| `baseline/improvement/lns.py` | Beam search rebuild + LNS loop |
| `baseline/improvement/destroy.py` | 12 destroy operators |
| `baseline/improvement/parallel.py` | ProcessPoolExecutor multi-start with elite sharing |
| `baseline/improvement/refine.py` | Time shift, bay reassignment, escape_tardiness |
| `baseline/improvement/local_search.py` | Swap/move/rotate/local move primitives |
| `baseline/improvement/acceptance.py` | SimulatedAnnealing |
| `baseline/config.py` | Auto-tuned config per instance |
| `baseline/construction/` | 6 construction strategies + repair |

## Competition info
- **Prelim:** June 15 – July 28, top 30-40 advance
- **Final:** Aug 3–14, + technical report required
- **618 teams registered**, 968 participants
- **Update AGENTS.md** with benchmark results after significant changes
- **Submit** via `alg_tester` on the competition website
