#!/usr/bin/env python3
"""Benchmark all 20 training instances at a given timelimit."""

import json
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from solver import solve


def main():
    timelimit = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    train_dir = os.path.join(os.path.dirname(__file__), "train")

    results = []
    for i in range(1, 21):
        path = os.path.join(train_dir, f"prob_{i}.json")
        with open(path) as f:
            prob = json.load(f)

        print(f"\n{'='*60}")
        label = prob.get("info", {}).get("label", f"prob_{i}")
        print(f"prob_{i}  ({label})")
        print(f"{'='*60}")

        t0 = time.time()
        sol = solve(prob, timelimit=timelimit)
        t = time.time() - t0

        if sol:
            # Re-check to get objective
            from utils import check_feasibility
            result = check_feasibility(prob, sol)
            obj = result.get("objective", 0)
            obj1 = result.get("obj1", 0) or 0
            obj2 = result.get("obj2", 0) or 0
            obj3 = result.get("obj3", 0) or 0
            feasible = result.get("feasible", False)
            results.append({
                "prob": f"prob_{i}",
                "feasible": feasible,
                "objective": obj,
                "obj1": obj1,
                "obj2": obj2,
                "obj3": obj3,
                "time": t,
            })
            status = "FEASIBLE" if feasible else "INFEASIBLE"
            print(f"  {status}: obj={obj:.0f}  obj1={obj1:.1f}  obj2={obj2:.1f}  obj3={obj3:.1f}  time={t:.1f}s")
        else:
            print(f"  No solution returned!")
            results.append({
                "prob": f"prob_{i}",
                "feasible": False,
                "objective": None,
                "obj1": None,
                "obj2": None,
                "obj3": None,
                "time": t,
            })

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_obj = 0
    feasible_count = 0
    for r in results:
        obj_str = f"{r['objective']:.0f}" if r['objective'] is not None else "N/A"
        print(f"  {r['prob']:8s}  {'OK' if r['feasible'] else 'FAIL'}  obj={obj_str:>12s}  time={r['time']:.1f}s")
        if r['feasible'] and r['objective'] is not None:
            total_obj += r['objective']
            feasible_count += 1
    print(f"\n  Feasible: {feasible_count}/20")
    print(f"  Total objective: {total_obj:.0f}")


if __name__ == "__main__":
    main()
