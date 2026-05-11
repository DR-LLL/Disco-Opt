
from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class TSPInstance:
    n: int
    xs: List[float]
    ys: List[float]

    @classmethod
    def from_file(cls, path: str) -> "TSPInstance":
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError("Empty file")
            n = int(header.strip())
            xs: List[float] = []
            ys: List[float] = []
            for i in range(n):
                line = f.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF in {path}, point line {i}")
                parts = line.strip().split()
                if len(parts) != 2:
                    raise ValueError(f"Bad point line {i} in {path}: {line!r}")
                x, y = map(float, parts)
                xs.append(x)
                ys.append(y)
        return cls(n=n, xs=xs, ys=ys)


def parse_solution(solution_str: str) -> List[int]:
    solution_str = solution_str.strip()
    if not solution_str:
        return []
    return [int(x) for x in solution_str.split()]


def dist(inst: TSPInstance, i: int, j: int) -> float:
    dx = inst.xs[i] - inst.xs[j]
    dy = inst.ys[i] - inst.ys[j]
    return math.hypot(dx, dy)


def check_solution(inst: TSPInstance, tour: List[int], objective_value: float, eps: float = 1e-8):
    if len(tour) != inst.n:
        return False, f"tour length mismatch: got {len(tour)}, expected {inst.n}"
    seen = [False] * inst.n
    for v in tour:
        if v < 0 or v >= inst.n:
            return False, f"vertex out of range: {v}"
        if seen[v]:
            return False, f"duplicate vertex: {v}"
        seen[v] = True

    total = 0.0
    for i in range(inst.n):
        total += dist(inst, tour[i], tour[(i + 1) % inst.n])

    if not math.isfinite(objective_value):
        return False, "objective is not finite"

    tol = eps * max(1.0, abs(objective_value), abs(total))
    if abs(total - objective_value) > tol:
        return False, f"objective mismatch: csv={objective_value}, real={total}"

    return True, f"OK, cost={total:.10f}, n={inst.n}"


def main():
    data_folder = "data"
    results_csv = "results.csv"

    if not os.path.isdir(data_folder):
        print(f"ERROR: folder does not exist: {data_folder}")
        sys.exit(1)

    if not os.path.isfile(results_csv):
        print(f"ERROR: csv does not exist: {results_csv}")
        sys.exit(1)

    ok_count = 0
    fail_count = 0
    skip_count = 0
    total_count = 0

    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"test_name", "objective_value", "best_found_time_sec", "solution"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: missing columns in csv: {sorted(missing)}")
            sys.exit(1)

        for row in reader:
            total_count += 1
            test_name = row["test_name"].strip()
            obj_str = row["objective_value"].strip()
            time_str = row["best_found_time_sec"].strip()
            solution_str = row["solution"]

            path = os.path.join(data_folder, test_name)
            if not os.path.isfile(path):
                fail_count += 1
                print(f"✗ {test_name}: instance file not found")
                continue

            if obj_str == "ERROR":
                skip_count += 1
                print(f"• {test_name}: solver wrote ERROR")
                continue

            try:
                objective_value = float(obj_str)
            except Exception:
                fail_count += 1
                print(f"✗ {test_name}: bad objective_value = {obj_str!r}")
                continue

            try:
                best_found_time = float(time_str)
                if best_found_time < 0:
                    fail_count += 1
                    print(f"✗ {test_name}: negative best_found_time_sec = {best_found_time}")
                    continue
            except Exception:
                fail_count += 1
                print(f"✗ {test_name}: bad best_found_time_sec = {time_str!r}")
                continue

            try:
                inst = TSPInstance.from_file(path)
                tour = parse_solution(solution_str)
                ok, msg = check_solution(inst, tour, objective_value)
                if ok:
                    ok_count += 1
                    print(f"✓ {test_name}: {msg}, best_found_time_sec={best_found_time:.6f}")
                else:
                    fail_count += 1
                    print(f"✗ {test_name}: {msg}, best_found_time_sec={best_found_time:.6f}")
            except Exception as e:
                fail_count += 1
                print(f"✗ {test_name}: exception during check: {e}")

    print()
    print(f"Total: {total_count}")
    print(f"Passed: {ok_count}")
    print(f"Failed: {fail_count}")
    print(f"Skipped:{skip_count}")


if __name__ == "__main__":
    main()
