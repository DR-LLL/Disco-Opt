#!/usr/bin/env python3

import csv
import math
import sys
from pathlib import Path


def read_instance(path):
    with open(path, "r") as f:
        n, v, cap = map(int, f.readline().split())

        demand = []
        x = []
        y = []

        for _ in range(n):
            parts = f.readline().split()
            demand.append(int(parts[0]))
            x.append(float(parts[1]))
            y.append(float(parts[2]))

    return n, v, cap, demand, x, y


def dist(i, j, x, y):
    return math.hypot(x[i] - x[j], y[i] - y[j])


def check_output(text, inst):
    n, v, cap, demand, x, y = inst

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    if len(lines) < v + 1:
        raise ValueError(f"not enough output lines: expected at least {v + 1}, got {len(lines)}")

    reported = float(lines[0].split()[0])

    seen = [0] * n
    total = 0.0

    for r_id in range(v):
        route = list(map(int, lines[1 + r_id].split()))

        if len(route) < 2:
            raise ValueError(f"route {r_id}: too short")

        if route[0] != 0 or route[-1] != 0:
            raise ValueError(f"route {r_id}: route must start and end with 0")

        load = 0

        for node in route[1:-1]:
            if node <= 0 or node >= n:
                raise ValueError(f"route {r_id}: bad customer index {node}")

            seen[node] += 1
            load += demand[node]

        if load > cap:
            raise ValueError(f"route {r_id}: capacity exceeded: {load} > {cap}")

        for a, b in zip(route, route[1:]):
            total += dist(a, b, x, y)

    bad = [i for i in range(1, n) if seen[i] != 1]

    if bad:
        raise ValueError(f"customers with wrong visit count: {bad[:20]}")

    return reported, total


def resolve_path(data_dir, value):
    p = Path(value)

    if p.is_absolute():
        return p

    return data_dir / p


def main():
    script_dir = Path(__file__).resolve().parent

    if len(sys.argv) >= 2:
        results_path = Path(sys.argv[1]).resolve()
    else:
        results_path = script_dir / "Data" / "results.csv"

    data_dir = results_path.parent
    checked_path = data_dir / "checked_results.csv"

    if not results_path.exists():
        print(f"ERROR: results.csv not found: {results_path}")
        sys.exit(1)

    rows_out = []

    with open(results_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            test_name = row.get("file", "").strip()
            solution_name = row.get("solution_file", "").strip()

            check_status = "OK"
            note = ""
            reported = ""
            real = ""
            csv_cost = row.get("cost", "")
            abs_error = ""

            try:
                if not test_name:
                    raise ValueError("empty file field in csv")

                if not solution_name:
                    raise ValueError("empty solution_file field in csv")

                inst_path = resolve_path(data_dir, test_name)
                sol_path = resolve_path(data_dir, solution_name)

                if not inst_path.exists():
                    raise ValueError(f"instance file not found: {inst_path}")

                if not sol_path.exists():
                    raise ValueError(f"solution file not found: {sol_path}")

                inst = read_instance(inst_path)

                with open(sol_path, "r") as sf:
                    text = sf.read()

                reported_value, real_value = check_output(text, inst)

                reported = f"{reported_value:.6f}"
                real = f"{real_value:.6f}"

                if csv_cost:
                    try:
                        abs_error_value = abs(float(csv_cost) - real_value)
                        abs_error = f"{abs_error_value:.6f}"
                    except Exception:
                        abs_error = ""

                if abs(reported_value - real_value) > 1e-4:
                    check_status = "WARN"
                    note = f"reported cost differs from recomputed: {reported_value:.6f} vs {real_value:.6f}"

            except Exception as e:
                check_status = "BAD"
                note = str(e)

            new_row = dict(row)
            new_row["check_status"] = check_status
            new_row["reported_in_solution"] = reported
            new_row["recomputed_cost"] = real
            new_row["csv_cost_abs_error"] = abs_error
            new_row["note"] = note

            rows_out.append(new_row)

            if check_status == "OK":
                print(f"{test_name}: OK, cost={real}")
            elif check_status == "WARN":
                print(f"{test_name}: WARN, cost={real}, {note}")
            else:
                print(f"{test_name}: BAD, {note}")

    if not rows_out:
        print("ERROR: no rows in results.csv")
        sys.exit(1)

    fieldnames = list(rows_out[0].keys())

    with open(checked_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\nchecked csv saved to: {checked_path}")


if __name__ == "__main__":
    main()
