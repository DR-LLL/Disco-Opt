#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent
TESTS_DIR = ROOT / "tests"
SOLUTIONS_DIR = ROOT / "solutions"
RESULTS_CSV = ROOT / "results.csv"
VALIDATION_CSV = ROOT / "validation_results.csv"


def read_graph(path: Path) -> Tuple[int, List[Tuple[int, int]]]:
    data = list(map(int, path.read_text().split()))
    n, m = data[0], data[1]
    nums = data[2:]
    edges = []
    for i in range(0, 2 * m, 2):
        edges.append((nums[i], nums[i + 1]))
    return n, edges


def parse_solution(text: str, n: int) -> Tuple[int, List[int]]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Solution file must contain at least two non-empty lines")
    declared = int(lines[0])
    colors = list(map(int, lines[1].split()))
    if len(colors) != n:
        raise ValueError(f"Expected {n} colors, got {len(colors)}")
    return declared, colors


def validate(n: int, edges: List[Tuple[int, int]], declared: int, colors: List[int]) -> Tuple[bool, int]:
    if any(c < 0 for c in colors):
        return False, -1
    for u, v in edges:
        if colors[u] == colors[v]:
            return False, -1
    used = len(set(colors))
    if declared != used:
        return False, used
    return True, used


def main() -> None:
    rows = []
    tests = sorted([p for p in TESTS_DIR.iterdir() if p.is_file() and p.stat().st_size > 0], key=lambda p: p.name)

    for test_path in tests:
        solution_path = SOLUTIONS_DIR / f"{test_path.name}.out"
        if not solution_path.exists():
            rows.append({
                "instance": test_path.name,
                "valid": False,
                "colors_used": "MISSING",
                "solution_file": solution_path.name,
                "note": "Run solver.py first",
            })
            continue

        n, edges = read_graph(test_path)
        try:
            declared, colors = parse_solution(solution_path.read_text(encoding="utf-8"), n)
            valid, used = validate(n, edges, declared, colors)
            rows.append({
                "instance": test_path.name,
                "valid": valid,
                "colors_used": used if valid else "INVALID",
                "solution_file": solution_path.name,
                "note": "" if valid else "Incorrect coloring or declared color count mismatch",
            })
        except Exception as exc:
            rows.append({
                "instance": test_path.name,
                "valid": False,
                "colors_used": "PARSE_ERR",
                "solution_file": solution_path.name,
                "note": str(exc),
            })

    with VALIDATION_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "valid", "colors_used", "solution_file", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved validation results to {VALIDATION_CSV}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
