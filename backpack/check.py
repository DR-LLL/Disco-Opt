import csv
import os
import sys


class Instance:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.n = 0
        self.capacity = 0
        self.values = []
        self.weights = []
        self._read()

    def _read(self):
        with open(self.path, "r", encoding="utf-8", errors="strict") as f:
            tokens = f.read().strip().split()

        if len(tokens) < 2:
            raise ValueError(f"{self.name}: file is too short")

        self.n = int(tokens[0])
        self.capacity = int(tokens[1])

        expected = 2 + 2 * self.n
        if len(tokens) != expected:
            raise ValueError(
                f"{self.name}: expected {expected} tokens, got {len(tokens)}"
            )

        self.values = []
        self.weights = []

        pos = 2
        for _ in range(self.n):
            value = int(tokens[pos])
            weight = int(tokens[pos + 1])
            pos += 2
            self.values.append(value)
            self.weights.append(weight)


def parse_solution(sol_text):
    sol_text = sol_text.strip()
    if sol_text == "":
        return []

    parts = sol_text.split()
    return [int(p) for p in parts]


def looks_like_text_file(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(4096)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except Exception:
        return False


def is_test_file(filename, full_path):
    if not os.path.isfile(full_path):
        return False
    if filename.startswith("."):
        return False
    if filename == "result.csv":
        return False
    return looks_like_text_file(full_path)


def load_instances(data_folder):
    instances = {}
    for filename in sorted(os.listdir(data_folder)):
        full_path = os.path.join(data_folder, filename)

        if not is_test_file(filename, full_path):
            print(f"SKIP: {filename}")
            continue

        try:
            instances[filename] = Instance(full_path, filename)
        except Exception as e:
            print(f"SKIP: {filename} ({e})")

    return instances


def check_result(data_folder, result_csv):
    instances = load_instances(data_folder)

    if not instances:
        print("ERROR: no valid test instances found in data folder")
        return 1

    if not os.path.exists(result_csv):
        print(f"ERROR: result file not found: {result_csv}")
        return 1

    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_fields = ["test_name", "objective_value", "solution"]

        if reader.fieldnames is None:
            print("ERROR: result.csv is empty")
            return 1

        for field in required_fields:
            if field not in reader.fieldnames:
                print(f"ERROR: missing column '{field}' in result.csv")
                return 1

        seen_tests = set()
        ok = True

        for row_num, row in enumerate(reader, start=2):
            test_name = row["test_name"].strip()
            objective_str = row["objective_value"].strip()
            solution_str = row["solution"].strip()

            if test_name == "":
                print(f"ERROR line {row_num}: empty test_name")
                ok = False
                continue

            if test_name not in instances:
                print(f"ERROR line {row_num}: unknown test '{test_name}'")
                ok = False
                continue

            if test_name in seen_tests:
                print(f"ERROR line {row_num}: duplicate test '{test_name}'")
                ok = False
                continue

            seen_tests.add(test_name)
            inst = instances[test_name]

            try:
                claimed_objective = int(objective_str)
            except ValueError:
                print(
                    f"ERROR line {row_num}: objective_value is not integer "
                    f"for test '{test_name}'"
                )
                ok = False
                continue

            try:
                solution = parse_solution(solution_str)
            except ValueError:
                print(
                    f"ERROR line {row_num}: solution contains non-integer "
                    f"indices for test '{test_name}'"
                )
                ok = False
                continue

            used = set()
            total_value = 0
            total_weight = 0
            local_ok = True

            for item in solution:
                if item < 0 or item >= inst.n:
                    print(
                        f"ERROR line {row_num}: item index {item} out of range "
                        f"[0, {inst.n - 1}] in test '{test_name}'"
                    )
                    ok = False
                    local_ok = False
                    break

                if item in used:
                    print(
                        f"ERROR line {row_num}: duplicate item index {item} "
                        f"in test '{test_name}'"
                    )
                    ok = False
                    local_ok = False
                    break

                used.add(item)
                total_value += inst.values[item]
                total_weight += inst.weights[item]

            if not local_ok:
                continue

            if total_weight > inst.capacity:
                print(
                    f"ERROR line {row_num}: infeasible solution in test "
                    f"'{test_name}' (weight {total_weight} > capacity {inst.capacity})"
                )
                ok = False
                continue

            if total_value != claimed_objective:
                print(
                    f"ERROR line {row_num}: wrong objective in test '{test_name}' "
                    f"(claimed {claimed_objective}, actual {total_value})"
                )
                ok = False
                continue

            print(
                f"OK: {test_name} | value = {total_value} | "
                f"weight = {total_weight}/{inst.capacity} | "
                f"items = {len(solution)}"
            )

        missing = sorted(set(instances.keys()) - seen_tests)
        if missing:
            print("WARNING: some tests are missing in result.csv:")
            for name in missing:
                print(f"  {name}")

    if ok:
        print("All checked solutions are feasible and objective values are correct.")
        return 0
    else:
        print("There were errors.")
        return 1


if __name__ == "__main__":
    DATA_FOLDER = "data"
    RESULT_FILE = "result.csv"
    sys.exit(check_result(DATA_FOLDER, RESULT_FILE))
