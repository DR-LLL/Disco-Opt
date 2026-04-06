#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class Instance {
public:
    struct Item {
        int id;                
        long long value;
        long long weight;
    };

    std::string name;
    int n;
    long long capacity;
    std::vector<Item> items;

    Instance() : n(0), capacity(0) {}

    bool loadFromFile(const std::string& filepath, const std::string& filename) {
        name = filename;
        std::ifstream in(filepath.c_str());
        if (!in) {
            return false;
        }

        in >> n >> capacity;
        if (!in) {
            return false;
        }

        items.clear();
        items.reserve(n);

        for (int i = 0; i < n; ++i) {
            long long value, weight;
            in >> value >> weight;
            if (!in) {
                return false;
            }

            Item item;
            item.id = i;
            item.value = value;
            item.weight = weight;
            items.push_back(item);
        }

        return true;
    }
};

class Solver {
private:
    const Instance& instance;
    double timeLimitSeconds;

    std::vector<Instance::Item> branchItems;   
    std::vector<Instance::Item> densityItems; 

    long long bestValue;
    long long bestWeight;
    std::vector<int> bestTaken;             

    std::vector<int> currentTaken;             

    bool timeLimitReached;

    std::chrono::steady_clock::time_point startTime;

private:
    double elapsedSeconds() const {
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = now - startTime;
        return diff.count();
    }

    bool isTimeOver() {
        if (elapsedSeconds() >= timeLimitSeconds) {
            timeLimitReached = true;
            return true;
        }
        return false;
    }

    void printImprovement(long long value) const {
        std::cout << "Improved solution: value = " << value
                  << ", time = " << std::fixed << std::setprecision(6)
                  << elapsedSeconds() << " sec" << std::endl;
    }

    void buildOrders() {
        branchItems = instance.items;
        densityItems = instance.items;

        std::sort(branchItems.begin(), branchItems.end(),
                  [](const Instance::Item& a, const Instance::Item& b) {
                      if (a.value != b.value) {
                          return a.value > b.value;
                      }
                      if (a.weight != b.weight) {
                          return a.weight < b.weight;
                      }
                      return a.id < b.id;
                  });

        std::sort(densityItems.begin(), densityItems.end(),
                  [](const Instance::Item& a, const Instance::Item& b) {
                      long double left = static_cast<long double>(a.value) * b.weight;
                      long double right = static_cast<long double>(b.value) * a.weight;
                      if (left != right) {
                          return left > right;
                      }
                      if (a.value != b.value) {
                          return a.value > b.value;
                      }
                      if (a.weight != b.weight) {
                          return a.weight < b.weight;
                      }
                      return a.id < b.id;
                  });
    }

    void runGreedyInitialSolution() {
        long long totalValue = 0;
        long long totalWeight = 0;
        std::vector<int> taken(instance.n, 0);

        for (size_t i = 0; i < densityItems.size(); ++i) {
            const Instance::Item& item = densityItems[i];
            if (totalWeight + item.weight <= instance.capacity) {
                totalWeight += item.weight;
                totalValue += item.value;
                taken[item.id] = 1;
            }
        }

        bestValue = totalValue;
        bestWeight = totalWeight;
        bestTaken = taken;

        printImprovement(bestValue);
    }

    void tryUpdateBest(long long currentValue,
                       long long currentWeight,
                       const std::vector<int>& takenByBranchOrder) {
        if (currentWeight > instance.capacity) {
            return;
        }

        if (currentValue <= bestValue) {
            return;
        }

        std::vector<int> taken(instance.n, 0);
        for (int i = 0; i < static_cast<int>(takenByBranchOrder.size()); ++i) {
            if (takenByBranchOrder[i]) {
                taken[branchItems[i].id] = 1;
            }
        }

        bestValue = currentValue;
        bestWeight = currentWeight;
        bestTaken = taken;

        printImprovement(bestValue);
    }

    double fractionalUpperBound(int level,
                                long long currentValue,
                                long long currentWeight,
                                const std::vector<int>& takenByBranchOrder,
                                const std::vector<int>& decidedByOriginalId) {
        if (currentWeight > instance.capacity) {
            return -1.0;
        }

        long double bound = static_cast<long double>(currentValue);
        long long remainingCapacity = instance.capacity - currentWeight;

        for (size_t i = 0; i < densityItems.size(); ++i) {
            const Instance::Item& item = densityItems[i];
            int id = item.id;

            if (decidedByOriginalId[id]) {
                continue;
            }

            if (item.weight <= remainingCapacity) {
                remainingCapacity -= item.weight;
                bound += static_cast<long double>(item.value);
            } else {
                if (item.weight > 0) {
                    bound += static_cast<long double>(item.value) *
                             static_cast<long double>(remainingCapacity) /
                             static_cast<long double>(item.weight);
                }
                break;
            }
        }

        return static_cast<double>(bound);
    }

    void dfs(int level,
             long long currentValue,
             long long currentWeight,
             std::vector<int>& takenByBranchOrder,
             std::vector<int>& decidedByOriginalId) {
        if (isTimeOver()) {
            return;
        }

        if (currentWeight > instance.capacity) {
            return;
        }

        double ub = fractionalUpperBound(level, currentValue, currentWeight,
                                         takenByBranchOrder, decidedByOriginalId);

        if (ub <= static_cast<double>(bestValue) + 1e-12) {
            return;
        }

        if (level == instance.n) {
            tryUpdateBest(currentValue, currentWeight, takenByBranchOrder);
            return;
        }

       
        tryUpdateBest(currentValue, currentWeight, takenByBranchOrder);

        const Instance::Item& item = branchItems[level];
        int origId = item.id;

       
        takenByBranchOrder[level] = 1;
        decidedByOriginalId[origId] = 1;

        dfs(level + 1,
            currentValue + item.value,
            currentWeight + item.weight,
            takenByBranchOrder,
            decidedByOriginalId);

        if (timeLimitReached) {
            return;
        }

      
        takenByBranchOrder[level] = 0;
        decidedByOriginalId[origId] = 1;

        dfs(level + 1,
            currentValue,
            currentWeight,
            takenByBranchOrder,
            decidedByOriginalId);

        decidedByOriginalId[origId] = 0;
    }

public:
    Solver(const Instance& inst, double timeLimit)
        : instance(inst),
          timeLimitSeconds(timeLimit),
          bestValue(0),
          bestWeight(0),
          timeLimitReached(false) {}

    void solve() {
        startTime = std::chrono::steady_clock::now();
        timeLimitReached = false;

        buildOrders();

        bestTaken.assign(instance.n, 0);
        currentTaken.assign(instance.n, 0);

        runGreedyInitialSolution();

        std::vector<int> takenByBranchOrder(instance.n, 0);
        std::vector<int> decidedByOriginalId(instance.n, 0);

        dfs(0, 0LL, 0LL, takenByBranchOrder, decidedByOriginalId);

        if (timeLimitReached) {
            std::cout << "Time limit reached. Returning best found solution: value = "
                      << bestValue << std::endl;
        }
    }

    long long getObjectiveValue() const {
        return bestValue;
    }

    long long getUsedWeight() const {
        return bestWeight;
    }

    std::vector<int> getChosenItems() const {
        std::vector<int> answer;
        for (int i = 0; i < instance.n; ++i) {
            if (bestTaken[i]) {
                answer.push_back(i);
            }
        }
        return answer;
    }
};

static bool looksLikeTextFile(const std::string& filepath) {
    std::ifstream in(filepath.c_str(), std::ios::binary);
    if (!in) {
        return false;
    }

    const int BUFFER_SIZE = 4096;
    char buffer[BUFFER_SIZE];
    in.read(buffer, BUFFER_SIZE);
    std::streamsize count = in.gcount();

    for (std::streamsize i = 0; i < count; ++i) {
        unsigned char c = static_cast<unsigned char>(buffer[i]);
        if (c == 0) {
            return false;
        }
    }

    return true;
}

static bool isGoodDataFile(const std::string& folder, const std::string& name) {
    if (name.empty()) return false;
    if (name == "." || name == "..") return false;
    if (name[0] == '.') return false;

    std::string path = folder + "/" + name;
    return looksLikeTextFile(path);
}

static std::vector<std::string> listFilesInDirectory(const std::string& folder) {
    std::vector<std::string> files;

    DIR* dir = opendir(folder.c_str());
    if (dir == NULL) {
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (isGoodDataFile(folder, name)) {
            files.push_back(name);
        }
    }

    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

static std::string joinSolution(const std::vector<int>& solution) {
    std::ostringstream out;
    for (size_t i = 0; i < solution.size(); ++i) {
        if (i) out << ' ';
        out << solution[i];
    }
    return out.str();
}

int main() {
    const std::string DATA_FOLDER = "data";
    const std::string RESULT_FILE = "result.csv";
    const double TIME_LIMIT_SECONDS = 300.0;

    std::vector<std::string> files = listFilesInDirectory(DATA_FOLDER);
    if (files.empty()) {
        std::cerr << "No valid test files found in folder: " << DATA_FOLDER << std::endl;
        return 1;
    }

    std::ofstream out(RESULT_FILE.c_str());
    if (!out) {
        std::cerr << "Cannot open result file: " << RESULT_FILE << std::endl;
        return 1;
    }

    out << "test_name,objective_value,solution\n";

    for (size_t i = 0; i < files.size(); ++i) {
        const std::string& filename = files[i];
        const std::string filepath = DATA_FOLDER + "/" + filename;

        std::cout << "==============================" << std::endl;
        std::cout << "Processing: " << filename << std::endl;

        Instance instance;
        if (!instance.loadFromFile(filepath, filename)) {
            std::cerr << "Failed to read instance: " << filepath << std::endl;
            continue;
        }

        Solver solver(instance, TIME_LIMIT_SECONDS);
        solver.solve();

        std::vector<int> solution = solver.getChosenItems();

        out << filename << ","
            << solver.getObjectiveValue() << ",\""
            << joinSolution(solution) << "\"\n";

        std::cout << "Final for " << filename
                  << ": value = " << solver.getObjectiveValue()
                  << ", weight = " << solver.getUsedWeight()
                  << ", items = " << solution.size()
                  << std::endl;
    }

    out.close();
    std::cout << "Results written to " << RESULT_FILE << std::endl;

    return 0;
}