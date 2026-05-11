#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

static const string TESTS_DIR = "tests";
static const string SOLUTIONS_DIR = "solutions";
static const string RESULTS_CSV = "results.csv";

static const int DEFAULT_RUNS_PER_START = 40;
static const uint64_t DEFAULT_RANDOM_SEED = 123445ull;
static const double DEFAULT_IMPROVEMENT_TIME_LIMIT_SEC = 240.0;

static const double ALPHA_MIN = 0.90;
static const double ALPHA_MAX = 1.00;

struct Graph {
    int n = 0;
    vector<pair<int, int>> edges;
    vector<vector<int>> adj;
};

struct ColoringResult {
    int colors_used = 0;
    vector<int> color;
};

struct SolveResult {
    int best_k = 0;
    double elapsed = 0.0;
    int primary_runs = 0;
    long long squeeze_attempts = 0;
    long long squeeze_successes = 0;
    long long improvement_iterations = 0;
    long long improvement_successes = 0;
};

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

Graph read_graph(const string& path) {
    ifstream fin(path);
    if (!fin) {
        throw runtime_error("Cannot open file: " + path);
    }

    int n, m;
    fin >> n >> m;

    Graph g;
    g.n = n;
    g.adj.assign(n, {});

    for (int i = 0; i < m; ++i) {
        int u, v;
        fin >> u >> v;

        if (u < 0 || u >= n || v < 0 || v >= n) {
            throw runtime_error("Bad edge endpoint in file: " + path);
        }

        if (u == v) continue;

        g.edges.push_back({u, v});
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
    }

    return g;
}

uint64_t graph_hash(const Graph& g) {
    uint64_t h = splitmix64(static_cast<uint64_t>(g.n));
    h ^= splitmix64(static_cast<uint64_t>(g.edges.size()) + 17ull);

    for (auto [u, v] : g.edges) {
        uint64_t a = static_cast<uint64_t>(min(u, v));
        uint64_t b = static_cast<uint64_t>(max(u, v));
        h ^= splitmix64((a + 1ull) * 1000003ull + (b + 1ull));
    }

    return h;
}

int score_key(int saturation, int degree, int n) {
    return saturation * (n + 1) + degree;
}

void erase_from_bucket(map<int, set<int>>& buckets, int key, int v) {
    auto it = buckets.find(key);
    if (it == buckets.end()) return;

    it->second.erase(v);

    if (it->second.empty()) {
        buckets.erase(it);
    }
}

void insert_to_bucket(map<int, set<int>>& buckets, int key, int v) {
    buckets[key].insert(v);
}

int choose_random_from_set(const set<int>& s, mt19937_64& rng) {
    if (s.empty()) return -1;

    uniform_int_distribution<int> dist(0, static_cast<int>(s.size()) - 1);
    int idx = dist(rng);

    auto it = s.begin();
    advance(it, idx);

    return *it;
}

int choose_random_best_vertex(const map<int, set<int>>& buckets, mt19937_64& rng) {
    if (buckets.empty()) return -1;
    return choose_random_from_set(buckets.rbegin()->second, rng);
}

bool is_valid_coloring(const Graph& g, const vector<int>& color) {
    if (static_cast<int>(color.size()) != g.n) return false;

    for (int c : color) {
        if (c < 0) return false;
    }

    for (auto [u, v] : g.edges) {
        if (color[u] == color[v]) return false;
    }

    return true;
}

int count_colors(const vector<int>& color) {
    int mx = -1;

    for (int c : color) {
        mx = max(mx, c);
    }

    return mx + 1;
}

ColoringResult normalize_coloring_labels(const vector<int>& color) {
    int mx = -1;

    for (int c : color) {
        if (c < 0) {
            throw runtime_error("Internal error: cannot normalize incomplete coloring");
        }
        mx = max(mx, c);
    }

    vector<int> old_to_new(mx + 1, -1);
    int next_color = 0;

    for (int c = 0; c <= mx; ++c) {
        bool exists = false;

        for (int x : color) {
            if (x == c) {
                exists = true;
                break;
            }
        }

        if (exists) {
            old_to_new[c] = next_color++;
        }
    }

    vector<int> normalized = color;

    for (int& c : normalized) {
        c = old_to_new[c];
    }

    return {next_color, normalized};
}

vector<vector<int>> color_classes_by_color(const vector<int>& color) {
    int k = count_colors(color);
    vector<vector<int>> classes(k);

    for (int v = 0; v < static_cast<int>(color.size()); ++v) {
        int c = color[v];

        if (c < 0 || c >= k) {
            throw runtime_error("Internal error: bad color while building color classes");
        }

        classes[c].push_back(v);
    }

    return classes;
}

vector<vector<int>> sorted_color_classes_desc(const ColoringResult& coloring) {
    vector<vector<int>> classes = color_classes_by_color(coloring.color);

    classes.erase(
        remove_if(classes.begin(), classes.end(), [](const vector<int>& cls) {
            return cls.empty();
        }),
        classes.end()
    );

    sort(classes.begin(), classes.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a.size() != b.size()) {
            return a.size() > b.size();
        }

        return *min_element(a.begin(), a.end()) < *min_element(b.begin(), b.end());
    });

    return classes;
}

vector<int> make_initial_coloring_from_class_prefix(
    int n,
    const vector<vector<int>>& classes,
    int prefix_len
) {
    vector<int> initial_color(n, -1);

    for (int class_id = 0; class_id < prefix_len; ++class_id) {
        for (int v : classes[class_id]) {
            initial_color[v] = class_id;
        }
    }

    return initial_color;
}

ColoringResult squeeze_coloring_by_small_classes(
    const Graph& g,
    const ColoringResult& input
) {
    if (!is_valid_coloring(g, input.color)) {
        throw runtime_error("Internal error: squeeze received invalid coloring");
    }

    ColoringResult normalized = normalize_coloring_labels(input.color);
    vector<int> color = normalized.color;

    while (true) {
        int k = count_colors(color);

        if (k <= 1) break;

        vector<vector<int>> classes = color_classes_by_color(color);

        vector<int> class_order(k);
        iota(class_order.begin(), class_order.end(), 0);

        sort(class_order.begin(), class_order.end(), [&](int a, int b) {
            if (classes[a].size() != classes[b].size()) {
                return classes[a].size() < classes[b].size();
            }

            return *min_element(classes[a].begin(), classes[a].end())
                 < *min_element(classes[b].begin(), classes[b].end());
        });

        bool removed_one_class = false;

        for (int class_id : class_order) {
            vector<int> candidate = color;

            vector<int> class_size(k, 0);
            for (int c = 0; c < k; ++c) {
                class_size[c] = static_cast<int>(classes[c].size());
            }

            bool can_remove_class = true;

            for (int v : classes[class_id]) {
                vector<unsigned char> forbidden(k, 0);

                for (int to : g.adj[v]) {
                    int neighbor_color = candidate[to];

                    if (neighbor_color >= 0 && neighbor_color < k) {
                        forbidden[neighbor_color] = 1;
                    }
                }

                int target_color = -1;
                int best_target_size = -1;

                for (int c = 0; c < k; ++c) {
                    if (c == class_id) continue;
                    if (forbidden[c]) continue;

                    if (
                        class_size[c] > best_target_size ||
                        (class_size[c] == best_target_size && (target_color == -1 || c < target_color))
                    ) {
                        best_target_size = class_size[c];
                        target_color = c;
                    }
                }

                if (target_color == -1) {
                    can_remove_class = false;
                    break;
                }

                candidate[v] = target_color;
                --class_size[class_id];
                ++class_size[target_color];
            }

            if (!can_remove_class) continue;

            ColoringResult squeezed = normalize_coloring_labels(candidate);

            if (squeezed.colors_used >= k) continue;

            if (!is_valid_coloring(g, squeezed.color)) {
                throw runtime_error("Internal error: squeeze produced invalid coloring");
            }

            color = squeezed.color;
            removed_one_class = true;
            break;
        }

        if (!removed_one_class) break;
    }

    return normalize_coloring_labels(color);
}

ColoringResult randomized_dsatur_with_initial_coloring(
    const Graph& g,
    const vector<int>& initial_color,
    mt19937_64& rng
) {
    const int n = g.n;

    if (static_cast<int>(initial_color.size()) != n) {
        throw runtime_error("Internal error: bad initial coloring size");
    }

    vector<int> color = initial_color;

    vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v) {
        degree[v] = static_cast<int>(g.adj[v].size());
    }

    int colored_count = 0;
    int max_color = -1;

    for (int v = 0; v < n; ++v) {
        if (color[v] != -1) {
            if (color[v] < 0 || color[v] >= n) {
                throw runtime_error("Internal error: bad initial color value");
            }

            ++colored_count;
            max_color = max(max_color, color[v]);
        }
    }

    for (auto [u, v] : g.edges) {
        if (color[u] != -1 && color[u] == color[v]) {
            throw runtime_error("Internal error: conflicting fixed color classes");
        }
    }

    vector<vector<unsigned char>> seen_color(n, vector<unsigned char>(n, 0));
    vector<int> saturation(n, 0);

    for (int v = 0; v < n; ++v) {
        if (color[v] != -1) continue;

        for (int to : g.adj[v]) {
            int c = color[to];

            if (c == -1) continue;

            if (!seen_color[v][c]) {
                seen_color[v][c] = 1;
                ++saturation[v];
            }
        }
    }

    map<int, set<int>> buckets;

    for (int v = 0; v < n; ++v) {
        if (color[v] == -1) {
            insert_to_bucket(buckets, score_key(saturation[v], degree[v], n), v);
        }
    }

    auto color_vertex = [&](int v, int c) {
        erase_from_bucket(buckets, score_key(saturation[v], degree[v], n), v);

        color[v] = c;
        max_color = max(max_color, c);

        for (int to : g.adj[v]) {
            if (color[to] != -1) continue;

            if (!seen_color[to][c]) {
                erase_from_bucket(buckets, score_key(saturation[to], degree[to], n), to);
                seen_color[to][c] = 1;
                ++saturation[to];
                insert_to_bucket(buckets, score_key(saturation[to], degree[to], n), to);
            }
        }
    };

    while (colored_count < n) {
        int v = choose_random_best_vertex(buckets, rng);

        if (v == -1) {
            throw runtime_error("Internal error: no uncolored vertex to choose");
        }

        int c = 0;

        while (c < n && seen_color[v][c]) {
            ++c;
        }

        if (c >= n) {
            throw runtime_error("Internal error: no available color found");
        }

        color_vertex(v, c);
        ++colored_count;
    }

    return {max_color + 1, color};
}

ColoringResult semi_randomized_dsatur_with_initial_coloring(
    const Graph& g,
    const vector<int>& initial_color,
    mt19937_64& rng
) {
    const int n = g.n;

    if (static_cast<int>(initial_color.size()) != n) {
        throw runtime_error("Internal error: bad initial coloring size");
    }

    vector<int> color = initial_color;

    vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v) {
        degree[v] = static_cast<int>(g.adj[v].size());
    }

    int colored_count = 0;
    int max_color = -1;

    for (int v = 0; v < n; ++v) {
        if (color[v] != -1) {
            if (color[v] < 0 || color[v] >= n) {
                throw runtime_error("Internal error: bad initial color value");
            }

            ++colored_count;
            max_color = max(max_color, color[v]);
        }
    }

    for (auto [u, v] : g.edges) {
        if (color[u] != -1 && color[u] == color[v]) {
            throw runtime_error("Internal error: conflicting fixed color classes");
        }
    }

    vector<vector<unsigned char>> seen_color(n, vector<unsigned char>(n, 0));
    vector<int> saturation(n, 0);

    for (int v = 0; v < n; ++v) {
        if (color[v] != -1) continue;

        for (int to : g.adj[v]) {
            int c = color[to];

            if (c == -1) continue;

            if (!seen_color[v][c]) {
                seen_color[v][c] = 1;
                ++saturation[v];
            }
        }
    }

    map<int, set<int>> buckets;
    set<int> uncolored;

    for (int v = 0; v < n; ++v) {
        if (color[v] == -1) {
            insert_to_bucket(buckets, score_key(saturation[v], degree[v], n), v);
            uncolored.insert(v);
        }
    }

    auto color_vertex = [&](int v, int c) {
        erase_from_bucket(buckets, score_key(saturation[v], degree[v], n), v);
        uncolored.erase(v);

        color[v] = c;
        max_color = max(max_color, c);

        for (int to : g.adj[v]) {
            if (color[to] != -1) continue;

            if (!seen_color[to][c]) {
                erase_from_bucket(buckets, score_key(saturation[to], degree[to], n), to);
                seen_color[to][c] = 1;
                ++saturation[to];
                insert_to_bucket(buckets, score_key(saturation[to], degree[to], n), to);
            }
        }
    };

    uniform_real_distribution<double> alpha_dist(ALPHA_MIN, ALPHA_MAX);
    uniform_real_distribution<double> coin_dist(0.0, 1.0);

    while (colored_count < n) {
        double alpha = alpha_dist(rng);

        int v;
        if (coin_dist(rng) <= alpha) {
            v = choose_random_best_vertex(buckets, rng);
        } else {
            v = choose_random_from_set(uncolored, rng);
        }

        if (v == -1) {
            throw runtime_error("Internal error: no uncolored vertex to choose");
        }

        int c = 0;

        while (c < n && seen_color[v][c]) {
            ++c;
        }

        if (c >= n) {
            throw runtime_error("Internal error: no available color found");
        }

        color_vertex(v, c);
        ++colored_count;
    }

    return {max_color + 1, color};
}

ColoringResult randomized_dsatur_from_start(
    const Graph& g,
    int start_vertex,
    mt19937_64& rng
) {
    vector<int> initial_color(g.n, -1);
    initial_color[start_vertex] = 0;

    return randomized_dsatur_with_initial_coloring(g, initial_color, rng);
}

SolveResult solve_one(
    const string& test_name,
    const string& path,
    int runs_per_start,
    uint64_t global_seed,
    double improvement_time_limit_sec
) {
    Graph g = read_graph(path);
    uint64_t g_hash = graph_hash(g);

    auto start_time = chrono::steady_clock::now();

    int best_k = g.n + 1;
    vector<int> best_coloring(g.n, -1);

    int primary_runs = g.n * runs_per_start;
    int completed_primary_runs = 0;

    long long squeeze_attempts = 0;
    long long squeeze_successes = 0;
    long long improvement_iterations = 0;
    long long improvement_successes = 0;

    auto try_update_best = [&](
        const ColoringResult& result,
        const string& phase,
        int start_vertex,
        int run_id,
        int prefix_len
    ) {
        if (!is_valid_coloring(g, result.color)) {
            throw runtime_error("Internal error: invalid coloring produced for " + test_name);
        }

        if (result.colors_used < best_k) {
            int old_best = best_k;

            best_k = result.colors_used;
            best_coloring = result.color;

            if (phase.find("improve") != string::npos) {
                ++improvement_successes;
            }

            cout << "[" << test_name << "] new best: "
                 << best_k << " colors";

            if (old_best <= g.n) {
                cout << " | old=" << old_best;
            }

            cout << " | phase=" << phase;

            if (start_vertex >= 0) {
                cout << " | start=" << start_vertex;
            }

            if (run_id >= 0) {
                cout << " | run=" << (run_id + 1) << "/" << runs_per_start;
            }

            if (prefix_len >= 0) {
                cout << " | fixed_prefix=" << prefix_len;
            }

            cout << " | primary_done=" << completed_primary_runs << "/" << primary_runs
                 << " | improvement_iter=" << improvement_iterations
                 << " | squeeze_attempts=" << squeeze_attempts
                 << " | squeeze_successes=" << squeeze_successes
                 << '\n';
        }
    };

    for (int start_vertex = 0; start_vertex < g.n; ++start_vertex) {
        for (int run_id = 0; run_id < runs_per_start; ++run_id) {
            uint64_t seed = splitmix64(
                global_seed
                ^ g_hash
                ^ (static_cast<uint64_t>(start_vertex) + 1ull) * 1000003ull
                ^ (static_cast<uint64_t>(run_id) + 1ull) * 9176ull
            );

            mt19937_64 rng(seed);

            ColoringResult current = randomized_dsatur_from_start(
                g,
                start_vertex,
                rng
            );

            ++completed_primary_runs;

            try_update_best(
                current,
                "dsatur",
                start_vertex,
                run_id,
                -1
            );

            ColoringResult squeezed = squeeze_coloring_by_small_classes(g, current);
            ++squeeze_attempts;

            if (squeezed.colors_used < current.colors_used) {
                ++squeeze_successes;
            }

            try_update_best(
                squeezed,
                "squeeze-after-dsatur",
                start_vertex,
                run_id,
                -1
            );
        }
    }

    if (best_k <= g.n && improvement_time_limit_sec > 0.0) {
        uint64_t improve_seed = splitmix64(
            global_seed
            ^ g_hash
            ^ 0xd1b54a32d192ed03ull
        );

        mt19937_64 improve_rng(improve_seed);

        auto improve_start_time = chrono::steady_clock::now();

        while (true) {
            double passed = chrono::duration<double>(
                chrono::steady_clock::now() - improve_start_time
            ).count();

            if (passed >= improvement_time_limit_sec) {
                break;
            }

            ColoringResult best_result{best_k, best_coloring};
            vector<vector<int>> classes = sorted_color_classes_desc(best_result);

            int k = static_cast<int>(classes.size());

            if (k <= 1) {
                break;
            }

            double expected = sqrt(static_cast<double>(k));
            expected = max(1.0, expected);

            double p = 1.0 / expected;
            p = min(1.0, max(1e-9, p));

            geometric_distribution<int> geom_dist(p);

            int prefix_len = geom_dist(improve_rng) + 1;

            if (prefix_len > k) {
                prefix_len = k;
            }

            vector<int> initial_color = make_initial_coloring_from_class_prefix(
                g.n,
                classes,
                prefix_len
            );

            ColoringResult improved = semi_randomized_dsatur_with_initial_coloring(
                g,
                initial_color,
                improve_rng
            );

            ++improvement_iterations;

            try_update_best(
                improved,
                "improve-semirandom-dsatur",
                -1,
                -1,
                prefix_len
            );

            ColoringResult squeezed = squeeze_coloring_by_small_classes(g, improved);
            ++squeeze_attempts;

            if (squeezed.colors_used < improved.colors_used) {
                ++squeeze_successes;
            }

            try_update_best(
                squeezed,
                "squeeze-after-improve-semirandom-dsatur",
                -1,
                -1,
                prefix_len
            );
        }
    }

    filesystem::create_directories(SOLUTIONS_DIR);

    string out_path = SOLUTIONS_DIR + "/" + test_name + ".out";
    ofstream fout(out_path);

    if (!fout) {
        throw runtime_error("Cannot write solution file: " + out_path);
    }

    fout << best_k << "\n";

    for (int i = 0; i < g.n; ++i) {
        if (i) fout << ' ';
        fout << best_coloring[i];
    }

    fout << "\n";

    double elapsed = chrono::duration<double>(
        chrono::steady_clock::now() - start_time
    ).count();

    return {
        best_k,
        elapsed,
        primary_runs,
        squeeze_attempts,
        squeeze_successes,
        improvement_iterations,
        improvement_successes
    };
}

int main(int argc, char** argv) {
    try {
        int runs_per_start = DEFAULT_RUNS_PER_START;
        uint64_t global_seed = DEFAULT_RANDOM_SEED;
        double improvement_time_limit_sec = DEFAULT_IMPROVEMENT_TIME_LIMIT_SEC;

        if (argc >= 2) {
            runs_per_start = stoi(argv[1]);

            if (runs_per_start <= 0) {
                throw runtime_error("runs_per_start must be positive");
            }
        }

        if (argc >= 3) {
            global_seed = stoull(argv[2]);
        }

        if (argc >= 4) {
            improvement_time_limit_sec = stod(argv[3]);

            if (improvement_time_limit_sec < 0.0) {
                throw runtime_error("improvement_time_limit_sec must be non-negative");
            }
        }

        if (!filesystem::exists(TESTS_DIR) || !filesystem::is_directory(TESTS_DIR)) {
            cout << "SEED_USED=" << global_seed << '\n';
            return 0;
        }

        vector<string> test_files;

        for (const auto& entry : filesystem::directory_iterator(TESTS_DIR)) {
            if (!entry.is_regular_file()) continue;

            string name = entry.path().filename().string();

            if (!name.empty() && name[0] != '.') {
                test_files.push_back(name);
            }
        }

        sort(test_files.begin(), test_files.end());

        if (test_files.empty()) {
            cout << "SEED_USED=" << global_seed << '\n';
            return 0;
        }

        ofstream csv(RESULTS_CSV);

        if (!csv) {
            throw runtime_error("Cannot write " + RESULTS_CSV);
        }

        csv << "instance,colors_used,runtime_sec,primary_runs,"
            << "squeeze_attempts,squeeze_successes,"
            << "improvement_iterations,improvement_successes,"
            << "runs_per_start,seed,improvement_time_limit_sec\n";

        for (const string& test_name : test_files) {
            string path = TESTS_DIR + "/" + test_name;

            SolveResult result = solve_one(
                test_name,
                path,
                runs_per_start,
                global_seed,
                improvement_time_limit_sec
            );

            csv << test_name << ","
                << result.best_k << ","
                << result.elapsed << ","
                << result.primary_runs << ","
                << result.squeeze_attempts << ","
                << result.squeeze_successes << ","
                << result.improvement_iterations << ","
                << result.improvement_successes << ","
                << runs_per_start << ","
                << global_seed << ","
                << improvement_time_limit_sec << "\n";
        }

        cout << "SEED_USED=" << global_seed << '\n';
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}