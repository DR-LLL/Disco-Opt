#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

static const double TIME_LIMIT_SECONDS = 120.0;
static const string TESTS_DIR = "tests";
static const string SOLUTIONS_DIR = "solutions";
static const string RESULTS_CSV = "results.csv";

struct Graph {
    int n;
    vector<pair<int, int>> edges;
    vector<vector<int>> adj;
};

uint32_t stable_name_hash(const string& s) {
    uint32_t h = 0;
    for (char ch : s) {
        h = h * 239017u + static_cast<unsigned char>(ch);
    }
    return h;
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
        if (u == v) continue;
        g.edges.push_back({u, v});
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
    }

    return g;
}

pair<int, vector<int>> dsatur_coloring(const Graph& g) {
    int n = g.n;
    vector<int> color(n, -1);
    vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v) {
        degree[v] = static_cast<int>(g.adj[v].size());
    }

    vector<int> saturation(n, 0);
    vector<int> uncolored_degree = degree;
    vector<int> mark(n, -1);

    int colored_count = 0;
    int max_color = -1;

    while (colored_count < n) {
        int best_v = -1;
        int best_sat = -1;
        int best_uncolored_deg = -1;
        int best_deg = -1;

        for (int v = 0; v < n; ++v) {
            if (color[v] != -1) continue;

            if (best_v == -1 ||
                saturation[v] > best_sat ||
                (saturation[v] == best_sat && uncolored_degree[v] > best_uncolored_deg) ||
                (saturation[v] == best_sat && uncolored_degree[v] == best_uncolored_deg && degree[v] > best_deg) ||
                (saturation[v] == best_sat && uncolored_degree[v] == best_uncolored_deg && degree[v] == best_deg && v < best_v)) {
                best_v = v;
                best_sat = saturation[v];
                best_uncolored_deg = uncolored_degree[v];
                best_deg = degree[v];
            }
        }

        for (int to : g.adj[best_v]) {
            int c = color[to];
            if (c != -1) {
                mark[c] = best_v;
            }
        }

        int c = 0;
        while (c < n && mark[c] == best_v) {
            ++c;
        }

        color[best_v] = c;
        max_color = max(max_color, c);
        ++colored_count;

        for (int to : g.adj[best_v]) {
            if (color[to] == -1) {
                --uncolored_degree[to];
            }
        }

        for (int v : g.adj[best_v]) {
            if (color[v] != -1) continue;

            int sat = 0;
            for (int to : g.adj[v]) {
                int cc = color[to];
                if (cc != -1 && mark[cc] != v) {
                    mark[cc] = v;
                    ++sat;
                }
            }
            saturation[v] = sat;
        }
    }

    return {max_color + 1, color};
}

vector<int> random_order(int n, mt19937_64& rng) {
    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    shuffle(order.begin(), order.end(), rng);
    return order;
}

int choose_dsatur_vertex(
    const vector<int>& color,
    const vector<int>& saturation,
    const vector<int>& uncolored_degree,
    const vector<int>& degree
) {
    int n = static_cast<int>(color.size());
    int best_v = -1;
    int best_sat = -1;
    int best_uncolored_deg = -1;
    int best_deg = -1;

    for (int v = 0; v < n; ++v) {
        if (color[v] != -1) continue;

        if (best_v == -1 ||
            saturation[v] > best_sat ||
            (saturation[v] == best_sat && uncolored_degree[v] > best_uncolored_deg) ||
            (saturation[v] == best_sat && uncolored_degree[v] == best_uncolored_deg && degree[v] > best_deg) ||
            (saturation[v] == best_sat && uncolored_degree[v] == best_uncolored_deg && degree[v] == best_deg && v < best_v)) {
            best_v = v;
            best_sat = saturation[v];
            best_uncolored_deg = uncolored_degree[v];
            best_deg = degree[v];
        }
    }

    return best_v;
}

int choose_permutation_vertex(const vector<int>& color, const vector<int>& order, int& ptr) {
    int n = static_cast<int>(order.size());
    while (ptr < n && color[order[ptr]] != -1) {
        ++ptr;
    }
    if (ptr == n) return -1;
    return order[ptr];
}

pair<int, vector<int>> semi_randomized_dsatur(
    const Graph& g,
    const vector<int>& order,
    double alpha,
    mt19937_64& rng
) {
    int n = g.n;

    vector<int> color(n, -1);
    vector<int> degree(n, 0);
    for (int v = 0; v < n; ++v) {
        degree[v] = static_cast<int>(g.adj[v].size());
    }

    vector<int> saturation(n, 0);
    vector<int> uncolored_degree = degree;
    vector<int> mark(n, -1);

    uniform_real_distribution<double> coin_dist(0.0, 1.0);

    int colored_count = 0;
    int max_color = -1;
    int perm_ptr = 0;

    while (colored_count < n) {
        int v = -1;
        double coin = coin_dist(rng);

        if (coin < alpha) {
            v = choose_dsatur_vertex(color, saturation, uncolored_degree, degree);
        } else {
            v = choose_permutation_vertex(color, order, perm_ptr);
            if (v == -1) {
                v = choose_dsatur_vertex(color, saturation, uncolored_degree, degree);
            }
        }

        for (int to : g.adj[v]) {
            int c = color[to];
            if (c != -1) {
                mark[c] = v;
            }
        }

        int c = 0;
        while (c < n && mark[c] == v) {
            ++c;
        }

        color[v] = c;
        max_color = max(max_color, c);
        ++colored_count;

        for (int to : g.adj[v]) {
            if (color[to] == -1) {
                --uncolored_degree[to];
            }
        }

        for (int u : g.adj[v]) {
            if (color[u] != -1) continue;

            int sat = 0;
            for (int to : g.adj[u]) {
                int cc = color[to];
                if (cc != -1 && mark[cc] != u) {
                    mark[cc] = u;
                    ++sat;
                }
            }
            saturation[u] = sat;
        }
    }

    return {max_color + 1, color};
}

struct SolveResult {
    int best_k;
    double elapsed;
    int iterations;
};

SolveResult solve_one(const string& test_name, const string& path, uint64_t global_seed) {
    Graph g = read_graph(path);

    uint64_t local_seed =
        global_seed ^
        static_cast<uint64_t>(stable_name_hash(test_name)) ^
        static_cast<uint64_t>(g.n) * 1000003ull;

    mt19937_64 rng(local_seed);
    uniform_real_distribution<double> alpha_dist(0.7, 1.0);

    auto start_time = chrono::steady_clock::now();
    auto deadline = chrono::time_point_cast<chrono::steady_clock::duration>(
        start_time + chrono::duration<double>(TIME_LIMIT_SECONDS)
    );

    int iterations = 0;
    int best_k = g.n + 1;
    vector<int> best_coloring(g.n, -1);

    {
        auto [k, colors] = dsatur_coloring(g);
        ++iterations;
        best_k = k;
        best_coloring = colors;

        cout << "[" << test_name << "] new best: " << best_k
             << " (iter=" << iterations << ", init=dsatur)" << endl;
    }

    while (chrono::steady_clock::now() < deadline) {
        vector<int> order = random_order(g.n, rng);
        double alpha = alpha_dist(rng);

        auto [k, colors] = semi_randomized_dsatur(g, order, alpha, rng);
        ++iterations;

        if (k < best_k) {
            best_k = k;
            best_coloring = std::move(colors);

            cout << "[" << test_name << "] new best: " << best_k
                 << " (iter=" << iterations
                 << ", alpha=" << alpha << ")" << endl;
        }
    }

    filesystem::create_directories(SOLUTIONS_DIR);
    string out_path = SOLUTIONS_DIR + "/" + test_name + ".out";
    ofstream fout(out_path);
    fout << best_k << "\n";
    for (int i = 0; i < g.n; ++i) {
        if (i) fout << ' ';
        fout << best_coloring[i];
    }
    fout << "\n";

    double elapsed = chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
    return {best_k, elapsed, iterations};
}

int main() {
    try {
        if (!filesystem::exists(TESTS_DIR) || !filesystem::is_directory(TESTS_DIR)) {
            cout << "Folder '" << TESTS_DIR << "' not found." << endl;
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
            cout << "No test files found in '" << TESTS_DIR << "'." << endl;
            return 0;
        }

        random_device rd;
        uint64_t global_seed = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());

        cout << "GLOBAL_RANDOM_SEED=" << global_seed << endl;
        cout << "TIME_LIMIT_SECONDS=" << TIME_LIMIT_SECONDS << endl;

        ofstream csv(RESULTS_CSV);
        csv << "instance,colors_used,runtime_sec,iterations,global_seed\n";

        for (const string& test_name : test_files) {
            string path = TESTS_DIR + "/" + test_name;
            cout << "Running " << test_name << " ..." << endl;

            SolveResult result = solve_one(test_name, path, global_seed);

            cout << "[" << test_name << "] final: " << result.best_k
                 << " colors, " << result.elapsed
                 << " sec, " << result.iterations << " iterations" << endl;

            csv << test_name << ","
                << result.best_k << ","
                << result.elapsed << ","
                << result.iterations << ","
                << global_seed << "\n";
        }

        cout << "Saved summary to " << RESULTS_CSV << endl;
    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}