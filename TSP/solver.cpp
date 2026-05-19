#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct SolveResult {
    double best_cost = numeric_limits<double>::infinity();
    double best_time_sec = 0.0;
    vector<int> best_tour;
    bool timed_out = false;
};

static constexpr double INF = 1e100;
static constexpr double EPS = 1e-9;

static int DEFAULT_CANDIDATES = 32;
static constexpr int EXACT_CANDIDATE_N_LIMIT = 3500;

static double MATCHING_ANNEAL_TIME_LIMIT_SEC = 20.0;
static int MATCHING_RANDOM_FALLBACK_TRIALS = 2048;

static double WINDOW_PERM_TIME_LIMIT_SEC = 100.0;
static double WINDOW_PERM_MEAN_SIZE = 5.0;
static int WINDOW_PERM_MAX_SIZE = 10;

class TSPSolver {
public:
    TSPSolver(vector<Point> points, string instance_name, double time_limit_sec)
        : pts_(std::move(points)),
          n_(static_cast<int>(pts_.size())),
          name_(std::move(instance_name)),
          time_limit_sec_(time_limit_sec) {}

    SolveResult solve() {
        start_time_ = chrono::steady_clock::now();

        rng_.seed(
            1469598103934665603ULL
            ^ std::hash<std::string>{}(name_)
            ^ static_cast<uint64_t>(n_)
        );

        SolveResult res;

        if (n_ == 0) {
            res.best_cost = 0.0;
            res.best_time_sec = 0.0;
            return res;
        }

        if (n_ == 1) {
            res.best_cost = 0.0;
            res.best_time_sec = 0.0;
            res.best_tour = {0};
            return res;
        }

        build_candidate_lists();

        vector<int> tour = christofides_start_tour();

        if ((int)tour.size() != n_) {
            cerr << "[" << name_ << "] christofides-start failed, using greedy fallback\n";
            cerr.flush();

            tour = greedy_nearest_neighbor_tour();
        }

        normalize_tour(tour);

        double cur_cost = tour_cost(tour);

        update_best(tour, cur_cost, "christofides-start");

        local_search(tour, cur_cost);
        update_best(tour, cur_cost, "initial-2-opt");

        int perturb_id = 0;

        while (!time_exceeded()) {
            if (WINDOW_PERM_TIME_LIMIT_SEC > 0.0 &&
                remaining_sec() <= WINDOW_PERM_TIME_LIMIT_SEC) {
                break;
            }

            ++perturb_id;

            if (!best_tour_.empty() && (perturb_id % 5 == 0 || cur_cost > best_cost_ * 1.05)) {
                tour = best_tour_;
                cur_cost = best_cost_;
            }

            if (!perturb_tour(tour, cur_cost, perturb_id)) {
                break;
            }

            local_search(tour, cur_cost);

            update_best(tour, cur_cost, "post-perturb-2-opt");
        }

        window_permutation_search(tour, cur_cost);

        res.best_cost = best_cost_;
        res.best_time_sec = best_time_sec_;
        res.best_tour = best_tour_;
        res.timed_out = time_exceeded();

        cerr << fixed << setprecision(6)
             << "[" << name_ << "] solver stopped: "
             << "best_cost=" << best_cost_
             << ", best_time=" << best_time_sec_
             << ", total_time=" << elapsed_sec()
             << " sec\n";

        cerr.flush();

        return res;
    }

private:
    vector<Point> pts_;
    int n_ = 0;
    string name_;
    double time_limit_sec_ = 30.0;
    chrono::steady_clock::time_point start_time_;

    vector<vector<int>> cand_;
    double best_cost_ = INF;
    double best_time_sec_ = 0.0;
    vector<int> best_tour_;
    mt19937_64 rng_{123456789ULL};

    double elapsed_sec() const {
        return chrono::duration<double>(chrono::steady_clock::now() - start_time_).count();
    }

    double remaining_sec() const {
        return max(0.0, time_limit_sec_ - elapsed_sec());
    }

    bool time_exceeded() const {
        return elapsed_sec() >= time_limit_sec_;
    }

    static double seconds_since(const chrono::steady_clock::time_point& t0) {
        return chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    }

    double dist2(int i, int j) const {
        const double dx = pts_[i].x - pts_[j].x;
        const double dy = pts_[i].y - pts_[j].y;
        return dx * dx + dy * dy;
    }

    double dist(int i, int j) const {
        return std::sqrt(dist2(i, j));
    }

    double tour_cost(const vector<int>& tour) const {
        double s = 0.0;

        for (int i = 0; i < n_; ++i) {
            s += dist(tour[i], tour[(i + 1) % n_]);
        }

        return s;
    }

    void update_best(const vector<int>& tour, double cost, const string& source) {
        if ((int)tour.size() != n_) {
            return;
        }

        if (cost + EPS < best_cost_) {
            best_cost_ = cost;
            best_time_sec_ = elapsed_sec();
            best_tour_ = tour;

            cerr << fixed << setprecision(6)
                 << "[" << name_ << "] new best from " << source
                 << ": cost=" << best_cost_
                 << " at " << best_time_sec_ << " sec\n";

            cerr.flush();
        }
    }

    void normalize_tour(vector<int>& tour) const {
        auto it = find(tour.begin(), tour.end(), 0);

        if (it != tour.end()) {
            rotate(tour.begin(), it, tour.end());
        }

        if (tour.size() >= 3 && tour[1] > tour.back()) {
            reverse(tour.begin() + 1, tour.end());
        }
    }

    static long long key_pair(int a, int b) {
        return (static_cast<long long>(a) << 32) ^ static_cast<unsigned int>(b);
    }

    void add_candidate(vector<pair<double, int>>& best, int v, double d2, int K) const {
        if ((int)best.size() < K) {
            best.push_back({d2, v});

            if ((int)best.size() == K) {
                sort(best.begin(), best.end());
            }

            return;
        }

        if (d2 < best.back().first) {
            best.back() = {d2, v};

            int p = K - 1;

            while (p > 0 && best[p] < best[p - 1]) {
                swap(best[p], best[p - 1]);
                --p;
            }
        }
    }

    void build_candidate_lists() {
        const int K = min(DEFAULT_CANDIDATES, max(1, n_ - 1));

        cand_.assign(n_, {});

        if (n_ <= 1) {
            return;
        }

        if (n_ <= EXACT_CANDIDATE_N_LIMIT) {
            vector<vector<pair<double, int>>> best(n_);

            for (int i = 0; i < n_; ++i) {
                best[i].reserve(K);
            }

            for (int i = 0; i < n_; ++i) {
                if (time_exceeded()) {
                    break;
                }

                for (int j = i + 1; j < n_; ++j) {
                    double d = dist2(i, j);
                    add_candidate(best[i], j, d, K);
                    add_candidate(best[j], i, d, K);
                }
            }

            for (int i = 0; i < n_; ++i) {
                sort(best[i].begin(), best[i].end());

                for (auto [_, v] : best[i]) {
                    cand_[i].push_back(v);
                }
            }

            return;
        }

        double minx = pts_[0].x;
        double maxx = pts_[0].x;
        double miny = pts_[0].y;
        double maxy = pts_[0].y;

        for (const auto& p : pts_) {
            minx = min(minx, p.x);
            maxx = max(maxx, p.x);
            miny = min(miny, p.y);
            maxy = max(maxy, p.y);
        }

        double area = max(1.0, (maxx - minx) * (maxy - miny));
        double cell = sqrt(area / max(1, n_)) * 2.0;

        if (cell <= 0.0 || !isfinite(cell)) {
            cell = 1.0;
        }

        unordered_map<long long, vector<int>> grid;
        grid.reserve(n_ * 2);

        vector<pair<int, int>> cell_of(n_);

        for (int i = 0; i < n_; ++i) {
            int cx = static_cast<int>(floor((pts_[i].x - minx) / cell));
            int cy = static_cast<int>(floor((pts_[i].y - miny) / cell));

            cell_of[i] = {cx, cy};
            grid[key_pair(cx, cy)].push_back(i);
        }

        for (int i = 0; i < n_; ++i) {
            if (time_exceeded()) {
                break;
            }

            vector<pair<double, int>> best;
            best.reserve(K);

            auto [cx, cy] = cell_of[i];

            int radius = 0;

            while ((int)best.size() < K && radius <= 32) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    for (int dy = -radius; dy <= radius; ++dy) {
                        if (max(abs(dx), abs(dy)) != radius) {
                            continue;
                        }

                        auto it = grid.find(key_pair(cx + dx, cy + dy));

                        if (it == grid.end()) {
                            continue;
                        }

                        for (int v : it->second) {
                            if (v == i) {
                                continue;
                            }

                            add_candidate(best, v, dist2(i, v), K);
                        }
                    }
                }

                ++radius;
            }

            if ((int)best.size() < K) {
                for (int v = 0; v < n_; ++v) {
                    if (v != i) {
                        add_candidate(best, v, dist2(i, v), K);
                    }

                    if ((int)best.size() >= K && v % 2048 == 0 && time_exceeded()) {
                        break;
                    }
                }
            }

            sort(best.begin(), best.end());

            for (auto [_, v] : best) {
                cand_[i].push_back(v);
            }
        }
    }

    vector<pair<int, int>> build_mst_edges() {
        vector<double> min_d2(n_, INF);
        vector<int> parent(n_, -1);
        vector<char> used(n_, 0);

        min_d2[0] = 0.0;

        vector<pair<int, int>> edges;
        edges.reserve(max(0, n_ - 1));

        for (int it = 0; it < n_; ++it) {
            if (time_exceeded()) {
                break;
            }

            int v = -1;

            for (int i = 0; i < n_; ++i) {
                if (!used[i] && (v == -1 || min_d2[i] < min_d2[v])) {
                    v = i;
                }
            }

            if (v == -1) {
                break;
            }

            used[v] = 1;

            if (parent[v] != -1) {
                edges.push_back({parent[v], v});
            }

            for (int to = 0; to < n_; ++to) {
                if (used[to]) {
                    continue;
                }

                double d = dist2(v, to);

                if (d < min_d2[to]) {
                    min_d2[to] = d;
                    parent[to] = v;
                }
            }
        }

        if ((int)edges.size() != n_ - 1) {
            edges.clear();

            fill(min_d2.begin(), min_d2.end(), INF);
            fill(parent.begin(), parent.end(), -1);
            fill(used.begin(), used.end(), 0);

            min_d2[0] = 0.0;

            for (int it = 0; it < n_; ++it) {
                int v = -1;

                for (int i = 0; i < n_; ++i) {
                    if (!used[i] && (v == -1 || min_d2[i] < min_d2[v])) {
                        v = i;
                    }
                }

                if (v == -1 || min_d2[v] >= INF / 2) {
                    for (int u = 0; u < n_; ++u) {
                        if (!used[u]) {
                            v = u;
                            break;
                        }
                    }

                    double bd = INF;
                    int bp = -1;

                    for (int u = 0; u < n_; ++u) {
                        if (used[u]) {
                            double d = dist2(v, u);

                            if (d < bd) {
                                bd = d;
                                bp = u;
                            }
                        }
                    }

                    parent[v] = bp;
                }

                used[v] = 1;

                if (parent[v] != -1) {
                    edges.push_back({parent[v], v});
                }

                for (int to : cand_[v]) {
                    if (!used[to]) {
                        double d = dist2(v, to);

                        if (d < min_d2[to]) {
                            min_d2[to] = d;
                            parent[to] = v;
                        }
                    }
                }
            }
        }

        return edges;
    }

    vector<pair<int, int>> greedy_initial_matching_for_anneal(const vector<int>& odd) {
        int m = static_cast<int>(odd.size());

        vector<int> odd_index_of_vertex(n_, -1);

        for (int i = 0; i < m; ++i) {
            odd_index_of_vertex[odd[i]] = i;
        }

        vector<int> remaining(m);
        iota(remaining.begin(), remaining.end(), 0);

        shuffle(remaining.begin(), remaining.end(), rng_);

        vector<int> pos(m, -1);

        for (int i = 0; i < m; ++i) {
            pos[remaining[i]] = i;
        }

        auto remove_idx = [&](int x) {
            int p = pos[x];

            if (p < 0) {
                return;
            }

            int last = remaining.back();

            remaining[p] = last;
            pos[last] = p;

            remaining.pop_back();
            pos[x] = -1;
        };

        vector<pair<int, int>> pairs;
        pairs.reserve(m / 2);

        while (!remaining.empty()) {
            int a = remaining.back();
            remove_idx(a);

            if (remaining.empty()) {
                break;
            }

            int best = -1;
            double bd = INF;
            int va = odd[a];

            for (int vertex_candidate : cand_[va]) {
                int b = odd_index_of_vertex[vertex_candidate];

                if (b >= 0 && pos[b] >= 0) {
                    double d = dist2(va, odd[b]);

                    if (d < bd) {
                        bd = d;
                        best = b;
                    }
                }
            }

            if (best == -1) {
                if ((int)remaining.size() <= MATCHING_RANDOM_FALLBACK_TRIALS) {
                    for (int b : remaining) {
                        double d = dist2(va, odd[b]);

                        if (d < bd) {
                            bd = d;
                            best = b;
                        }
                    }
                } else {
                    uniform_int_distribution<int> idx_dist(0, (int)remaining.size() - 1);

                    for (int t = 0; t < MATCHING_RANDOM_FALLBACK_TRIALS; ++t) {
                        int b = remaining[idx_dist(rng_)];
                        double d = dist2(va, odd[b]);

                        if (d < bd) {
                            bd = d;
                            best = b;
                        }
                    }
                }
            }

            if (best == -1) {
                best = remaining.back();
            }

            remove_idx(best);

            pairs.push_back({odd[a], odd[best]});
        }

        return pairs;
    }

    vector<pair<int, int>> annealed_matching(const vector<int>& odd) {
        int m = static_cast<int>(odd.size());

        if (m == 0) {
            return {};
        }

        if (m == 2) {
            return {{odd[0], odd[1]}};
        }

        auto phase_start = chrono::steady_clock::now();

        cerr << fixed << setprecision(6)
             << "[" << name_ << "] matching annealing started: "
             << "odd_count=" << m
             << ", time_limit=" << MATCHING_ANNEAL_TIME_LIMIT_SEC
             << " sec\n";

        cerr.flush();

        vector<pair<int, int>> pairs = greedy_initial_matching_for_anneal(odd);

        auto matching_cost = [&](const vector<pair<int, int>>& ps) {
            double s = 0.0;

            for (auto [u, v] : ps) {
                s += dist(u, v);
            }

            return s;
        };

        double cur_cost = matching_cost(pairs);
        double best_cost = cur_cost;
        vector<pair<int, int>> best_pairs = pairs;

        int pcount = static_cast<int>(pairs.size());

        double avg_edge = cur_cost / max(1, pcount);
        double temperature = max(1e-9, avg_edge * 0.05);
        double temperature_min = max(1e-12, avg_edge * 1e-6);

        long long iterations = 0;
        long long accepted = 0;
        long long improved = 0;

        uniform_real_distribution<double> real_dist(0.0, 1.0);

        while (!time_exceeded() && seconds_since(phase_start) < MATCHING_ANNEAL_TIME_LIMIT_SEC) {
            if (pcount < 2) {
                break;
            }

            uniform_int_distribution<int> pair_dist(0, pcount - 1);

            int p1 = pair_dist(rng_);
            int p2 = pair_dist(rng_);

            if (p1 == p2) {
                continue;
            }

            auto [a, b] = pairs[p1];
            auto [c, d] = pairs[p2];

            double old_cost = dist(a, b) + dist(c, d);

            double new1 = dist(a, c) + dist(b, d);
            double new2 = dist(a, d) + dist(b, c);

            int mode = 1;
            double new_cost = new1;

            if (new2 < new1) {
                mode = 2;
                new_cost = new2;
            }

            double delta = new_cost - old_cost;

            bool accept = false;

            if (delta < -EPS) {
                accept = true;
            } else if (temperature > temperature_min) {
                double prob = exp(-delta / temperature);
                accept = real_dist(rng_) < prob;
            }

            if (accept) {
                if (mode == 1) {
                    pairs[p1] = {a, c};
                    pairs[p2] = {b, d};
                } else {
                    pairs[p1] = {a, d};
                    pairs[p2] = {b, c};
                }

                cur_cost += delta;
                ++accepted;

                if (cur_cost + EPS < best_cost) {
                    best_cost = cur_cost;
                    best_pairs = pairs;
                    ++improved;
                }
            }

            ++iterations;

            if ((iterations & 4095LL) == 0) {
                temperature *= 0.995;

                if (temperature < temperature_min) {
                    temperature = temperature_min;
                }
            }
        }

        cerr << fixed << setprecision(6)
             << "[" << name_ << "] matching annealing stopped: "
             << "iterations=" << iterations
             << ", accepted=" << accepted
             << ", improved=" << improved
             << ", best_matching_cost=" << best_cost
             << ", phase_time=" << seconds_since(phase_start)
             << " sec\n";

        cerr.flush();

        return best_pairs;
    }

    vector<int> euler_tour(vector<vector<int>>& g) {
        vector<int> st;
        vector<int> euler;

        st.push_back(0);

        while (!st.empty()) {
            int v = st.back();

            if (g[v].empty()) {
                euler.push_back(v);
                st.pop_back();
            } else {
                int to = g[v].back();

                g[v].pop_back();

                auto it = find(g[to].begin(), g[to].end(), v);

                if (it != g[to].end()) {
                    g[to].erase(it);
                }

                st.push_back(to);
            }
        }

        return euler;
    }

    vector<int> christofides_start_tour() {
        auto mst = build_mst_edges();

        if ((int)mst.size() != n_ - 1) {
            return {};
        }

        vector<int> deg(n_, 0);
        vector<vector<int>> multigraph(n_);

        for (auto [u, v] : mst) {
            ++deg[u];
            ++deg[v];

            multigraph[u].push_back(v);
            multigraph[v].push_back(u);
        }

        vector<int> odd;

        for (int i = 0; i < n_; ++i) {
            if (deg[i] & 1) {
                odd.push_back(i);
            }
        }

        vector<pair<int, int>> matching = annealed_matching(odd);

        for (auto [u, v] : matching) {
            multigraph[u].push_back(v);
            multigraph[v].push_back(u);
        }

        vector<int> euler = euler_tour(multigraph);

        vector<char> seen(n_, 0);
        vector<int> tour;

        tour.reserve(n_);

        for (int v : euler) {
            if (!seen[v]) {
                seen[v] = 1;
                tour.push_back(v);
            }
        }

        if ((int)tour.size() != n_) {
            return {};
        }

        return tour;
    }

    vector<int> greedy_nearest_neighbor_tour() {
        vector<int> tour;
        vector<char> used(n_, 0);

        tour.reserve(n_);

        int cur = 0;

        tour.push_back(cur);
        used[cur] = 1;

        for (int step = 1; step < n_; ++step) {
            int best = -1;
            double bd = INF;

            for (int v : cand_[cur]) {
                if (!used[v]) {
                    double d = dist2(cur, v);

                    if (d < bd) {
                        bd = d;
                        best = v;
                    }
                }
            }

            if (best == -1) {
                for (int v = 0; v < n_; ++v) {
                    if (!used[v]) {
                        double d = dist2(cur, v);

                        if (d < bd) {
                            bd = d;
                            best = v;
                        }
                    }
                }
            }

            cur = best;
            used[cur] = 1;
            tour.push_back(cur);
        }

        return tour;
    }

    vector<int> build_pos(const vector<int>& tour) const {
        vector<int> pos(n_);

        for (int i = 0; i < n_; ++i) {
            pos[tour[i]] = i;
        }

        return pos;
    }

    bool apply_2opt(vector<int>& tour, vector<int>& pos, int i, int j, double& cost) {
        if (i > j) {
            swap(i, j);
        }

        if (i == j) {
            return false;
        }

        if (j == i + 1) {
            return false;
        }

        if (i == 0 && j == n_ - 1) {
            return false;
        }

        int a = tour[i];
        int b = tour[(i + 1) % n_];
        int c = tour[j];
        int d = tour[(j + 1) % n_];

        double delta = dist(a, c) + dist(b, d) - dist(a, b) - dist(c, d);

        if (delta < -EPS) {
            reverse(tour.begin() + i + 1, tour.begin() + j + 1);

            for (int t = i + 1; t <= j; ++t) {
                pos[tour[t]] = t;
            }

            cost += delta;

            update_best(tour, cost, "2-opt");

            return true;
        }

        return false;
    }

    bool two_opt_pass(vector<int>& tour, vector<int>& pos, double& cost) {
        for (int i = 0; i < n_ && !time_exceeded(); ++i) {
            int a = tour[i];

            for (int c : cand_[a]) {
                int j = pos[c];

                if (apply_2opt(tour, pos, i, j, cost)) {
                    return true;
                }
            }
        }

        return false;
    }

    void local_search(vector<int>& tour, double& cost) {
        vector<int> pos = build_pos(tour);

        while (!time_exceeded()) {
            bool improved = two_opt_pass(tour, pos, cost);

            if (!improved) {
                break;
            }
        }
    }

    bool double_bridge_perturb(vector<int>& tour, double& cost) {
        if (n_ < 8) {
            return false;
        }

        uniform_int_distribution<int> dist_index(1, n_ - 2);

        int a = 0;
        int b = 0;
        int c = 0;
        int d = 0;

        for (int attempt = 0; attempt < 100; ++attempt) {
            vector<int> cuts;
            cuts.reserve(4);

            while ((int)cuts.size() < 4) {
                int x = dist_index(rng_);

                bool ok = true;

                for (int y : cuts) {
                    if (abs(x - y) <= 1) {
                        ok = false;
                        break;
                    }
                }

                if (ok) {
                    cuts.push_back(x);
                }
            }

            sort(cuts.begin(), cuts.end());

            a = cuts[0];
            b = cuts[1];
            c = cuts[2];
            d = cuts[3];

            if (a > 0 && a < b && b < c && c < d && d < n_) {
                break;
            }
        }

        if (!(a > 0 && a < b && b < c && c < d && d < n_)) {
            return false;
        }

        vector<int> ntour;

        ntour.reserve(n_);

        ntour.insert(ntour.end(), tour.begin(), tour.begin() + a);
        ntour.insert(ntour.end(), tour.begin() + c, tour.begin() + d);
        ntour.insert(ntour.end(), tour.begin() + b, tour.begin() + c);
        ntour.insert(ntour.end(), tour.begin() + a, tour.begin() + b);
        ntour.insert(ntour.end(), tour.begin() + d, tour.end());

        tour.swap(ntour);

        normalize_tour(tour);

        cost = tour_cost(tour);

        return true;
    }

    bool random_segment_perturb(vector<int>& tour, double& cost) {
        if (n_ < 5) {
            return false;
        }

        uniform_int_distribution<int> dist_index(1, n_ - 1);

        int a = dist_index(rng_);
        int b = dist_index(rng_);

        if (a == b) {
            return false;
        }

        if (a > b) {
            swap(a, b);
        }

        if (b - a < 2) {
            return false;
        }

        reverse(tour.begin() + a, tour.begin() + b);

        normalize_tour(tour);

        cost = tour_cost(tour);

        return true;
    }

    bool perturb_tour(vector<int>& tour, double& cost, int perturb_id) {
        if (time_exceeded()) {
            return false;
        }

        bool ok = false;

        if (perturb_id % 4 == 0) {
            ok = random_segment_perturb(tour, cost);
        }

        if (!ok) {
            ok = double_bridge_perturb(tour, cost);
        }

        if (!ok) {
            ok = random_segment_perturb(tour, cost);
        }

        return ok;
    }

    int sample_window_perm_size() {
        double mean = WINDOW_PERM_MEAN_SIZE;

        if (!isfinite(mean) || mean < 1.0) {
            mean = 1.0;
        }

        double p = 1.0 / mean;

        if (p < 1e-9) {
            p = 1e-9;
        }

        if (p > 1.0) {
            p = 1.0;
        }

        geometric_distribution<int> geom(p);

        int k = 1 + geom(rng_);

        k = max(3, k);
        k = min(WINDOW_PERM_MAX_SIZE, k);
        k = min(k, max(2, n_ - 2));

        return k;
    }

    double window_order_cost(
        int left,
        const vector<int>& w,
        const vector<int>& order,
        int right
    ) const {
        double s = dist(left, w[order[0]]);

        for (int i = 0; i + 1 < (int)order.size(); ++i) {
            s += dist(w[order[i]], w[order[i + 1]]);
        }

        s += dist(w[order.back()], right);

        return s;
    }

    bool try_window_permutation_once(
        vector<int>& tour,
        vector<int>& pos,
        double& cost,
        long long iteration
    ) {
        if ((int)tour.size() != n_ || n_ < 5) {
            return false;
        }

        int k = sample_window_perm_size();

        if (k < 2 || k > n_ - 2) {
            return false;
        }

        uniform_int_distribution<int> start_dist(0, n_ - 1);

        int start_pos = start_dist(rng_);

        int left = tour[(start_pos - 1 + n_) % n_];
        int right = tour[(start_pos + k) % n_];

        vector<int> w;
        w.reserve(k);

        for (int t = 0; t < k; ++t) {
            w.push_back(tour[(start_pos + t) % n_]);
        }

        vector<int> base_order(k);
        iota(base_order.begin(), base_order.end(), 0);

        double old_local_cost = window_order_cost(left, w, base_order, right);

        vector<int> order = base_order;
        vector<int> best_order = base_order;

        double best_local_cost = old_local_cost;

        long long checked = 0;

        do {
            double candidate_cost = window_order_cost(left, w, order, right);

            if (candidate_cost + EPS < best_local_cost) {
                best_local_cost = candidate_cost;
                best_order = order;
            }

            ++checked;

            if ((checked & 8191LL) == 0 && time_exceeded()) {
                break;
            }
        } while (next_permutation(order.begin(), order.end()));

        double delta = best_local_cost - old_local_cost;

        if (delta >= -EPS) {
            return false;
        }

        for (int t = 0; t < k; ++t) {
            int p = (start_pos + t) % n_;
            int new_vertex = w[best_order[t]];

            tour[p] = new_vertex;
            pos[new_vertex] = p;
        }

        cost += delta;

        update_best(
            tour,
            cost,
            "window-permutation-k=" + to_string(k) +
            "-iter=" + to_string(iteration)
        );

        return true;
    }

    void window_permutation_search(vector<int>& tour, double& cost) {
        if (WINDOW_PERM_TIME_LIMIT_SEC <= 0.0 || n_ < 5 || time_exceeded()) {
            return;
        }

        if (!best_tour_.empty() && best_cost_ + EPS < cost) {
            tour = best_tour_;
            cost = best_cost_;
        }

        vector<int> pos = build_pos(tour);

        auto phase_start = chrono::steady_clock::now();

        cerr << fixed << setprecision(6)
             << "[" << name_ << "] window permutation phase started: "
             << "time_limit=" << WINDOW_PERM_TIME_LIMIT_SEC
             << " sec, mean_size=" << WINDOW_PERM_MEAN_SIZE
             << ", max_size=" << WINDOW_PERM_MAX_SIZE
             << "\n";

        cerr.flush();

        long long iterations = 0;
        long long accepted = 0;

        while (!time_exceeded() && seconds_since(phase_start) < WINDOW_PERM_TIME_LIMIT_SEC) {
            ++iterations;

            bool improved = try_window_permutation_once(
                tour,
                pos,
                cost,
                iterations
            );

            if (improved) {
                ++accepted;
            }
        }

        string stop_reason;

        if (time_exceeded()) {
            stop_reason = "global time limit reached";
        } else {
            stop_reason = "window permutation time limit reached";
        }

        cerr << fixed << setprecision(6)
             << "[" << name_ << "] window permutation phase stopped: "
             << "reason=\"" << stop_reason << "\""
             << ", iterations=" << iterations
             << ", accepted=" << accepted
             << ", best_cost=" << best_cost_
             << ", total_time=" << elapsed_sec()
             << " sec, phase_time=" << seconds_since(phase_start)
             << " sec\n";

        cerr.flush();
    }
};

static double get_time_limit() {
    const char* env = std::getenv("TSP_TIME_LIMIT");
    double d = 400.0;

    if (!env) {
        return d;
    }

    char* end = nullptr;

    double v = std::strtod(env, &end);

    if (end == env || !isfinite(v) || v <= 0.0) {
        return d;
    }

    return v;
}

static void read_int_env(const char* name, int& target, int min_value) {
    const char* env = getenv(name);

    if (!env) {
        return;
    }

    char* end = nullptr;
    long v = strtol(env, &end, 10);

    if (end != env && v >= min_value) {
        target = static_cast<int>(v);
    }
}

static void read_double_env(const char* name, double& target, double min_value, bool strict_min) {
    const char* env = getenv(name);

    if (!env) {
        return;
    }

    char* end = nullptr;
    double v = strtod(env, &end);

    if (end == env || !isfinite(v)) {
        return;
    }

    if (strict_min) {
        if (v > min_value) {
            target = v;
        }
    } else {
        if (v >= min_value) {
            target = v;
        }
    }
}

static void read_settings() {
    read_int_env("TSP_CANDIDATES", DEFAULT_CANDIDATES, 1);

    read_double_env("TSP_MATCHING_ANNEAL_TIME_LIMIT", MATCHING_ANNEAL_TIME_LIMIT_SEC, 0.0, false);
    read_int_env("TSP_MATCHING_RANDOM_FALLBACK_TRIALS", MATCHING_RANDOM_FALLBACK_TRIALS, 1);

    read_double_env("TSP_WINDOW_PERM_TIME_LIMIT", WINDOW_PERM_TIME_LIMIT_SEC, 0.0, false);
    read_double_env("TSP_WINDOW_PERM_MEAN_SIZE", WINDOW_PERM_MEAN_SIZE, 1.0, false);
    read_int_env("TSP_WINDOW_PERM_MAX_SIZE", WINDOW_PERM_MAX_SIZE, 3);

    WINDOW_PERM_MAX_SIZE = max(3, WINDOW_PERM_MAX_SIZE);
}

static bool read_instance_from_stream(istream& in, vector<Point>& pts) {
    int n;

    if (!(in >> n)) {
        return false;
    }

    pts.assign(n, {});

    for (int i = 0; i < n; ++i) {
        in >> pts[i].x >> pts[i].y;
    }

    return true;
}

static string tour_to_string(const vector<int>& tour) {
    ostringstream out;

    for (int i = 0; i < (int)tour.size(); ++i) {
        if (i) {
            out << ' ';
        }

        out << tour[i];
    }

    return out.str();
}

static bool is_tsp_name(const string& name) {
    return name.rfind("tsp_", 0) == 0;
}

static SolveResult solve_one_instance(const vector<Point>& pts, const string& name, double time_limit) {
    TSPSolver solver(pts, name, time_limit);

    return solver.solve();
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    read_settings();

    double time_limit = get_time_limit();

    if (argc >= 3) {
        char* end = nullptr;

        double v = std::strtod(argv[2], &end);

        if (end != argv[2] && isfinite(v) && v > 0.0) {
            time_limit = v;
        }
    }

    if (argc == 1) {
        vector<Point> pts;

        if (!read_instance_from_stream(cin, pts)) {
            return 0;
        }

        SolveResult res = solve_one_instance(pts, "stdin", time_limit);

        cout << fixed << setprecision(6) << res.best_cost << "\n";
        cout << tour_to_string(res.best_tour) << "\n";

        return 0;
    }

    string dir = argv[1];

    vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        string name = entry.path().filename().string();

        if (is_tsp_name(name)) {
            files.push_back(entry.path());
        }
    }

    sort(files.begin(), files.end());

    ofstream csv("results.csv");

    csv << "test_name,objective_value,best_found_time_sec,solution\n";

    for (const auto& path : files) {
        string name = path.filename().string();

        ifstream fin(path);

        vector<Point> pts;

        if (!read_instance_from_stream(fin, pts)) {
            csv << name << ",ERROR,-1,\"\"\n";
            continue;
        }

        SolveResult res = solve_one_instance(pts, name, time_limit);

        csv << name << ",";

        if (!isfinite(res.best_cost) || res.best_tour.empty()) {
            csv << "ERROR,-1,\"\"\n";
        } else {
            csv << fixed << setprecision(6) << res.best_cost << ","
                << fixed << setprecision(6) << res.best_time_sec << ",\""
                << tour_to_string(res.best_tour) << "\"\n";
        }

        csv.flush();
    }

    return 0;
}