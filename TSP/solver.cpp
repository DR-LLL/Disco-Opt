#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct Point {
    double x = 0.0;
    double y = 0.0;
};

static constexpr double INF = 1e100;
static constexpr double EPS = 1e-12;
static constexpr double TIME_LIMIT_SEC = 1800.0;

struct SolveResult {
    double best_cost = INF;
    double best_time_sec = -1.0;
    vector<int> best_tour;
    bool timed_out = false;
};

struct NodeEval {
    double lb = INF;
    double ub = INF;
    vector<int> greedy_tour;
};

struct ChildCandidate {
    int v = -1;
    double lb = INF;
    double ub = INF;
};

class TSPSolverParallel {
public:
    TSPSolverParallel(vector<Point> pts, string instance_name)
        : pts_(std::move(pts)),
          n_(static_cast<int>(pts_.size())),
          instance_name_(std::move(instance_name)) {
        dist_.assign(n_, vector<double>(n_, 0.0));
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                double dx = pts_[i].x - pts_[j].x;
                double dy = pts_[i].y - pts_[j].y;
                dist_[i][j] = std::sqrt(dx * dx + dy * dy);
            }
        }
    }

    SolveResult solve() {
        start_time_ = chrono::steady_clock::now();

        SolveResult ans;
        if (n_ == 0) {
            ans.best_cost = 0.0;
            ans.best_time_sec = 0.0;
            return ans;
        }
        if (n_ == 1) {
            ans.best_cost = 0.0;
            ans.best_time_sec = 0.0;
            ans.best_tour = {0};
            return ans;
        }

        global_best_cost_.store(INF);
        global_best_time_sec_.store(-1.0);
        timed_out_.store(false);

        {
            lock_guard<mutex> lock(best_mutex_);
            global_best_tour_.clear();
        }

        unsigned hw = thread::hardware_concurrency();
        int num_threads = max(1u, hw == 0 ? 4u : hw);

        // build top-level jobs: fixed start at 0, then choose second vertex
        vector<int> top_vertices;
        for (int v = 1; v < n_; ++v) top_vertices.push_back(v);

        // order top jobs by quick UB
        vector<pair<double, int>> scored_jobs;
        for (int v : top_vertices) {
            vector<int> prefix = {0, v};
            vector<char> used(n_, 0);
            used[0] = 1;
            used[v] = 1;
            double prefix_cost = dist_[0][v];
            NodeEval ev = evaluate_node(prefix, used, prefix_cost);
            if (ev.ub < INF / 2) maybe_update_best(ev.greedy_tour, ev.ub, "greedy-from-root");
            if (ev.lb < global_best_cost_.load() - EPS) {
                scored_jobs.push_back({ev.ub, v});
            }
        }

        sort(scored_jobs.begin(), scored_jobs.end(), [](const auto& a, const auto& b) {
            if (fabs(a.first - b.first) > EPS) return a.first < b.first;
            return a.second < b.second;
        });

        vector<int> jobs;
        jobs.reserve(scored_jobs.size());
        for (auto& p : scored_jobs) jobs.push_back(p.second);

        if (jobs.empty()) {
            SolveResult res;
            res.best_cost = global_best_cost_.load();
            res.best_time_sec = global_best_time_sec_.load();
            {
                lock_guard<mutex> lock(best_mutex_);
                res.best_tour = global_best_tour_;
            }
            if (res.best_cost >= INF / 2) {
                res.best_cost = -1.0;
                res.best_time_sec = -1.0;
                res.best_tour.clear();
            }
            return res;
        }

        int actual_threads = min<int>(num_threads, max<int>(1, jobs.size()));
        next_job_.store(0);
        thread_active_lb_.assign(actual_threads, INF);

        vector<thread> workers;
        workers.reserve(actual_threads);

        for (int tid = 0; tid < actual_threads; ++tid) {
            workers.emplace_back([this, tid, &jobs]() {
                std::mt19937 rng(1234567u + 10007u * static_cast<unsigned>(tid));
                worker_loop(tid, jobs, rng);
            });
        }

        for (auto& th : workers) th.join();

        ans.best_cost = global_best_cost_.load();
        ans.best_time_sec = global_best_time_sec_.load();
        {
            lock_guard<mutex> lock(best_mutex_);
            ans.best_tour = global_best_tour_;
        }
        ans.timed_out = timed_out_.load();

        if (ans.best_cost >= INF / 2) {
            ans.best_cost = -1.0;
            ans.best_time_sec = -1.0;
            ans.best_tour.clear();
        }

        return ans;
    }

private:
    vector<Point> pts_;
    int n_ = 0;
    string instance_name_;
    vector<vector<double>> dist_;
    chrono::steady_clock::time_point start_time_;

    atomic<double> global_best_cost_{INF};
    atomic<double> global_best_time_sec_{-1.0};
    vector<int> global_best_tour_;
    mutex best_mutex_;

    atomic<bool> timed_out_{false};
    atomic<int> next_job_{0};

    vector<double> thread_active_lb_;
    mutex lb_mutex_;

    double elapsed_sec() const {
        return chrono::duration<double>(chrono::steady_clock::now() - start_time_).count();
    }

    bool time_exceeded() const {
        return elapsed_sec() >= TIME_LIMIT_SEC;
    }

    vector<int> get_unvisited(const vector<char>& used) const {
        vector<int> u;
        u.reserve(n_);
        for (int i = 0; i < n_; ++i) if (!used[i]) u.push_back(i);
        return u;
    }

    double mst_cost(const vector<int>& verts) const {
        int m = static_cast<int>(verts.size());
        if (m <= 1) return 0.0;

        vector<double> min_edge(m, INF);
        vector<char> in_tree(m, 0);
        min_edge[0] = 0.0;
        double total = 0.0;

        for (int it = 0; it < m; ++it) {
            if (time_exceeded()) return INF;

            int v = -1;
            for (int i = 0; i < m; ++i) {
                if (!in_tree[i] && (v == -1 || min_edge[i] < min_edge[v])) v = i;
            }
            if (v == -1 || min_edge[v] >= INF / 2) return INF;

            in_tree[v] = 1;
            total += min_edge[v];

            for (int to = 0; to < m; ++to) {
                if (!in_tree[to]) {
                    double w = dist_[verts[v]][verts[to]];
                    if (w < min_edge[to]) min_edge[to] = w;
                }
            }
        }
        return total;
    }

    double lower_bound(const vector<int>& prefix, const vector<char>& used, double prefix_cost) const {
        vector<int> unvisited = get_unvisited(used);

        if (unvisited.empty()) {
            return prefix_cost + dist_[prefix.back()][prefix.front()];
        }

        double mst = mst_cost(unvisited);
        if (mst >= INF / 2) return INF;

        int left = prefix.front();
        int right = prefix.back();

        double best_left = INF;
        double best_right = INF;
        for (int u : unvisited) {
            best_left = min(best_left, dist_[left][u]);
            best_right = min(best_right, dist_[right][u]);
        }

        return prefix_cost + mst + best_left + best_right;
    }

    pair<double, vector<int>> greedy_complete(const vector<int>& prefix,
                                              const vector<char>& used,
                                              double prefix_cost) const {
        vector<int> tour = prefix;
        vector<char> mark = used;
        double cost = prefix_cost;
        int cur = tour.back();

        while ((int)tour.size() < n_) {
            if (time_exceeded()) return {INF, {}};

            int nxt = -1;
            double best = INF;
            for (int v = 0; v < n_; ++v) {
                if (!mark[v]) {
                    double w = dist_[cur][v];
                    if (w < best - EPS || (fabs(w - best) <= EPS && (nxt == -1 || v < nxt))) {
                        best = w;
                        nxt = v;
                    }
                }
            }
            if (nxt == -1) return {INF, {}};

            tour.push_back(nxt);
            mark[nxt] = 1;
            cost += dist_[cur][nxt];
            cur = nxt;
        }

        cost += dist_[tour.back()][tour.front()];
        return {cost, tour};
    }

    NodeEval evaluate_node(const vector<int>& prefix, const vector<char>& used, double prefix_cost) const {
        NodeEval ev;
        ev.lb = lower_bound(prefix, used, prefix_cost);
        auto [ub, tour] = greedy_complete(prefix, used, prefix_cost);
        ev.ub = ub;
        ev.greedy_tour = std::move(tour);
        return ev;
    }

    void maybe_update_best(const vector<int>& tour, double cost, const string& source) {
        if (tour.empty()) return;

        double cur = global_best_cost_.load();
        while (cost < cur - EPS) {
            if (global_best_cost_.compare_exchange_weak(cur, cost)) {
                double t = elapsed_sec();
                global_best_time_sec_.store(t);
                {
                    lock_guard<mutex> lock(best_mutex_);
                    global_best_tour_ = tour;
                }
                cerr << fixed << setprecision(6)
                     << "[BnB][" << instance_name_ << "] new best from " << source
                     << ": cost=" << cost
                     << " at " << t << " sec\n";
                cerr.flush();
                return;
            }
        }
    }

    void set_thread_lb(int tid, double value) {
        lock_guard<mutex> lock(lb_mutex_);
        thread_active_lb_[tid] = value;
    }

    double global_active_lb() {
        lock_guard<mutex> lock(lb_mutex_);
        double best = INF;
        for (double x : thread_active_lb_) best = min(best, x);
        return best;
    }

    void worker_loop(int tid, const vector<int>& jobs, mt19937& rng) {
        while (true) {
            if (time_exceeded()) {
                timed_out_.store(true);
                return;
            }

            int idx = next_job_.fetch_add(1);
            if (idx >= (int)jobs.size()) {
                set_thread_lb(tid, INF);
                return;
            }

            int v = jobs[idx];
            vector<int> prefix = {0, v};
            vector<char> used(n_, 0);
            used[0] = 1;
            used[v] = 1;
            double prefix_cost = dist_[0][v];

            dfs(tid, prefix, used, prefix_cost, rng);
        }
    }

    void dfs(int tid, vector<int>& prefix, vector<char>& used, double prefix_cost, mt19937& rng) {
        if (time_exceeded()) {
            timed_out_.store(true);
            return;
        }

        NodeEval cur = evaluate_node(prefix, used, prefix_cost);
        set_thread_lb(tid, cur.lb);

        double best_ub = global_best_cost_.load();

        if (cur.ub < best_ub - EPS) {
            maybe_update_best(cur.greedy_tour, cur.ub, "greedy-from-node");
            best_ub = global_best_cost_.load();
        }

        if (cur.lb >= best_ub - EPS) {
            return;
        }

        if ((int)prefix.size() == n_) {
            double full_cost = prefix_cost + dist_[prefix.back()][prefix.front()];
            if (full_cost < global_best_cost_.load() - EPS) {
                maybe_update_best(prefix, full_cost, "leaf");
            }
            return;
        }

        int last = prefix.back();
        vector<ChildCandidate> children;
        children.reserve(n_);

        for (int v = 0; v < n_; ++v) {
            if (used[v]) continue;
            if (time_exceeded()) {
                timed_out_.store(true);
                return;
            }

            used[v] = 1;
            prefix.push_back(v);
            double child_prefix_cost = prefix_cost + dist_[last][v];

            NodeEval ev = evaluate_node(prefix, used, child_prefix_cost);

            if (ev.ub < global_best_cost_.load() - EPS) {
                maybe_update_best(ev.greedy_tour, ev.ub, "greedy-from-node");
            }

            if (ev.lb < global_best_cost_.load() - EPS) {
                children.push_back({v, ev.lb, ev.ub});
            }

            prefix.pop_back();
            used[v] = 0;
        }

        // sort by UB, add randomness among close candidates
        sort(children.begin(), children.end(), [](const ChildCandidate& a, const ChildCandidate& b) {
            if (fabs(a.ub - b.ub) > EPS) return a.ub < b.ub;
            if (fabs(a.lb - b.lb) > EPS) return a.lb < b.lb;
            return a.v < b.v;
        });

        const int K = min<int>(4, children.size());
        if (K > 1) {
            shuffle(children.begin(), children.begin() + K, rng);
            stable_sort(children.begin(), children.end(), [](const ChildCandidate& a, const ChildCandidate& b) {
                if (fabs(a.ub - b.ub) > 1e-6) return a.ub < b.ub;
                if (fabs(a.lb - b.lb) > 1e-6) return a.lb < b.lb;
                return a.v < b.v;
            });
        }

        for (const ChildCandidate& ch : children) {
            if (time_exceeded()) {
                timed_out_.store(true);
                return;
            }

            if (ch.lb >= global_best_cost_.load() - EPS) continue;

            used[ch.v] = 1;
            prefix.push_back(ch.v);
            double next_cost = prefix_cost + dist_[last][ch.v];

            dfs(tid, prefix, used, next_cost, rng);

            prefix.pop_back();
            used[ch.v] = 0;
        }
    }
};

static bool read_instance_from_stream(istream& in, vector<Point>& pts) {
    int n;
    if (!(in >> n)) return false;
    pts.assign(n, {});
    for (int i = 0; i < n; ++i) {
        in >> pts[i].x >> pts[i].y;
    }
    return true;
}

static SolveResult solve_one_instance(const vector<Point>& pts, const string& name) {
    TSPSolverParallel solver(pts, name);
    return solver.solve();
}

static string tour_to_string(const vector<int>& tour) {
    ostringstream out;
    for (int i = 0; i < (int)tour.size(); ++i) {
        if (i) out << ' ';
        out << tour[i];
    }
    return out.str();
}

static bool is_tsp_name(const string& name) {
    return name.rfind("tsp_", 0) == 0;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc == 1) {
        vector<Point> pts;
        if (!read_instance_from_stream(cin, pts)) return 0;

        SolveResult res = solve_one_instance(pts, "stdin");
        if (res.best_cost < 0) {
            cout << "-1\n\n";
        } else {
            cout << fixed << setprecision(6) << res.best_cost << "\n";
            cout << tour_to_string(res.best_tour) << "\n";
        }
        return 0;
    }

    string dir = argv[1];
    vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        string name = entry.path().filename().string();
        if (is_tsp_name(name)) files.push_back(entry.path());
    }
    sort(files.begin(), files.end());

    ofstream csv("results.csv");
    csv << "test_name,objective_value,best_found_time_sec,solution\n";

    for (const auto& path : files) {
        string name = path.filename().string();
        ifstream fin(path);

        vector<Point> pts;
        if (!read_instance_from_stream(fin, pts)) {
            cerr << "[" << name << "] failed to read instance\n";
            continue;
        }

        SolveResult res = solve_one_instance(pts, name);

        csv << name << ",";
        if (res.best_cost < 0) {
            csv << "-1,-1,\"\"\n";
            cerr << "[" << name << "] done: no feasible solution";
            if (res.timed_out) cerr << " timed_out=1";
            cerr << "\n";
        } else {
            csv << fixed << setprecision(6) << res.best_cost << ","
                << fixed << setprecision(6) << res.best_time_sec << ",\""
                << tour_to_string(res.best_tour) << "\"\n";
            cerr << "[" << name << "] done: cost=" << fixed << setprecision(6) << res.best_cost
                 << " best_time=" << fixed << setprecision(6) << res.best_time_sec;
            if (res.timed_out) cerr << " timed_out=1";
            cerr << "\n";
        }
        cerr.flush();
    }

    return 0;
}