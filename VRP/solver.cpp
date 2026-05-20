#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

struct Customer {
    int demand;
    double x, y;
};

static const double EPS = 1e-9;

int N, V, CAP;
vector<Customer> a;
vector<vector<double>> distm;

mt19937 rng(1234567);

chrono::steady_clock::time_point start_time;
double TIME_LIMIT = 590.0;

struct Solution {
    vector<vector<int>> r;
    vector<int> load;
    double cost = 1e100;
};

double elapsed() {
    return chrono::duration<double>(chrono::steady_clock::now() - start_time).count();
}

bool time_left() {
    return elapsed() < TIME_LIMIT;
}

void build_dist() {
    distm.assign(N, vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = a[i].x - a[j].x;
            double dy = a[i].y - a[j].y;
            double d = sqrt(dx * dx + dy * dy);
            distm[i][j] = distm[j][i] = d;
        }
    }
}

int route_load(const vector<int>& r) {
    int s = 0;
    for (int x : r) {
        s += a[x].demand;
    }
    return s;
}

double route_cost(const vector<int>& r) {
    if (r.empty()) {
        return 0.0;
    }

    double c = distm[0][r[0]];
    for (int i = 0; i + 1 < (int)r.size(); ++i) {
        c += distm[r[i]][r[i + 1]];
    }
    c += distm[r.back()][0];

    return c;
}

double total_cost(const vector<vector<int>>& routes) {
    double s = 0.0;
    for (const auto& r : routes) {
        s += route_cost(r);
    }
    return s;
}

void recompute(Solution& sol) {
    sol.load.assign(sol.r.size(), 0);
    for (int i = 0; i < (int)sol.r.size(); ++i) {
        sol.load[i] = route_load(sol.r[i]);
    }
    sol.cost = total_cost(sol.r);
}

double insert_delta(const vector<int>& r, int pos, int node) {
    int prev = (pos == 0 ? 0 : r[pos - 1]);
    int next = (pos == (int)r.size() ? 0 : r[pos]);
    return distm[prev][node] + distm[node][next] - distm[prev][next];
}

double remove_delta(const vector<int>& r, int pos) {
    int prev = (pos == 0 ? 0 : r[pos - 1]);
    int node = r[pos];
    int next = (pos + 1 == (int)r.size() ? 0 : r[pos + 1]);
    return distm[prev][next] - distm[prev][node] - distm[node][next];
}

double replace_delta(const vector<int>& r, int pos, int new_node) {
    int prev = (pos == 0 ? 0 : r[pos - 1]);
    int old = r[pos];
    int next = (pos + 1 == (int)r.size() ? 0 : r[pos + 1]);

    return distm[prev][new_node] + distm[new_node][next]
         - distm[prev][old] - distm[old][next];
}

bool regret_insert_all(
    vector<vector<int>>& routes,
    vector<int> nodes,
    int max_routes,
    bool allow_empty_new
) {
    vector<int> loads(routes.size());
    for (int i = 0; i < (int)routes.size(); ++i) {
        loads[i] = route_load(routes[i]);
    }

    while (!nodes.empty()) {
        int choose_idx = -1;
        int choose_route = -1;
        int choose_pos = -1;
        double choose_score = -1e100;

        for (int t = 0; t < (int)nodes.size(); ++t) {
            int node = nodes[t];

            double best = 1e100;
            double second = 1e100;
            int br = -1;
            int bp = -1;

            int R = (int)routes.size();
            for (int rr = 0; rr < R; ++rr) {
                if (loads[rr] + a[node].demand > CAP) {
                    continue;
                }

                for (int p = 0; p <= (int)routes[rr].size(); ++p) {
                    double d = insert_delta(routes[rr], p, node);
                    if (d < best) {
                        second = best;
                        best = d;
                        br = rr;
                        bp = p;
                    } else if (d < second) {
                        second = d;
                    }
                }
            }

            if (
                allow_empty_new &&
                (int)routes.size() < max_routes &&
                a[node].demand <= CAP
            ) {
                double d = 2.0 * distm[0][node];
                if (d < best) {
                    second = best;
                    best = d;
                    br = (int)routes.size();
                    bp = 0;
                } else if (d < second) {
                    second = d;
                }
            }

            if (br == -1) {
                continue;
            }

            double regret = (second >= 1e90 ? 1e6 : second - best);
            double score = regret - 0.03 * best;

            if (score > choose_score) {
                choose_score = score;
                choose_idx = t;
                choose_route = br;
                choose_pos = bp;
            }
        }

        if (choose_idx == -1) {
            return false;
        }

        int node = nodes[choose_idx];

        if (choose_route == (int)routes.size()) {
            routes.push_back(vector<int>{node});
            loads.push_back(a[node].demand);
        } else {
            routes[choose_route].insert(routes[choose_route].begin() + choose_pos, node);
            loads[choose_route] += a[node].demand;
        }

        nodes.erase(nodes.begin() + choose_idx);
    }

    return true;
}

Solution make_sweep_solution(double offset, bool reverse_order, double noise) {
    vector<int> nodes;
    for (int i = 1; i < N; ++i) {
        nodes.push_back(i);
    }

    vector<double> key(N, 0.0);
    const double PI = acos(-1.0);

    uniform_real_distribution<double> U(-noise, noise);

    for (int i = 1; i < N; ++i) {
        double ang = atan2(a[i].y - a[0].y, a[i].x - a[0].x);
        if (ang < 0) {
            ang += 2.0 * PI;
        }

        double k = ang - offset;
        while (k < 0) {
            k += 2.0 * PI;
        }
        while (k >= 2.0 * PI) {
            k -= 2.0 * PI;
        }

        key[i] = k + U(rng);
    }

    sort(nodes.begin(), nodes.end(), [&](int x, int y) {
        return key[x] < key[y];
    });

    if (reverse_order) {
        reverse(nodes.begin(), nodes.end());
    }

    vector<vector<int>> routes(V);
    vector<int> loads(V, 0);

    for (int node : nodes) {
        int best_r = -1;
        int best_p = -1;
        double best = 1e100;

        for (int rr = 0; rr < V; ++rr) {
            if (loads[rr] + a[node].demand > CAP) {
                continue;
            }

            for (int p = 0; p <= (int)routes[rr].size(); ++p) {
                double d = insert_delta(routes[rr], p, node);
                double balance_penalty =
                    0.01 * distm[0][node] * (double)loads[rr] / max(1, CAP);

                if (d + balance_penalty < best) {
                    best = d + balance_penalty;
                    best_r = rr;
                    best_p = p;
                }
            }
        }

        if (best_r == -1) {
            Solution bad;
            bad.r.assign(V, {});
            recompute(bad);
            bad.cost = 1e100;
            return bad;
        }

        routes[best_r].insert(routes[best_r].begin() + best_p, node);
        loads[best_r] += a[node].demand;
    }

    Solution sol;
    sol.r = routes;
    recompute(sol);
    return sol;
}

Solution make_savings_solution(double noise) {
    vector<vector<int>> routes;
    vector<int> loads;
    vector<int> rid(N, -1);

    routes.reserve(N - 1);
    loads.reserve(N - 1);

    for (int i = 1; i < N; ++i) {
        routes.push_back(vector<int>{i});
        loads.push_back(a[i].demand);
        rid[i] = (int)routes.size() - 1;
    }

    struct Save {
        int i;
        int j;
        double s;
    };

    vector<Save> sv;
    sv.reserve((N - 1) * (N - 2) / 2);

    uniform_real_distribution<double> U(-noise, noise);

    for (int i = 1; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double s = distm[0][i] + distm[0][j] - distm[i][j] + U(rng);
            sv.push_back(Save{i, j, s});
        }
    }

    sort(sv.begin(), sv.end(), [](const Save& A, const Save& B) {
        return A.s > B.s;
    });

    for (const auto& e : sv) {
        int ri = rid[e.i];
        int rj = rid[e.j];

        if (ri < 0 || rj < 0 || ri == rj) {
            continue;
        }
        if (routes[ri].empty() || routes[rj].empty()) {
            continue;
        }
        if (loads[ri] + loads[rj] > CAP) {
            continue;
        }

        if (routes[ri].front() == e.i) {
            reverse(routes[ri].begin(), routes[ri].end());
        }
        if (routes[ri].back() != e.i) {
            continue;
        }

        if (routes[rj].back() == e.j) {
            reverse(routes[rj].begin(), routes[rj].end());
        }
        if (routes[rj].front() != e.j) {
            continue;
        }

        for (int x : routes[rj]) {
            routes[ri].push_back(x);
            rid[x] = ri;
        }

        loads[ri] += loads[rj];
        routes[rj].clear();
        loads[rj] = 0;
    }

    vector<vector<int>> compact;
    for (const auto& r : routes) {
        if (!r.empty()) {
            compact.push_back(r);
        }
    }

    while ((int)compact.size() > V) {
        vector<int> order(compact.size());
        iota(order.begin(), order.end(), 0);

        sort(order.begin(), order.end(), [&](int x, int y) {
            return route_load(compact[x]) < route_load(compact[y]);
        });

        bool ok = false;

        for (int idx : order) {
            vector<vector<int>> tmp = compact;
            vector<int> removed = tmp[idx];

            tmp.erase(tmp.begin() + idx);

            sort(removed.begin(), removed.end(), [&](int x, int y) {
                return a[x].demand > a[y].demand;
            });

            if (regret_insert_all(tmp, removed, V, false)) {
                compact.swap(tmp);
                ok = true;
                break;
            }
        }

        if (!ok) {
            break;
        }
    }

    if ((int)compact.size() > V) {
        Solution bad;
        bad.r.assign(V, {});
        recompute(bad);
        bad.cost = 1e100;
        return bad;
    }

    while ((int)compact.size() < V) {
        compact.push_back(vector<int>{});
    }

    Solution sol;
    sol.r = compact;
    recompute(sol);
    return sol;
}

bool improve_2opt_route(vector<int>& r) {
    int m = (int)r.size();

    if (m < 3) {
        return false;
    }

    double best = -EPS;
    int bi = -1;
    int bk = -1;

    for (int i = 0; i < m - 1; ++i) {
        int A = (i == 0 ? 0 : r[i - 1]);
        int B = r[i];

        for (int k = i + 1; k < m; ++k) {
            int C = r[k];
            int D = (k + 1 == m ? 0 : r[k + 1]);

            double delta =
                distm[A][C] + distm[B][D] -
                distm[A][B] - distm[C][D];

            if (delta < best) {
                best = delta;
                bi = i;
                bk = k;
            }
        }
    }

    if (bi != -1) {
        reverse(r.begin() + bi, r.begin() + bk + 1);
        return true;
    }

    return false;
}

bool improve_intra_2opt(Solution& sol) {
    bool changed = false;

    for (auto& r : sol.r) {
        int iter = 0;

        while (time_left() && improve_2opt_route(r)) {
            changed = true;
            ++iter;

            if (iter > 2000) {
                break;
            }
        }
    }

    if (changed) {
        recompute(sol);
    }

    return changed;
}

bool improve_relocate(Solution& sol) {
    int R = (int)sol.r.size();

    double best = -EPS;
    int ba = -1;
    int bb = -1;
    int bi = -1;
    int bp = -1;

    for (int ra = 0; ra < R; ++ra) {
        for (int i = 0; i < (int)sol.r[ra].size(); ++i) {
            int node = sol.r[ra][i];
            double rem = remove_delta(sol.r[ra], i);

            for (int rb = 0; rb < R; ++rb) {
                if (rb == ra) {
                    continue;
                }

                if (sol.load[rb] + a[node].demand > CAP) {
                    continue;
                }

                for (int p = 0; p <= (int)sol.r[rb].size(); ++p) {
                    double delta = rem + insert_delta(sol.r[rb], p, node);

                    if (delta < best) {
                        best = delta;
                        ba = ra;
                        bb = rb;
                        bi = i;
                        bp = p;
                    }
                }
            }
        }
    }

    if (ba != -1) {
        int node = sol.r[ba][bi];

        sol.r[ba].erase(sol.r[ba].begin() + bi);
        sol.r[bb].insert(sol.r[bb].begin() + bp, node);

        sol.load[ba] -= a[node].demand;
        sol.load[bb] += a[node].demand;
        sol.cost += best;

        return true;
    }

    return false;
}

bool improve_swap(Solution& sol) {
    int R = (int)sol.r.size();

    double best = -EPS;
    int ba = -1;
    int bb = -1;
    int bi = -1;
    int bj = -1;

    for (int ra = 0; ra < R; ++ra) {
        for (int rb = ra + 1; rb < R; ++rb) {
            for (int i = 0; i < (int)sol.r[ra].size(); ++i) {
                int x = sol.r[ra][i];

                for (int j = 0; j < (int)sol.r[rb].size(); ++j) {
                    int y = sol.r[rb][j];

                    int la = sol.load[ra] - a[x].demand + a[y].demand;
                    int lb = sol.load[rb] - a[y].demand + a[x].demand;

                    if (la > CAP || lb > CAP) {
                        continue;
                    }

                    double delta =
                        replace_delta(sol.r[ra], i, y) +
                        replace_delta(sol.r[rb], j, x);

                    if (delta < best) {
                        best = delta;
                        ba = ra;
                        bb = rb;
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
    }

    if (ba != -1) {
        int x = sol.r[ba][bi];
        int y = sol.r[bb][bj];

        sol.r[ba][bi] = y;
        sol.r[bb][bj] = x;

        sol.load[ba] = sol.load[ba] - a[x].demand + a[y].demand;
        sol.load[bb] = sol.load[bb] - a[y].demand + a[x].demand;
        sol.cost += best;

        return true;
    }

    return false;
}

bool improve_2opt_star(Solution& sol) {
    int R = (int)sol.r.size();

    double best = -EPS;
    int ba = -1;
    int bb = -1;
    int bi = -2;
    int bj = -2;

    for (int ra = 0; ra < R; ++ra) {
        int ma = (int)sol.r[ra].size();

        vector<int> sufA(ma + 1, 0);
        for (int i = ma - 1; i >= 0; --i) {
            sufA[i] = sufA[i + 1] + a[sol.r[ra][i]].demand;
        }

        for (int rb = ra + 1; rb < R; ++rb) {
            int mb = (int)sol.r[rb].size();

            vector<int> sufB(mb + 1, 0);
            for (int j = mb - 1; j >= 0; --j) {
                sufB[j] = sufB[j + 1] + a[sol.r[rb][j]].demand;
            }

            for (int i = -1; i < ma; ++i) {
                int tailA = sufA[i + 1];

                int A = (i == -1 ? 0 : sol.r[ra][i]);
                int B = (i + 1 == ma ? 0 : sol.r[ra][i + 1]);

                for (int j = -1; j < mb; ++j) {
                    int tailB = sufB[j + 1];

                    if (tailA == 0 && tailB == 0) {
                        continue;
                    }

                    int newLa = sol.load[ra] - tailA + tailB;
                    int newLb = sol.load[rb] - tailB + tailA;

                    if (newLa > CAP || newLb > CAP) {
                        continue;
                    }

                    int C = (j == -1 ? 0 : sol.r[rb][j]);
                    int D = (j + 1 == mb ? 0 : sol.r[rb][j + 1]);

                    double delta =
                        distm[A][D] + distm[C][B] -
                        distm[A][B] - distm[C][D];

                    if (delta < best) {
                        best = delta;
                        ba = ra;
                        bb = rb;
                        bi = i;
                        bj = j;
                    }
                }
            }
        }
    }

    if (ba != -1) {
        vector<int> nr1;
        vector<int> nr2;

        for (int t = 0; t <= bi; ++t) {
            nr1.push_back(sol.r[ba][t]);
        }
        for (int t = bj + 1; t < (int)sol.r[bb].size(); ++t) {
            nr1.push_back(sol.r[bb][t]);
        }

        for (int t = 0; t <= bj; ++t) {
            nr2.push_back(sol.r[bb][t]);
        }
        for (int t = bi + 1; t < (int)sol.r[ba].size(); ++t) {
            nr2.push_back(sol.r[ba][t]);
        }

        sol.r[ba].swap(nr1);
        sol.r[bb].swap(nr2);

        recompute(sol);
        return true;
    }

    return false;
}

void local_search(Solution& sol) {
    recompute(sol);

    bool changed = true;
    int rounds = 0;

    while (time_left() && changed) {
        changed = false;

        if (improve_intra_2opt(sol)) {
            changed = true;
        }

        if (time_left() && improve_relocate(sol)) {
            changed = true;
        }

        if (time_left() && improve_swap(sol)) {
            changed = true;
        }

        if (time_left() && improve_2opt_star(sol)) {
            changed = true;
        }

        ++rounds;

        if (rounds > 2000) {
            break;
        }
    }

    improve_intra_2opt(sol);
    recompute(sol);
}

bool feasible(const Solution& sol) {
    vector<int> seen(N, 0);

    if ((int)sol.r.size() != V) {
        return false;
    }

    for (int rr = 0; rr < V; ++rr) {
        int l = 0;

        for (int x : sol.r[rr]) {
            if (x <= 0 || x >= N) {
                return false;
            }

            seen[x]++;
            l += a[x].demand;
        }

        if (l > CAP) {
            return false;
        }
    }

    for (int i = 1; i < N; ++i) {
        if (seen[i] != 1) {
            return false;
        }
    }

    return true;
}

Solution lns_step(const Solution& base, int remove_count) {
    Solution sol = base;

    vector<int> all;
    for (int rr = 0; rr < V; ++rr) {
        for (int x : sol.r[rr]) {
            all.push_back(x);
        }
    }

    if (all.empty()) {
        return sol;
    }

    shuffle(all.begin(), all.end(), rng);

    remove_count = min(remove_count, (int)all.size());

    vector<int> removed(all.begin(), all.begin() + remove_count);

    vector<char> take(N, 0);
    for (int x : removed) {
        take[x] = 1;
    }

    for (int rr = 0; rr < V; ++rr) {
        vector<int> nr;

        for (int x : sol.r[rr]) {
            if (!take[x]) {
                nr.push_back(x);
            }
        }

        sol.r[rr].swap(nr);
    }

    recompute(sol);

    sort(removed.begin(), removed.end(), [&](int x, int y) {
        return a[x].demand > a[y].demand;
    });

    if (!regret_insert_all(sol.r, removed, V, false)) {
        Solution bad = base;
        return bad;
    }

    recompute(sol);
    local_search(sol);

    return sol;
}


void write_solution(ostream& out, const Solution& sol) {
    out.setf(ios::fixed);
    out << setprecision(6) << sol.cost << " 0\n";

    for (int rr = 0; rr < V; ++rr) {
        out << 0;

        for (int x : sol.r[rr]) {
            out << ' ' << x;
        }

        out << ' ' << 0 << "\n";
    }
}

string fmt_double(double x, int prec = 6) {
    if (x >= 1e90) {
        return "INF";
    }

    ostringstream ss;
    ss.setf(ios::fixed);
    ss << setprecision(prec) << x;
    return ss.str();
}

string csv_escape(const string& s) {
    bool need_quotes = false;

    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') {
            need_quotes = true;
            break;
        }
    }

    if (!need_quotes) {
        return s;
    }

    string res = "\"";
    for (char c : s) {
        if (c == '"') {
            res += "\"\"";
        } else {
            res += c;
        }
    }
    res += "\"";
    return res;
}

ofstream* PHASE_CSV = nullptr;
string CURRENT_FILE_NAME = "single";

void write_phase_csv_header(ofstream& out) {
    out << "file,phase,candidate,status,constructed_cost,after_local_cost,"
        << "best_before,best_after,improvement,time_seconds,note\n";
}

void log_phase_csv(
    const string& phase,
    const string& candidate,
    const string& status,
    double constructed_cost,
    double after_local_cost,
    double best_before,
    double best_after,
    double improvement,
    double time_seconds,
    const string& note
) {
    if (PHASE_CSV == nullptr) {
        return;
    }

    (*PHASE_CSV) << csv_escape(CURRENT_FILE_NAME) << ','
                 << csv_escape(phase) << ','
                 << csv_escape(candidate) << ','
                 << csv_escape(status) << ','
                 << fmt_double(constructed_cost) << ','
                 << fmt_double(after_local_cost) << ','
                 << fmt_double(best_before) << ','
                 << fmt_double(best_after) << ','
                 << fmt_double(improvement) << ','
                 << fmt_double(time_seconds, 3) << ','
                 << csv_escape(note) << "\n";

    PHASE_CSV->flush();
}

struct PhaseStats {
    string name;
    int candidates = 0;
    int feasible = 0;
    int improved = 0;
    double best_before = 1e100;
    double best_after = 1e100;
    double start_time_sec = 0.0;
    double end_time_sec = 0.0;
};

void begin_phase(PhaseStats& st, const string& name, double current_best, bool verbose) {
    st = PhaseStats();
    st.name = name;
    st.best_before = current_best;
    st.best_after = current_best;
    st.start_time_sec = elapsed();

    if (verbose) {
        cerr << "\n[phase-begin] " << name
             << " best_before=" << fmt_double(current_best)
             << " time=" << fmt_double(elapsed(), 3) << "s\n";
    }
}

void end_phase(PhaseStats& st, double current_best, bool verbose) {
    st.best_after = current_best;
    st.end_time_sec = elapsed();

    double gain = st.best_before - st.best_after;
    bool useful = gain > 1e-7;

    if (verbose) {
        cerr << "[phase-end] " << st.name
             << " candidates=" << st.candidates
             << " feasible=" << st.feasible
             << " improved=" << st.improved
             << " best_before=" << fmt_double(st.best_before)
             << " best_after=" << fmt_double(st.best_after)
             << " gain=" << fmt_double(max(0.0, gain))
             << " useful=" << (useful ? "YES" : "NO")
             << " phase_time=" << fmt_double(st.end_time_sec - st.start_time_sec, 3) << "s\n";
    }

    log_phase_csv(
        st.name,
        "PHASE_SUMMARY",
        useful ? "USEFUL" : "NOT_USEFUL",
        1e100,
        1e100,
        st.best_before,
        st.best_after,
        max(0.0, gain),
        st.end_time_sec - st.start_time_sec,
        "candidates=" + to_string(st.candidates) +
        "; feasible=" + to_string(st.feasible) +
        "; improved=" + to_string(st.improved)
    );
}

bool read_instance_from_file(const fs::path& filename) {
    ifstream in(filename);

    if (!in) {
        return false;
    }

    if (!(in >> N >> V >> CAP)) {
        return false;
    }

    a.assign(N, Customer{0, 0.0, 0.0});

    for (int i = 0; i < N; ++i) {
        if (!(in >> a[i].demand >> a[i].x >> a[i].y)) {
            return false;
        }
    }

    build_dist();
    return true;
}

void try_candidate_solution(
    Solution sol,
    const string& phase,
    const string& candidate,
    Solution& best,
    PhaseStats& st,
    bool verbose
) {
    ++st.candidates;

    if (!time_left()) {
        log_phase_csv(phase, candidate, "TIME_LIMIT", 1e100, 1e100, best.cost, best.cost, 0.0, elapsed(), "candidate skipped by time limit");
        return;
    }

    double constructed = sol.cost;
    double best_before = best.cost;

    if (!feasible(sol)) {
        if (verbose) {
            cerr << "[candidate] phase=" << phase
                 << " name=" << candidate
                 << " status=INFEASIBLE"
                 << " constructed=" << fmt_double(constructed)
                 << " best=" << fmt_double(best.cost) << "\n";
        }

        log_phase_csv(phase, candidate, "INFEASIBLE", constructed, 1e100, best_before, best.cost, 0.0, elapsed(), "construction did not produce feasible solution");
        return;
    }

    ++st.feasible;

    double before_local = sol.cost;
    local_search(sol);
    double after_local = sol.cost;
    double local_gain = before_local - after_local;

    bool improved = false;
    if (feasible(sol) && sol.cost + 1e-7 < best.cost) {
        best = sol;
        ++st.improved;
        improved = true;
    }

    double global_gain = best_before - best.cost;

    if (verbose) {
        cerr << "[candidate] phase=" << phase
             << " name=" << candidate
             << " status=OK"
             << " constructed=" << fmt_double(constructed)
             << " after_local=" << fmt_double(after_local)
             << " local_gain=" << fmt_double(max(0.0, local_gain))
             << " best_before=" << fmt_double(best_before)
             << " best_after=" << fmt_double(best.cost)
             << " useful=" << (improved ? "YES" : "NO")
             << " time=" << fmt_double(elapsed(), 3) << "s\n";
    }

    log_phase_csv(
        phase,
        candidate,
        improved ? "IMPROVED_BEST" : "OK_NO_BEST",
        constructed,
        after_local,
        best_before,
        best.cost,
        max(0.0, global_gain),
        elapsed(),
        "local_gain=" + fmt_double(max(0.0, local_gain))
    );
}

Solution solve_instance(bool verbose) {
    rng.seed(1234567);
    start_time = chrono::steady_clock::now();

    Solution best;
    best.r.assign(V, {});
    recompute(best);
    best.cost = 1e100;

    if (verbose) {
        cerr << "[solve] start instance=" << CURRENT_FILE_NAME
             << " n=" << N
             << " v=" << V
             << " capacity=" << CAP
             << " time_limit=" << TIME_LIMIT << "s\n";
        cerr << "[mode] simplified without fallback-regret: savings + sweep-grid + local_search + LNS\n";
    }

    PhaseStats st;

    begin_phase(st, "savings", best.cost, verbose);
    try_candidate_solution(make_savings_solution(0.0), "savings", "noise=0", best, st, verbose);
    end_phase(st, best.cost, verbose);

    const double PI = acos(-1.0);

    begin_phase(st, "sweep-grid", best.cost, verbose);
    for (int t = 0; t < 40 && time_left(); ++t) {
        double offset = 2.0 * PI * (double)t / 40.0;

        try_candidate_solution(
            make_sweep_solution(offset, false, 0.0),
            "sweep-grid",
            "try=" + to_string(t) + ";dir=forward;offset=" + fmt_double(offset, 3),
            best,
            st,
            verbose
        );

        if (time_left()) {
            try_candidate_solution(
                make_sweep_solution(offset, true, 0.0),
                "sweep-grid",
                "try=" + to_string(t) + ";dir=reverse;offset=" + fmt_double(offset, 3),
                best,
                st,
                verbose
            );
        }
    }
    end_phase(st, best.cost, verbose);

    if (!feasible(best)) {
        if (verbose) {
            cerr << "[LNS] skipped: savings and sweep-grid did not produce a feasible solution, "
                 << "and fallback-regret is disabled\n";
        }

        log_phase_csv(
            "LNS",
            "SKIPPED",
            "NO_INITIAL_FEASIBLE",
            1e100,
            1e100,
            best.cost,
            best.cost,
            0.0,
            elapsed(),
            "fallback-regret disabled; no feasible initial solution for LNS"
        );

        if (verbose) {
            cerr << "[solve-finish] instance=" << CURRENT_FILE_NAME
                 << " best=INF feasible=NO"
                 << " total_time=" << fmt_double(elapsed(), 3) << "s\n";
        }

        return best;
    }

    begin_phase(st, "LNS", best.cost, verbose);

    int iter = 0;
    int lns_feasible = 0;
    int lns_improved = 0;
    double last_log = elapsed();

    while (time_left()) {
        int customers = max(1, N - 1);

        int rem;
        if (customers <= 30) {
            // После удаления noisy-sweep маленькие тесты компенсируем более сильным LNS:
            // перебираем разрушения от 3 клиентов почти до всего маршрута.
            int max_rem = max(3, customers - 1);
            rem = 3 + (iter % max(1, max_rem - 2));
            rem = min(rem, max_rem);
        } else {
            rem = 4 + (iter % 7) * max(1, customers / 80);
            rem = min(rem, max(3, customers / 5));
        }

        double best_before = best.cost;
        Solution cand = lns_step(best, rem);

        ++st.candidates;

        if (feasible(cand)) {
            ++st.feasible;
            ++lns_feasible;

            if (cand.cost + 1e-7 < best.cost) {
                best = cand;
                ++st.improved;
                ++lns_improved;

                if (verbose) {
                    cerr.setf(ios::fixed);
                    cerr << setprecision(3)
                         << "[lns-improve] iter=" << iter
                         << " removed=" << rem
                         << " best_before=" << best_before
                         << " best_after=" << best.cost
                         << " gain=" << (best_before - best.cost)
                         << " time=" << elapsed() << "s\n";
                }

                log_phase_csv(
                    "LNS",
                    "iter=" + to_string(iter) + ";removed=" + to_string(rem),
                    "IMPROVED_BEST",
                    cand.cost,
                    cand.cost,
                    best_before,
                    best.cost,
                    max(0.0, best_before - best.cost),
                    elapsed(),
                    "LNS destroy-repair improved incumbent"
                );
            }
        } else {
            log_phase_csv(
                "LNS",
                "iter=" + to_string(iter) + ";removed=" + to_string(rem),
                "INFEASIBLE",
                cand.cost,
                cand.cost,
                best_before,
                best.cost,
                0.0,
                elapsed(),
                "LNS repair returned infeasible solution"
            );
        }

        ++iter;

        if (verbose && elapsed() - last_log >= 15.0) {
            cerr.setf(ios::fixed);
            cerr << setprecision(3)
                 << "[lns-progress] time=" << elapsed()
                 << "s iter=" << iter
                 << " feasible=" << lns_feasible
                 << " improved=" << lns_improved
                 << " current_best=" << best.cost
                 << " useful_so_far=" << (lns_improved > 0 ? "YES" : "NO") << "\n";

            log_phase_csv(
                "LNS",
                "PROGRESS",
                lns_improved > 0 ? "USEFUL_SO_FAR" : "NO_IMPROVEMENT_YET",
                1e100,
                1e100,
                st.best_before,
                best.cost,
                max(0.0, st.best_before - best.cost),
                elapsed(),
                "iter=" + to_string(iter) + "; feasible=" + to_string(lns_feasible) + "; improved=" + to_string(lns_improved)
            );

            last_log = elapsed();
        }

        long long iter_limit = (N <= 30 ? 5000000LL : 200000LL);
        if (iter > iter_limit) {
            break;
        }
    }

    end_phase(st, best.cost, verbose);

    recompute(best);

    if (verbose) {
        cerr.setf(ios::fixed);
        cerr << setprecision(3)
             << "[solve-finish] instance=" << CURRENT_FILE_NAME
             << " best=" << best.cost
             << " feasible=" << (feasible(best) ? "YES" : "NO")
             << " total_time=" << elapsed() << "s\n";
    }

    return best;
}

fs::path executable_dir(char** argv) {
    fs::path exe = argv[0];

    if (exe.has_parent_path()) {
        fs::path parent = exe.parent_path();

        if (parent.is_relative()) {
            return fs::absolute(parent).lexically_normal();
        }

        return parent.lexically_normal();
    }

    return fs::current_path();
}

bool is_input_file(const fs::path& p) {
    if (!fs::is_regular_file(p)) {
        return false;
    }

    string name = p.filename().string();

    if (name.empty()) {
        return false;
    }

    if (name[0] == '.') {
        return false;
    }

    if (name == "results.csv" || name == "phase_log.csv" || name == "checked_results.csv") {
        return false;
    }

    if (p.extension() == ".out" || p.extension() == ".csv") {
        return false;
    }

    return true;
}

vector<fs::path> collect_data_files(const fs::path& data_dir) {
    vector<fs::path> files;

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        fs::path p = entry.path();

        if (is_input_file(p)) {
            files.push_back(p);
        }
    }

    sort(files.begin(), files.end());
    return files;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc >= 2) {
        TIME_LIMIT = atof(argv[1]);
    }

    fs::path root = executable_dir(argv);
    fs::path data_dir = root / "Data";
    fs::path csv_path = data_dir / "results.csv";
    fs::path phase_csv_path = data_dir / "phase_log.csv";

    cerr << "[config] executable dir: " << root.string() << "\n";
    cerr << "[config] data dir: " << data_dir.string() << "\n";
    cerr << "[config] time limit per file: " << TIME_LIMIT << "s\n";

    if (!fs::exists(data_dir) || !fs::is_directory(data_dir)) {
        cerr << "[error] Data folder not found: " << data_dir.string() << "\n";
        return 1;
    }

    vector<fs::path> files = collect_data_files(data_dir);

    if (files.empty()) {
        cerr << "[error] no input files found in Data folder\n";
        return 1;
    }

    ofstream csv(csv_path);

    if (!csv) {
        cerr << "[error] cannot create csv: " << csv_path.string() << "\n";
        return 1;
    }

    ofstream phase_csv(phase_csv_path);

    if (!phase_csv) {
        cerr << "[error] cannot create phase csv: " << phase_csv_path.string() << "\n";
        return 1;
    }

    PHASE_CSV = &phase_csv;
    write_phase_csv_header(phase_csv);

    csv << "file,n,v,capacity,cost,time_seconds,solution_file,status\n";

    cerr << "[batch] files found: " << files.size() << "\n";
    cerr << "[batch] results csv: " << csv_path.string() << "\n";
    cerr << "[batch] phase log csv: " << phase_csv_path.string() << "\n";

    for (int idx = 0; idx < (int)files.size(); ++idx) {
        const fs::path& input_path = files[idx];
        string file_name = input_path.filename().string();
        CURRENT_FILE_NAME = file_name;

        cerr << "\n============================================================\n";
        cerr << "[batch] (" << (idx + 1) << "/" << files.size()
             << ") file=" << file_name << "\n";

        if (!read_instance_from_file(input_path)) {
            cerr << "[error] cannot read as VRP instance: "
                 << input_path.string() << "\n";
            csv << file_name << ",,,,,,,READ_ERROR\n";
            csv.flush();
            continue;
        }

        int total_demand = 0;
        int max_demand = 0;
        for (int i = 1; i < N; ++i) {
            total_demand += a[i].demand;
            max_demand = max(max_demand, a[i].demand);
        }

        cerr << "[instance] n=" << N
             << " v=" << V
             << " capacity=" << CAP
             << " total_demand=" << total_demand
             << " total_capacity=" << (long long)V * CAP
             << " max_demand=" << max_demand << "\n";

        if (total_demand > (long long)V * CAP || max_demand > CAP) {
            cerr << "[warning] necessary feasibility condition failed: "
                 << "total_demand <= V*capacity and max_demand <= capacity\n";
        }

        auto wall_start = chrono::steady_clock::now();
        Solution sol = solve_instance(true);
        auto wall_finish = chrono::steady_clock::now();

        double seconds = chrono::duration<double>(wall_finish - wall_start).count();

        if (!feasible(sol)) {
            cerr << "[done] file=" << file_name
                 << " status=NO_FEASIBLE"
                 << " time=" << fmt_double(seconds, 3) << "s\n";

            csv << file_name << ','
                << N << ','
                << V << ','
                << CAP << ','
                << "INF" << ','
                << fixed << setprecision(3) << seconds << ','
                << "" << ",NO_FEASIBLE\n";

            csv.flush();
            continue;
        }

        fs::path out_path = input_path;
        out_path += ".out";

        ofstream out(out_path);

        if (!out) {
            cerr << "[error] cannot write solution file: "
                 << out_path.string() << "\n";

            csv << file_name << ','
                << N << ','
                << V << ','
                << CAP << ','
                << fixed << setprecision(6) << sol.cost << ','
                << setprecision(3) << seconds << ','
                << out_path.filename().string() << ",WRITE_ERROR\n";

            csv.flush();
            continue;
        }

        write_solution(out, sol);
        out.close();

        csv << file_name << ','
            << N << ','
            << V << ','
            << CAP << ','
            << fixed << setprecision(6) << sol.cost << ','
            << setprecision(3) << seconds << ','
            << out_path.filename().string() << ",OK\n";

        csv.flush();

        cerr << fixed << setprecision(6)
             << "[done] file=" << file_name
             << " cost=" << sol.cost;

        cerr << setprecision(3)
             << " time=" << seconds << "s"
             << " solution=" << out_path.string() << "\n";
    }

    cerr << "\n[finish] results saved to " << csv_path.string() << "\n";
    cerr << "[finish] phase logs saved to " << phase_csv_path.string() << "\n";
    return 0;
}
