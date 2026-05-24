#include <ATen/Parallel.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Args {
    int threads = 1;
    int runs = 7;
    int warmups = 3;
    std::string output;
};

struct Stats {
    double median_ms = 0.0;
    double iqr_ms = 0.0;
};

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        return fallback;
    }
}

std::string env_string(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

Args parse_args(int argc, char** argv) {
    Args args;
    args.threads = env_int("OMP_NUM_THREADS", 1);
    args.runs = env_int("BENCH_RUNS", env_string("PUBLICATION_GATE_PROFILE", "quick") == "full" ? 15 : 7);
    args.warmups = env_int("BENCH_WARMUPS", 3);

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--num-threads" && i + 1 < argc) {
            args.threads = std::stoi(argv[++i]);
        } else if (key == "--output" && i + 1 < argc) {
            args.output = argv[++i];
        } else if (key == "--runs" && i + 1 < argc) {
            args.runs = std::stoi(argv[++i]);
        } else if (key == "--warmups" && i + 1 < argc) {
            args.warmups = std::stoi(argv[++i]);
        }
    }

    if (args.output.empty()) {
        throw std::invalid_argument("--output is required");
    }
    return args;
}

bool suite_enabled(const std::string& suite) {
    const auto selected = env_string("PUBLICATION_GATE_SUITE", "all");
    return selected == "all" || selected == suite;
}

std::vector<int64_t> small_sizes() {
    return env_string("PUBLICATION_GATE_PROFILE", "quick") == "full"
        ? std::vector<int64_t>{2, 4, 8, 16, 32}
        : std::vector<int64_t>{2, 4, 8};
}

std::vector<int64_t> profile_sizes(std::vector<int64_t> quick, std::vector<int64_t> full) {
    return env_string("PUBLICATION_GATE_PROFILE", "quick") == "full" ? full : quick;
}

torch::Tensor data(const std::vector<int64_t>& shape, uint64_t seed, bool requires_grad = false) {
    int64_t len = 1;
    for (const auto dim : shape) {
        len *= dim;
    }
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(len));
    for (int64_t i = 0; i < len; ++i) {
        const auto x = static_cast<uint64_t>(i) * 6364136223846793005ULL
            + seed * 1442695040888963407ULL;
        values.push_back((static_cast<double>(x % 1024ULL) - 512.0) / 512.0);
    }
    auto tensor = torch::from_blob(values.data(), shape, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    tensor.set_requires_grad(requires_grad);
    return tensor;
}

torch::Tensor well_conditioned(int64_t n, uint64_t seed, bool requires_grad = false) {
    auto matrix = data({n, n}, seed) * 0.05;
    for (int64_t j = 0; j < n; ++j) {
        matrix.index_put_({j, j}, matrix.index({j, j}) + 1.0 + static_cast<double>(j) / static_cast<double>(std::max<int64_t>(n, 1)));
    }
    matrix.set_requires_grad(requires_grad);
    return matrix;
}

torch::Tensor spd(int64_t n, uint64_t seed, bool requires_grad = false) {
    auto base = well_conditioned(n, seed);
    auto matrix = base.transpose(0, 1).matmul(base) + torch::eye(n, torch::TensorOptions().dtype(torch::kFloat64));
    matrix.set_requires_grad(requires_grad);
    return matrix;
}

torch::Tensor batched_well_conditioned(int64_t n, int64_t batch, uint64_t seed, bool requires_grad = false) {
    std::vector<torch::Tensor> matrices;
    matrices.reserve(static_cast<std::size_t>(batch));
    for (int64_t i = 0; i < batch; ++i) {
        matrices.push_back(well_conditioned(n, seed + static_cast<uint64_t>(i)));
    }
    auto result = torch::stack(matrices, 0);
    result.set_requires_grad(requires_grad);
    return result;
}

torch::Tensor batched_spd(int64_t n, int64_t batch, uint64_t seed, bool requires_grad = false) {
    std::vector<torch::Tensor> matrices;
    matrices.reserve(static_cast<std::size_t>(batch));
    for (int64_t i = 0; i < batch; ++i) {
        matrices.push_back(spd(n, seed + static_cast<uint64_t>(i)));
    }
    auto result = torch::stack(matrices, 0);
    result.set_requires_grad(requires_grad);
    return result;
}

Stats compute_stats(std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const auto n = times.size();
    return Stats{times[n / 2], times[(3 * n) / 4] - times[n / 4]};
}

Stats bench(const Args& args, const std::function<torch::Tensor()>& fn) {
    torch::NoGradGuard no_grad;
    for (int i = 0; i < args.warmups; ++i) {
        volatile auto sink = fn().numel();
        (void)sink;
    }
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(args.runs));
    for (int i = 0; i < args.runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        volatile auto sink = fn().numel();
        (void)sink;
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return compute_stats(std::move(times));
}

Stats bench_grad(const Args& args, const std::function<torch::Tensor()>& fn) {
    for (int i = 0; i < args.warmups; ++i) {
        volatile auto sink = fn().numel();
        (void)sink;
    }
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(args.runs));
    for (int i = 0; i < args.runs; ++i) {
        const auto start = std::chrono::steady_clock::now();
        volatile auto sink = fn().numel();
        (void)sink;
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    return compute_stats(std::move(times));
}

void write_row(
    std::ofstream& out,
    const Args& args,
    const std::string& suite,
    const std::string& benchmark,
    const std::string& shape,
    const std::function<Stats()>& run
) {
    out << suite << "," << benchmark << ",f64," << args.threads << ",\"" << shape
        << "\",libtorch-cpu,";
    try {
        const auto stats = run();
        out << std::fixed << std::setprecision(6) << stats.median_ms << ","
            << stats.iqr_ms << ",ok\n";
    } catch (const std::exception& exc) {
        out << ",,\"error: " << exc.what() << "\"\n";
    }
}

torch::Tensor grad_matmul(int64_t n) {
    auto a = data({n, n}, 8, true);
    auto b = data({n, n}, 9, true);
    auto loss = a.matmul(b).sum();
    loss.backward();
    return a.grad();
}

torch::Tensor grad_svd(int64_t n) {
    auto a = well_conditioned(n, 10, true);
    auto result = torch::linalg_svd(a);
    auto loss = std::get<1>(result).sum();
    loss.backward();
    return a.grad();
}

torch::Tensor grad_solve(int64_t n) {
    auto a = spd(n, 11, true);
    auto b = data({n, 1}, 12, true);
    auto loss = torch::linalg_solve(a, b).sum();
    loss.backward();
    return a.grad();
}

torch::Tensor grad_solve_tensor(torch::Tensor a, torch::Tensor b) {
    auto loss = torch::linalg_solve(a, b).sum();
    loss.backward();
    return a.grad();
}

torch::Tensor grad_batched_matmul(int64_t n, int64_t batch) {
    auto a = data({batch, n, n}, 48, true);
    auto b = data({batch, n, n}, 49, true);
    auto loss = torch::einsum("bik,bkj->bij", {a, b}).sum();
    loss.backward();
    return a.grad();
}

void run_small(const Args& args, std::ofstream& out) {
    if (!suite_enabled("small")) {
        return;
    }
    for (const auto n : small_sizes()) {
        const auto shape = std::to_string(n) + "x" + std::to_string(n);
        write_row(out, args, "small", "matmul", shape, [&] {
            return bench(args, [&] { return data({n, n}, 1).matmul(data({n, n}, 2)); });
        });
        write_row(out, args, "small", "einsum_ij_jk_ik", shape, [&] {
            return bench(args, [&] { return torch::einsum("ij,jk->ik", {data({n, n}, 1), data({n, n}, 2)}); });
        });
        write_row(out, args, "small", "svd", shape, [&] {
            return bench(args, [&] { return std::get<1>(torch::linalg_svd(well_conditioned(n, 3))); });
        });
        write_row(out, args, "small", "qr", shape, [&] {
            return bench(args, [&] { return std::get<0>(torch::linalg_qr(well_conditioned(n, 4))); });
        });
        write_row(out, args, "small", "eigh", shape, [&] {
            return bench(args, [&] { return std::get<0>(torch::linalg_eigh(spd(n, 5))); });
        });
        for (const auto rhs_cols : {1, 4}) {
            write_row(out, args, "small", "solve", shape + ",rhs=" + std::to_string(rhs_cols), [&] {
                return bench(args, [&] { return torch::linalg_solve(spd(n, 6), data({n, rhs_cols}, 7)); });
            });
        }
        write_row(out, args, "small", "grad_sum_matmul_backward", shape, [&] {
            return bench_grad(args, [&] { return grad_matmul(n); });
        });
        write_row(out, args, "small", "grad_sum_svd_s_backward", shape, [&] {
            return bench_grad(args, [&] { return grad_svd(n); });
        });
        write_row(out, args, "small", "grad_sum_solve_backward", shape + ",rhs=1", [&] {
            return bench_grad(args, [&] { return grad_solve(n); });
        });
    }
}

void run_large(const Args& args, std::ofstream& out) {
    if (!suite_enabled("large")) {
        return;
    }
    for (const auto n : profile_sizes({128, 256}, {128, 256, 512, 1024})) {
        const auto shape = std::to_string(n) + "x" + std::to_string(n);
        write_row(out, args, "large", "matmul", shape, [&] {
            return bench(args, [&] { return data({n, n}, 21).matmul(data({n, n}, 22)); });
        });
    }
    const std::vector<std::tuple<int64_t, int64_t, int64_t>> rects{{1024, 256, 1024}, {256, 1024, 256}};
    for (const auto& rect : rects) {
        const auto m = std::get<0>(rect);
        const auto k = std::get<1>(rect);
        const auto n = std::get<2>(rect);
        if (env_string("PUBLICATION_GATE_PROFILE", "quick") != "full" && m > 256) {
            continue;
        }
        const auto shape = std::to_string(m) + "x" + std::to_string(k) + " * " + std::to_string(k) + "x" + std::to_string(n);
        write_row(out, args, "large", "matmul_rect", shape, [&] {
            return bench(args, [&] { return data({m, k}, 23).matmul(data({k, n}, 24)); });
        });
    }
    for (const auto n : profile_sizes({64}, {64, 128, 256})) {
        const auto shape = std::to_string(n) + "x" + std::to_string(n);
        write_row(out, args, "large", "svd", shape, [&] {
            return bench(args, [&] { return std::get<1>(torch::linalg_svd(well_conditioned(n, 25))); });
        });
        write_row(out, args, "large", "qr", shape, [&] {
            return bench(args, [&] { return std::get<0>(torch::linalg_qr(well_conditioned(n, 26))); });
        });
        write_row(out, args, "large", "eigh", shape, [&] {
            return bench(args, [&] { return std::get<0>(torch::linalg_eigh(spd(n, 27))); });
        });
        for (const auto rhs_cols : {1, 16, 64}) {
            write_row(out, args, "large", "solve", shape + ",rhs=" + std::to_string(rhs_cols), [&] {
                return bench(args, [&] { return torch::linalg_solve(spd(n, 28), data({n, rhs_cols}, 29)); });
            });
        }
    }
    for (const auto n : profile_sizes({64}, {64, 128})) {
        const auto shape = std::to_string(n) + "x" + std::to_string(n);
        write_row(out, args, "large", "grad_sum_matmul", shape, [&] {
            return bench_grad(args, [&] { return data({n, n}, 30, true).matmul(data({n, n}, 31, true)).sum(); });
        });
        write_row(out, args, "large", "grad_sum_matmul_backward", shape, [&] {
            return bench_grad(args, [&] {
                auto a = data({n, n}, 32, true);
                auto b = data({n, n}, 33, true);
                auto loss = a.matmul(b).sum();
                loss.backward();
                return a.grad();
            });
        });
        write_row(out, args, "large", "grad_sum_svd_s_backward", shape, [&] {
            return bench_grad(args, [&] {
                auto a = well_conditioned(n, 34, true);
                auto loss = std::get<1>(torch::linalg_svd(a)).sum();
                loss.backward();
                return a.grad();
            });
        });
        write_row(out, args, "large", "grad_sum_solve_backward", shape + ",rhs=1", [&] {
            return bench_grad(args, [&] { return grad_solve_tensor(spd(n, 35, true), data({n, 1}, 36, true)); });
        });
    }
}

void run_batched(const Args& args, std::ofstream& out) {
    if (!suite_enabled("batched")) {
        return;
    }
    for (const auto batch : profile_sizes({16, 64}, {16, 64, 256, 1024})) {
        for (const auto n : profile_sizes({2, 4}, {2, 4, 8, 16})) {
            const auto shape = std::to_string(n) + "x" + std::to_string(n) + "xbatch" + std::to_string(batch) + " (rightmost batch)";
            write_row(out, args, "batched", "batched_matmul_ikb_kjb_ijb", shape, [&] {
                return bench(args, [&] { return torch::einsum("bik,bkj->bij", {data({batch, n, n}, 41), data({batch, n, n}, 42)}); });
            });
            write_row(out, args, "batched", "batched_svd", shape, [&] {
                return bench(args, [&] { return std::get<1>(torch::linalg_svd(batched_well_conditioned(n, batch, 43))); });
            });
            write_row(out, args, "batched", "batched_qr", shape, [&] {
                return bench(args, [&] { return std::get<0>(torch::linalg_qr(batched_well_conditioned(n, batch, 44))); });
            });
            write_row(out, args, "batched", "batched_eigh", shape, [&] {
                return bench(args, [&] { return std::get<0>(torch::linalg_eigh(batched_spd(n, batch, 45))); });
            });
            write_row(out, args, "batched", "batched_solve", shape + ",rhs=1", [&] {
                return bench(args, [&] { return torch::linalg_solve(batched_spd(n, batch, 46), data({batch, n, 1}, 47)); });
            });
            write_row(out, args, "batched", "grad_sum_batched_matmul_backward", shape, [&] {
                return bench_grad(args, [&] { return grad_batched_matmul(n, batch); });
            });
            write_row(out, args, "batched", "grad_sum_batched_solve_backward", shape + ",rhs=1", [&] {
                return bench_grad(args, [&] { return grad_solve_tensor(batched_spd(n, batch, 50, true), data({batch, n, 1}, 51, true)); });
            });
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        at::set_num_threads(args.threads);
        std::ofstream out(args.output, std::ios::app);
        if (!out) {
            throw std::runtime_error("failed to open output CSV: " + args.output);
        }
        run_small(args, out);
        run_large(args, out);
        run_batched(args, out);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << "\n";
        return 1;
    }
    return 0;
}
