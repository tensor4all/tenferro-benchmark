#include "einsum_plan.h"

#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr int kNumWarmup = 3;
constexpr int kNumRuns = 15;

struct PathMeta {
    std::vector<std::pair<std::size_t, std::size_t>> path;
    double log2_size = 0.0;
    double log10_flops = 0.0;
};

struct Instance {
    std::string name;
    std::string format_string;
    std::vector<std::vector<std::int64_t>> shapes;
    std::string dtype;
    std::size_t num_tensors = 0;
    PathMeta opt_size;
    PathMeta opt_flops;
};

struct ParsedLabels {
    std::vector<std::vector<std::string>> inputs;
    std::vector<std::string> output;
};

struct Stats {
    double median_ms = 0.0;
    double iqr_ms = 0.0;
};

std::string getenv_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : std::string(value);
}

int parse_num_threads() {
    const auto value = getenv_or("OMP_NUM_THREADS", "1");
    try {
        return std::max(1, std::stoi(value));
    } catch (const std::exception&) {
        return 1;
    }
}

PathMeta parse_path_meta(const nlohmann::json& json) {
    PathMeta meta;
    meta.log2_size = json.at("log2_size").get<double>();
    meta.log10_flops = json.at("log10_flops").get<double>();
    for (const auto& pair_json : json.at("path")) {
        if (pair_json.size() != 2) {
            throw std::runtime_error("path pair must have exactly two entries");
        }
        meta.path.emplace_back(
            pair_json.at(0).get<std::size_t>(),
            pair_json.at(1).get<std::size_t>()
        );
    }
    return meta;
}

Instance parse_instance(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }

    nlohmann::json json;
    input >> json;

    Instance instance;
    instance.name = json.at("name").get<std::string>();
    instance.format_string = json.value(
        "format_string_rowmajor",
        json.at("format_string").get<std::string>()
    );
    instance.dtype = json.value("dtype", "float64");
    instance.num_tensors = json.at("num_tensors").get<std::size_t>();

    for (const auto& shape_json : json.at("shapes")) {
        std::vector<std::int64_t> shape;
        for (const auto& dim : shape_json) {
            shape.push_back(dim.get<std::int64_t>());
        }
        instance.shapes.push_back(std::move(shape));
    }

    const auto& paths = json.at("paths");
    instance.opt_size = parse_path_meta(paths.at("opt_size"));
    instance.opt_flops = parse_path_meta(paths.at("opt_flops"));
    return instance;
}

std::vector<Instance> load_instances(const std::filesystem::path& data_dir, const std::string& filter) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".json") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    std::vector<Instance> instances;
    for (const auto& path : paths) {
        try {
            auto instance = parse_instance(path);
            if (filter.empty() || instance.name == filter) {
                instances.push_back(std::move(instance));
            }
        } catch (const std::exception& exc) {
            std::cerr << "Warning: skip " << path.filename().string()
                      << " (" << exc.what() << ")\n";
        }
    }
    return instances;
}

std::vector<torch::Tensor> create_operands(const Instance& instance) {
    std::vector<torch::Tensor> operands;
    operands.reserve(instance.shapes.size());
    const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    for (const auto& shape : instance.shapes) {
        operands.push_back(torch::zeros(shape, options));
    }
    return operands;
}

std::vector<std::string> split_utf8_labels(const std::string& text) {
    std::vector<std::string> labels;
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        std::size_t len = 1;
        if ((byte & 0x80U) == 0) {
            len = 1;
        } else if ((byte & 0xE0U) == 0xC0U) {
            len = 2;
        } else if ((byte & 0xF0U) == 0xE0U) {
            len = 3;
        } else if ((byte & 0xF8U) == 0xF0U) {
            len = 4;
        } else {
            throw std::runtime_error("invalid UTF-8 einsum label");
        }
        if (i + len > text.size()) {
            throw std::runtime_error("truncated UTF-8 einsum label");
        }
        labels.push_back(text.substr(i, len));
        i += len;
    }
    return labels;
}

ParsedLabels parse_label_format(const std::string& format) {
    const auto arrow = format.find("->");
    if (arrow == std::string::npos) {
        throw std::invalid_argument("einsum format must contain ->");
    }

    ParsedLabels parsed;
    const auto inputs = format.substr(0, arrow);
    std::size_t start = 0;
    while (start <= inputs.size()) {
        const auto comma = inputs.find(',', start);
        const auto end = comma == std::string::npos ? inputs.size() : comma;
        parsed.inputs.push_back(split_utf8_labels(inputs.substr(start, end - start)));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    parsed.output = split_utf8_labels(format.substr(arrow + 2));
    return parsed;
}

bool contains_label(const std::vector<std::string>& labels, const std::string& label) {
    return std::find(labels.begin(), labels.end(), label) != labels.end();
}

void append_unique_label(std::vector<std::string>& labels, const std::string& label) {
    if (!contains_label(labels, label)) {
        labels.push_back(label);
    }
}

std::vector<std::string> intermediate_output_labels(
    const std::vector<std::vector<std::string>>& subscripts,
    const std::vector<std::string>& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    if (lhs_index >= subscripts.size() || rhs_index >= subscripts.size()) {
        throw std::out_of_range("contraction index out of range");
    }
    if (lhs_index == rhs_index) {
        throw std::invalid_argument("cannot contract an operand with itself");
    }

    std::unordered_map<std::string, std::size_t> outside_counts;
    for (std::size_t i = 0; i < subscripts.size(); ++i) {
        if (i == lhs_index || i == rhs_index) {
            continue;
        }
        for (const auto& label : subscripts[i]) {
            outside_counts[label] += 1;
        }
    }

    std::vector<std::string> labels_in_order;
    for (const auto& label : subscripts[lhs_index]) {
        append_unique_label(labels_in_order, label);
    }
    for (const auto& label : subscripts[rhs_index]) {
        append_unique_label(labels_in_order, label);
    }

    std::vector<std::string> output;
    for (const auto& label : final_output) {
        if (contains_label(labels_in_order, label)) {
            append_unique_label(output, label);
        }
    }
    for (const auto& label : labels_in_order) {
        if (outside_counts[label] > 0) {
            append_unique_label(output, label);
        }
    }
    return output;
}

std::string torch_binary_equation(
    const std::vector<std::vector<std::string>>& subscripts,
    const std::vector<std::string>& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    static const std::string torch_labels =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const auto output = intermediate_output_labels(
        subscripts,
        final_output,
        lhs_index,
        rhs_index
    );

    std::vector<std::string> labels_in_order;
    for (const auto& label : subscripts[lhs_index]) {
        append_unique_label(labels_in_order, label);
    }
    for (const auto& label : subscripts[rhs_index]) {
        append_unique_label(labels_in_order, label);
    }
    for (const auto& label : output) {
        append_unique_label(labels_in_order, label);
    }
    if (labels_in_order.size() > torch_labels.size()) {
        throw std::runtime_error("binary contraction uses more than 52 unique labels");
    }

    std::unordered_map<std::string, char> mapped;
    for (std::size_t i = 0; i < labels_in_order.size(); ++i) {
        mapped.emplace(labels_in_order[i], torch_labels[i]);
    }

    auto encode = [&](const std::vector<std::string>& labels) {
        std::string result;
        result.reserve(labels.size());
        for (const auto& label : labels) {
            result.push_back(mapped.at(label));
        }
        return result;
    };

    return encode(subscripts[lhs_index]) + "," + encode(subscripts[rhs_index])
        + "->" + encode(output);
}

std::vector<std::vector<std::string>> contract_label_subscripts(
    const std::vector<std::vector<std::string>>& subscripts,
    const std::vector<std::string>& final_output,
    std::size_t lhs_index,
    std::size_t rhs_index
) {
    const auto output = intermediate_output_labels(
        subscripts,
        final_output,
        lhs_index,
        rhs_index
    );
    auto next = subscripts;
    const auto first_remove = std::max(lhs_index, rhs_index);
    const auto second_remove = std::min(lhs_index, rhs_index);
    next.erase(next.begin() + static_cast<std::ptrdiff_t>(first_remove));
    next.erase(next.begin() + static_cast<std::ptrdiff_t>(second_remove));
    next.push_back(output);
    return next;
}

torch::Tensor contract_once(
    const Instance& instance,
    const PathMeta& path_meta,
    const std::vector<torch::Tensor>& base_operands
) {
    auto parsed = parse_label_format(instance.format_string);
    auto subscripts = parsed.inputs;
    auto operands = base_operands;

    for (const auto& [lhs_index, rhs_index] : path_meta.path) {
        const auto equation = torch_binary_equation(
            subscripts,
            parsed.output,
            lhs_index,
            rhs_index
        );
        auto result = torch::einsum(
            equation,
            {operands.at(lhs_index), operands.at(rhs_index)}
        );

        const auto first_remove = std::max(lhs_index, rhs_index);
        const auto second_remove = std::min(lhs_index, rhs_index);
        operands.erase(operands.begin() + static_cast<std::ptrdiff_t>(first_remove));
        operands.erase(operands.begin() + static_cast<std::ptrdiff_t>(second_remove));
        operands.push_back(result);

        subscripts = contract_label_subscripts(
            subscripts,
            parsed.output,
            lhs_index,
            rhs_index
        );
    }

    if (operands.size() != 1) {
        throw std::runtime_error("contraction did not reduce to one output tensor");
    }
    return operands.front();
}

Stats compute_stats(std::vector<double> times_ms) {
    std::sort(times_ms.begin(), times_ms.end());
    const auto n = times_ms.size();
    return Stats{
        times_ms.at(n / 2),
        times_ms.at((3 * n) / 4) - times_ms.at(n / 4),
    };
}

Stats benchmark_instance(const Instance& instance, const PathMeta& path_meta) {
    if (instance.dtype.find("complex") != std::string::npos) {
        throw std::runtime_error("complex dtype (" + instance.dtype + ") not supported");
    }

    const auto operands = create_operands(instance);
    for (int i = 0; i < kNumWarmup; ++i) {
        auto result = contract_once(instance, path_meta, operands);
        volatile auto sink = result.numel();
        (void)sink;
    }

    std::vector<double> times_ms;
    times_ms.reserve(kNumRuns);
    for (int i = 0; i < kNumRuns; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto result = contract_once(instance, path_meta, operands);
        volatile auto sink = result.numel();
        (void)sink;
        const auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count()
        );
    }
    return compute_stats(std::move(times_ms));
}

void print_strategy(
    const std::string& strategy,
    const std::vector<Instance>& instances,
    const PathMeta& (*select_path)(const Instance&)
) {
    constexpr int col_w = 106;
    std::cout << "\nStrategy: " << strategy << "\n";
    std::cout << std::left << std::setw(50) << "Instance"
              << std::right << std::setw(8) << "Tensors"
              << std::setw(11) << "log10FLOPS"
              << std::setw(12) << "log2SIZE"
              << std::setw(13) << "Median (ms)"
              << std::setw(10) << "IQR (ms)" << "\n";
    std::cout << std::string(col_w, '-') << "\n";

    for (std::size_t i = 0; i < instances.size(); ++i) {
        const auto& instance = instances[i];
        const auto& path_meta = select_path(instance);
        std::cerr << "  [" << (i + 1) << "/" << instances.size() << "] "
                  << instance.name << "...\n";

        try {
            const auto stats = benchmark_instance(instance, path_meta);
            std::cout << std::left << std::setw(50) << instance.name
                      << std::right << std::setw(8) << instance.num_tensors
                      << std::setw(11) << std::fixed << std::setprecision(2) << path_meta.log10_flops
                      << std::setw(12) << std::fixed << std::setprecision(2) << path_meta.log2_size
                      << std::setw(13) << std::fixed << std::setprecision(3) << stats.median_ms
                      << std::setw(10) << std::fixed << std::setprecision(3) << stats.iqr_ms
                      << "\n";
        } catch (const std::exception& exc) {
            std::cerr << "  -> " << instance.name << " (error: " << exc.what() << ")\n";
            std::cout << std::left << std::setw(50) << instance.name
                      << std::right << std::setw(8) << instance.num_tensors
                      << std::setw(11) << std::fixed << std::setprecision(2) << path_meta.log10_flops
                      << std::setw(12) << std::fixed << std::setprecision(2) << path_meta.log2_size
                      << std::setw(13) << "SKIP"
                      << std::setw(10) << "-"
                      << "\n";
        }
    }
}

const PathMeta& opt_flops(const Instance& instance) {
    return instance.opt_flops;
}

const PathMeta& opt_size(const Instance& instance) {
    return instance.opt_size;
}

}  // namespace

int main() {
    const auto data_dir = std::filesystem::path(TENFERRO_BENCHMARK_DATA_DIR);
    const auto instance_filter = getenv_or("BENCH_INSTANCE", "");
    const auto num_threads = parse_num_threads();
    at::set_num_threads(num_threads);
    at::set_num_interop_threads(num_threads);

    auto instances = load_instances(data_dir, instance_filter);
    if (!instance_filter.empty() && instances.empty()) {
        std::cerr << "BENCH_INSTANCE=" << instance_filter << ": no matching instance found\n";
        return 1;
    }

    std::cout << "libtorch-cpu einsum benchmark suite\n";
    std::cout << "==================================\n";
    std::cout << "Loaded " << instances.size() << " instances from " << data_dir.string() << "\n";
    std::cout << "Backend: libtorch-cpu\n";
    std::cout << "OMP_NUM_THREADS=" << num_threads << "\n";
    std::cout << "Torch intra-op threads=" << at::get_num_threads() << "\n";
    std::cout << "Torch inter-op threads=" << at::get_num_interop_threads() << "\n";
    std::cout << "BLAS policy: OpenBLAS required by runner configuration\n";
    std::cout << "Timing: median ± IQR of " << kNumRuns << " runs ("
              << kNumWarmup << " warmup), path precomputed\n";

    print_strategy("opt_flops", instances, opt_flops);
    print_strategy("opt_size", instances, opt_size);
    return 0;
}
