// CUTLASS f64 GEMM benchmark for the GPU benchmark contract suite.
//
// Usage:
//   benchmark_gpu_cutlass OUTPUT_JSONL DEVICE_ORDINAL PROBLEM_FILTER BACKEND... -- SUITE_JSON...
//
// SUITE_JSON files are JSON-converted suite files (from YAML via the Python runner).
// Only problems whose backend_candidates include "cutlass" are executed.

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSON output helpers (no external dependencies)
// ---------------------------------------------------------------------------

static std::string json_str(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + "\"";
}

static std::string json_null() { return "null"; }
static std::string json_double(double v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << v;
    return ss.str();
}
static std::string json_int(long long v) {
    return std::to_string(v);
}

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        std::ostringstream _ss; \
        _ss << "CUDA error at " __FILE__ ":" << __LINE__ \
            << " — " << cudaGetErrorString(_e); \
        throw std::runtime_error(_ss.str()); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Timing helper using CUDA events
// ---------------------------------------------------------------------------

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void record_start() { CUDA_CHECK(cudaEventRecord(start)); }
    void record_stop()  { CUDA_CHECK(cudaEventRecord(stop)); }
    float elapsed_ms() {
        float ms = 0;
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct TimingStats {
    double first_ms, median_ms, min_ms, p95_ms, iqr_ms;
};

static TimingStats compute_stats(std::vector<double> times) {
    size_t n = times.size();
    std::vector<double> s = times;
    std::sort(s.begin(), s.end());
    double first  = times[0];
    double median = (n % 2 == 0) ? (s[n/2-1] + s[n/2]) / 2.0 : s[n/2];
    double min_t  = s[0];
    double p95    = s[std::min((size_t)(0.95 * n), n - 1)];
    double q1     = s[std::max((size_t)(0.25 * n), (size_t)0)];
    double q3     = s[std::min((size_t)(0.75 * n), n - 1)];
    return { first, median, min_t, p95, q3 - q1 };
}

// ---------------------------------------------------------------------------
// CUTLASS GEMM types for float64
// ---------------------------------------------------------------------------

using CutlassGemmF64 = cutlass::gemm::device::Gemm<
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

// ---------------------------------------------------------------------------
// JSONL record builders
// ---------------------------------------------------------------------------

static std::string base_env_json(int device_ordinal) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_ordinal);
    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);

    std::ostringstream ss;
    ss << "{"
       << "\"hostname\":" << json_null() << ","
       << "\"timestamp_utc\":" << json_null() << ","
       << "\"os\":" << json_str("linux") << ","
       << "\"gpu_name\":" << json_str(props.name) << ","
       << "\"gpu_uuid\":" << json_null() << ","
       << "\"gpu_memory_bytes\":" << json_int((long long)props.totalGlobalMem) << ","
       << "\"driver_version\":" << json_str(std::to_string(driver_version)) << ","
       << "\"cuda_version\":" << json_null() << ","
       << "\"cudnn_version\":" << json_null() << ","
       << "\"framework_version\":" << json_str("cutlass-v3.7") << ","
       << "\"tenferro_rs_commit\":" << json_null() << ","
       << "\"benchmark_repo_commit\":" << json_null() << ","
       << "\"env\":{}"
       << "}";
    return ss.str();
}

static std::string ok_record_json(
    const std::string& suite_id, const std::string& problem_id,
    const std::string& op, const std::string& backend,
    int device_ordinal,
    int n_warmup, int n_runs,
    const TimingStats& stats,
    const std::string& ver_status,
    const std::string& layout_str, const std::string& dtype_str)
{
    std::ostringstream ss;
    ss << "{"
       << "\"schema_version\":1,"
       << "\"suite_id\":" << json_str(suite_id) << ","
       << "\"problem_id\":" << json_str(problem_id) << ","
       << "\"op\":" << json_str(op) << ","
       << "\"backend\":" << json_str(backend) << ","
       << "\"status\":" << json_str("ok") << ","
       << "\"timing\":{"
           << "\"warmup_runs\":" << json_int(n_warmup) << ","
           << "\"timed_runs\":" << json_int(n_runs) << ","
           << "\"compile_time_ms\":" << json_null() << ","
           << "\"first_run_ms\":" << json_double(stats.first_ms) << ","
           << "\"median_ms\":" << json_double(stats.median_ms) << ","
           << "\"min_ms\":" << json_double(stats.min_ms) << ","
           << "\"p95_ms\":" << json_double(stats.p95_ms) << ","
           << "\"iqr_ms\":" << json_double(stats.iqr_ms) << ","
           << "\"timing_scope\":" << json_str("steady_state_host_api_plus_device_sync")
       << "},"
       << "\"performance\":{\"tflops\":" << json_null()
           << ",\"effective_bandwidth_gbps\":" << json_null()
           << ",\"peak_memory_bytes\":" << json_null() << "},"
       << "\"verification\":{"
           << "\"status\":" << json_str(ver_status) << ","
           << "\"reference_backend\":" << json_null() << ","
           << "\"max_abs_error\":" << json_null() << ","
           << "\"max_rel_error\":" << json_null() << ","
           << "\"residual\":" << json_null() << ","
           << "\"rtol\":" << json_null() << ","
           << "\"atol\":" << json_null() << ","
           << "\"reason\":" << json_null()
       << "},"
       << "\"environment\":" << base_env_json(device_ordinal) << ","
       << "\"execution\":{"
           << "\"device\":" << json_str("cuda") << ","
           << "\"device_ordinal\":" << json_int(device_ordinal) << ","
           << "\"execution_path\":" << json_str("phase2-measured-cutlass") << ","
           << "\"synchronization\":" << json_str("cudaEventSynchronize") << ","
           << "\"layout\":" << json_str(layout_str) << ","
           << "\"dtype\":" << json_str(dtype_str) << ","
           << "\"notes\":" << json_str("CUTLASS f64 GEMM via cutlass::gemm::device::Gemm<double,...,Sm80>") << ","
           << "\"unsupported_reason\":" << json_null()
       << "}"
       << "}";
    return ss.str();
}

static std::string stub_record_json(
    const std::string& suite_id, const std::string& problem_id,
    const std::string& op, const std::string& backend,
    int device_ordinal, const std::string& status, const std::string& reason,
    const std::string& layout_str, const std::string& dtype_str)
{
    std::ostringstream ss;
    ss << "{"
       << "\"schema_version\":1,"
       << "\"suite_id\":" << json_str(suite_id) << ","
       << "\"problem_id\":" << json_str(problem_id) << ","
       << "\"op\":" << json_str(op) << ","
       << "\"backend\":" << json_str(backend) << ","
       << "\"status\":" << json_str(status) << ","
       << "\"timing\":{"
           << "\"warmup_runs\":0,\"timed_runs\":0,"
           << "\"compile_time_ms\":" << json_null() << ","
           << "\"first_run_ms\":" << json_null() << ","
           << "\"median_ms\":" << json_null() << ","
           << "\"min_ms\":" << json_null() << ","
           << "\"p95_ms\":" << json_null() << ","
           << "\"iqr_ms\":" << json_null() << ","
           << "\"timing_scope\":" << json_str("steady_state_host_api_plus_device_sync")
       << "},"
       << "\"performance\":{\"tflops\":" << json_null()
           << ",\"effective_bandwidth_gbps\":" << json_null()
           << ",\"peak_memory_bytes\":" << json_null() << "},"
       << "\"verification\":{"
           << "\"status\":" << json_str("skipped") << ","
           << "\"reference_backend\":" << json_null() << ","
           << "\"max_abs_error\":" << json_null() << ","
           << "\"max_rel_error\":" << json_null() << ","
           << "\"residual\":" << json_null() << ","
           << "\"rtol\":" << json_null() << ","
           << "\"atol\":" << json_null() << ","
           << "\"reason\":" << json_str(reason)
       << "},"
       << "\"environment\":" << base_env_json(device_ordinal) << ","
       << "\"execution\":{"
           << "\"device\":" << json_str("cuda") << ","
           << "\"device_ordinal\":" << json_int(device_ordinal) << ","
           << "\"execution_path\":" << json_str("phase2-runner") << ","
           << "\"synchronization\":" << json_str("not executed") << ","
           << "\"layout\":" << json_str(layout_str) << ","
           << "\"dtype\":" << json_str(dtype_str) << ","
           << "\"notes\":" << json_null() << ","
           << "\"unsupported_reason\":" << json_str(reason)
       << "}"
       << "}";
    return ss.str();
}

// ---------------------------------------------------------------------------
// CUTLASS GEMM benchmark
// ---------------------------------------------------------------------------

static std::string bench_cutlass_gemm(
    const std::string& suite_id, const std::string& problem_id,
    int m, int n, int k, int device_ordinal,
    int n_warmup, int n_runs,
    const std::string& layout_str, const std::string& dtype_str)
{
    CUDA_CHECK(cudaSetDevice(device_ordinal));

    // Allocate device memory
    size_t sz_a = (size_t)m * k * sizeof(double);
    size_t sz_b = (size_t)k * n * sizeof(double);
    size_t sz_c = (size_t)m * n * sizeof(double);
    double *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, sz_a));
    CUDA_CHECK(cudaMalloc(&d_b, sz_b));
    CUDA_CHECK(cudaMalloc(&d_c, sz_c));
    CUDA_CHECK(cudaMemset(d_a, 0, sz_a));
    CUDA_CHECK(cudaMemset(d_b, 0, sz_b));
    CUDA_CHECK(cudaMemset(d_c, 0, sz_c));

    double alpha = 1.0, beta = 0.0;

    CutlassGemmF64 gemm_op;
    CutlassGemmF64::Arguments args(
        {m, n, k},
        {d_a, k},
        {d_b, n},
        {d_c, n},
        {d_c, n},
        {alpha, beta}
    );

    // Check if CUTLASS can handle this problem
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
        return stub_record_json(suite_id, problem_id, "matmul", "cutlass",
                                device_ordinal, "not_configured",
                                "CUTLASS Sm80 f64 GEMM cannot implement this problem size",
                                layout_str, dtype_str);
    }

    // Warmup
    for (int i = 0; i < n_warmup; ++i) {
        status = gemm_op(args);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs using CUDA events
    CudaTimer timer;
    std::vector<double> times_ms;
    for (int i = 0; i < n_runs; ++i) {
        timer.record_start();
        gemm_op(args);
        timer.record_stop();
        times_ms.push_back((double)timer.elapsed_ms());
    }

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    TimingStats stats = compute_stats(times_ms);
    return ok_record_json(suite_id, problem_id, "matmul", "cutlass",
                          device_ordinal, n_warmup, n_runs, stats,
                          "skipped", layout_str, dtype_str);
}

// ---------------------------------------------------------------------------
// Simple JSON parser helpers (for reading suite JSON)
// ---------------------------------------------------------------------------

// Minimal string extract: find key and return its value as a string
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    auto end = json.find("\"", pos);
    return json.substr(pos, end - pos);
}

static int json_get_int(const std::string& json, const std::string& key, int def = 0) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    try { return std::stoi(json.substr(pos)); }
    catch (...) { return def; }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: benchmark_gpu_cutlass OUTPUT_JSONL DEVICE_ORDINAL "
                     "PROBLEM_FILTER BACKEND... -- SUITE_JSON...\n";
        return 1;
    }

    std::string out_path = argv[1];
    int device_ordinal = std::stoi(argv[2]);
    std::string problem_filter = argv[3];

    // Find "--" separator
    int sep = -1;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--") { sep = i; break; }
    }
    if (sep < 0) { std::cerr << "Missing -- separator\n"; return 1; }

    std::vector<std::string> backends(argv + 4, argv + sep);
    std::vector<std::string> suite_files(argv + sep + 1, argv + argc);

    bool handle_cutlass = false;
    for (auto& b : backends) {
        if (b == "cutlass") { handle_cutlass = true; break; }
    }
    if (!handle_cutlass) {
        // Nothing to do - write empty file
        std::ofstream(out_path).close();
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(device_ordinal));

    std::ofstream out(out_path);

    for (auto& sf : suite_files) {
        // Read JSON suite file
        std::ifstream f(sf);
        if (!f.is_open()) {
            std::cerr << "Cannot open suite file: " << sf << "\n";
            continue;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());

        std::string suite_id = json_get_str(content, "suite_id");

        // Find each problem block (very simple search for "matmul" ops only)
        // We look for problem blocks containing "cutlass" in backend_candidates
        // and op = "matmul" or "batched_matmul"

        // For now, handle only matmul with explicit CUTLASS GEMM
        // Parse problem blocks by finding "\"id\":" patterns
        size_t pos = 0;
        while ((pos = content.find("\"id\":", pos)) != std::string::npos) {
            pos += 5; // skip "id":
            // Extract problem id
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '"')) {
                if (content[pos] == '"') { pos++; break; }
                pos++;
            }
            auto end = content.find('"', pos);
            if (end == std::string::npos) break;
            std::string problem_id = content.substr(pos, end - pos);
            pos = end + 1;

            if (!problem_filter.empty() && problem_id != problem_filter) continue;

            // Find the surrounding problem block
            // Extract op
            auto op_start = content.rfind("\"op\":", end > 100 ? end - 100 : 0);
            if (op_start == std::string::npos) op_start = content.find("\"op\":", pos);
            std::string op = "";
            if (op_start != std::string::npos) {
                auto os = content.find('"', op_start + 5);
                auto oe = content.find('"', os + 1);
                if (os != std::string::npos && oe != std::string::npos)
                    op = content.substr(os + 1, oe - os - 1);
            }

            // Check if cutlass is in backend_candidates (simple text search in nearby region)
            auto candidates_end = content.find(']', content.find("\"backend_candidates\"", pos > 200 ? pos - 200 : 0));
            bool has_cutlass = (content.find("\"cutlass\"", pos > 200 ? pos - 200 : 0) < candidates_end + 100);

            if (!has_cutlass) continue;

            // Get matmul params
            auto matmul_pos = content.find("\"matmul\":", pos > 200 ? pos - 200 : 0);
            int m = 256, n = 256, k = 256, n_warmup = 3, n_runs = 7;

            if (matmul_pos != std::string::npos && matmul_pos < pos + 500) {
                // Extract m, n, k from the matmul block
                auto block_end = content.find('}', matmul_pos);
                auto block = content.substr(matmul_pos, block_end - matmul_pos + 1);
                m = json_get_int(block, "m", 256);
                n = json_get_int(block, "n", 256);
                k = json_get_int(block, "k", 256);
            }

            // Get run params
            auto run_pos = content.find("\"run\":", pos > 200 ? pos - 200 : 0);
            if (run_pos != std::string::npos && run_pos < pos + 500) {
                auto block_end = content.find('}', run_pos);
                auto block = content.substr(run_pos, block_end - run_pos + 1);
                n_warmup = json_get_int(block, "warmups", 3);
                n_runs = json_get_int(block, "runs", 7);
            }

            std::string layout_str = "{}";
            std::string dtype_str = "{}";

            std::string record;
            if (op == "matmul" || op == "einsum") {
                try {
                    record = bench_cutlass_gemm(suite_id, problem_id, m, n, k,
                                                device_ordinal, n_warmup, n_runs,
                                                layout_str, dtype_str);
                    record.replace(record.find("\"matmul\""), 8, "\"" + op + "\"");
                } catch (const std::exception& e) {
                    record = stub_record_json(suite_id, problem_id, op, "cutlass",
                                             device_ordinal, "runtime_failed",
                                             std::string("CUTLASS error: ") + e.what(),
                                             layout_str, dtype_str);
                }
            } else {
                record = stub_record_json(suite_id, problem_id, op, "cutlass",
                                         device_ordinal, "not_configured",
                                         "cutlass: op=" + op + " not implemented (matmul/einsum only)",
                                         layout_str, dtype_str);
            }

            out << record << "\n";
        }
    }

    return 0;
}
