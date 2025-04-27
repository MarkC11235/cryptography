// project_gpu_primes_scaled_styled.cu
// Compile with: nvcc -O3 -std=c++17 project_gpu_primes_scaled_styled.cu -o subset_sum_gpu_primes_scaled_styled

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda_runtime.h>

// --- Type aliases ------------------------------------------------------------
using ValueVector   = std::vector<double>;
using SelectionMask = std::vector<int>;
using Combination   = std::vector<int>;

// --- Utility print functions ------------------------------------------------
void print_error_report(double err) {
    std::cout << "Error: " << err << "\n";
}

void print_solution_mask(const SelectionMask &mask) {
    std::cout << "Mask: [";
    for (size_t i = 0; i < mask.size(); ++i) {
        std::cout << mask[i];
        if (i + 1 < mask.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

// --- Configuration structs --------------------------------------------------
enum class StarterAlgorithm { GreedyBackward };
struct PhaseConfig { int k, l; };
struct SolverConfig {
    StarterAlgorithm starter = StarterAlgorithm::GreedyBackward;
    std::vector<PhaseConfig> phases = {
        {1,1}, {1,2}, {2,1}, {2,2}, {2,3}, {3,2}, {3,3}, {3,4}, {4,3}
    };
};

// --- Combination generator (bounds-safe) -------------------------------------
std::vector<Combination>
generate_combinations(const std::vector<int> &elements, int k) {
    std::vector<Combination> combos;
    int n = elements.size();
    if (k <= 0 || k > n) return combos;
    std::vector<int> idx(k);
    std::iota(idx.begin(), idx.end(), 0);
    while (true) {
        Combination comb(k);
        for (int i = 0; i < k; ++i)
            comb[i] = elements[idx[i]];
        combos.push_back(std::move(comb));
        int i = k - 1;
        while (i >= 0 && idx[i] == n - k + i) --i;
        if (i < 0) break;
        ++idx[i];
        for (int j = i + 1; j < k; ++j)
            idx[j] = idx[j-1] + 1;
    }
    return combos;
}

// --- CUDA kernel: evaluate R×A trials in parallel ----------------------------
__global__
void evaluatePhaseKernel(
    const double *values,   // [N]
    const double *rem_sums, // [R]
    const int    *add_idxs, // [A*l]
    int           N,
    int           R,
    int           A,
    int           l,
    double        current_sum,
    double        target,
    double       *out_errors // [R*A]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= R * A) return;
    int rem_id = tid / A;
    int add_id = tid % A;
    double rem = rem_sums[rem_id];
    double add = 0.0;
    const int *aidx = add_idxs + add_id * l;
    for (int j = 0; j < l; ++j)
        add += values[aidx[j]];
    out_errors[tid] = fabs(target - (current_sum - rem + add));
}

// --- Refinement with GPU acceleration ---------------------------------------
SelectionMask
refine_solution(const ValueVector &values,
                double target,
                const SolverConfig &config)
{
    int N = values.size();

    // 1) Greedy start
    SelectionMask mask(N, 0);
    double current_sum = 0.0;
    {
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return values[a] > values[b]; });
        for (int i : order) {
            if (current_sum + values[i] <= target) {
                mask[i] = 1;
                current_sum += values[i];
            }
        }
    }
    double best_error = fabs(target - current_sum);

    std::cout << "\n--- Greedy start ---\n";
    print_error_report(best_error);
    print_solution_mask(mask);

    // Copy values[] to GPU once
    double *d_values = nullptr;
    cudaMalloc(&d_values, N * sizeof(double));
    cudaMemcpy(d_values, values.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // 2) Phases
    for (size_t p = 0; p < config.phases.size(); ++p) {
        int k = config.phases[p].k;
        int l = config.phases[p].l;
        std::cout << "\n--- Phase " << (p+1)
                  << " (Remove " << k << ", Add " << l << ") ---\n";

        // Build included/excluded index lists
        std::vector<int> included, excluded;
        included.reserve(N);
        excluded.reserve(N);
        for (int i = 0; i < N; ++i)
            (mask[i] ? included : excluded).push_back(i);
        if ((int)included.size() < k || (int)excluded.size() < l)
            continue;

        // Generate combos
        auto removal_cand = generate_combinations(included, k);
        auto addition_cand = generate_combinations(excluded, l);
        int R = removal_cand.size();
        int A = addition_cand.size();
        size_t trials = (size_t)R * A;
        if (trials == 0) continue;

        // Precompute removal sums
        std::vector<double> h_rem_sums(R);
        for (int i = 0; i < R; ++i)
            h_rem_sums[i] = std::accumulate(
                removal_cand[i].begin(),
                removal_cand[i].end(),
                0.0,
                [&](double s,int idx){ return s + values[idx]; }
            );

        // Flatten addition indices
        std::vector<int> h_add_idxs(A * l);
        for (int i = 0; i < A; ++i)
            for (int j = 0; j < l; ++j)
                h_add_idxs[i*l + j] = addition_cand[i][j];

        // Allocate & copy GPU buffers
        double *d_rem_sums = nullptr, *d_errors = nullptr;
        int    *d_add_idxs = nullptr;
        cudaMalloc(&d_rem_sums, R * sizeof(double));
        cudaMalloc(&d_add_idxs, A * l * sizeof(int));
        cudaMalloc(&d_errors,   trials * sizeof(double));

        cudaMemcpy(d_rem_sums, h_rem_sums.data(), R * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_add_idxs, h_add_idxs.data(),    A * l * sizeof(int), cudaMemcpyHostToDevice);

        // Launch kernel
        int TPB = 256;
        int blocks = (trials + TPB - 1) / TPB;
        evaluatePhaseKernel<<<blocks,TPB>>>(
            d_values, d_rem_sums, d_add_idxs,
            N, R, A, l,
            current_sum, target,
            d_errors
        );
        cudaDeviceSynchronize();

        // Copy back & find best
        std::vector<double> h_errors(trials);
        cudaMemcpy(h_errors.data(), d_errors, trials * sizeof(double), cudaMemcpyDeviceToHost);

        double phase_best = best_error;
        int    best_tid   = -1;
        for (int t = 0; t < (int)trials; ++t) {
            if (h_errors[t] < phase_best) {
                phase_best = h_errors[t];
                best_tid   = t;
            }
        }

        // Free GPU buffers
        cudaFree(d_rem_sums);
        cudaFree(d_add_idxs);
        cudaFree(d_errors);

        // On improvement, print and apply
        if (best_tid >= 0 && phase_best < best_error) {
            int rem_id = best_tid / A;
            int add_id = best_tid % A;
            for (int idx : removal_cand[rem_id]) mask[idx] = 0;
            for (int idx : addition_cand[add_id]) mask[idx] = 1;

            current_sum = 0.0;
            for (int i = 0; i < N; ++i)
                if (mask[i]) current_sum += values[i];

            best_error = phase_best;
            print_error_report(best_error);
            print_solution_mask(mask);

            --p; // re-run
        }
    }

    cudaFree(d_values);
    return mask;
}

// --- main() ---------------------------------------------------------------
int main() {
    const int N = 100;

    // Build M = { sqrt(p_i) | 1 <= i <= 100 }
    std::vector<int> primes;
    primes.reserve(N);
    for (int cand = 2; primes.size() < N; ++cand) {
        bool is_prime = true;
        for (int d = 2; d * d <= cand; ++d)
            if (cand % d == 0) { is_prime = false; break; }
        if (is_prime) primes.push_back(cand);
    }
    ValueVector values(N);
    for (int i = 0; i < N; ++i)
        values[i] = std::sqrt(primes[i]);

    // Target (unscaled)
    double target = 7 * 113596441 * 1e-6;

    std::cout << std::fixed << std::setprecision(6)
              << "Subset Sum Problem (N=" << N << ")\n"
              << "Target value: " << target << "\n";

    // Solve
    SolverConfig config;
    auto t0 = std::chrono::high_resolution_clock::now();
    SelectionMask solution = refine_solution(values, target, config);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Final sums & error
    double final_sum   = 0.0;
    for (int i = 0; i < N; ++i)
        if (solution[i]) final_sum += values[i];
    double final_error = fabs(target - final_sum);

    // Scaled difference and digits
    long double scaled_diff = (long double)final_error * powl(10.0L, 100);
    long double loge       = log10l((long double)final_error);
    long long digits       = (long long)std::floor(loge) + 1 + 100;

    std::cout << "\n=== Final Solution ===\n";
    print_error_report(final_error);
    print_solution_mask(solution);
    std::cout << "Scaled difference (×10^100): " << std::scientific << scaled_diff << "\n";
    std::cout << "Digits in difference: " << digits << "\n";

    // Performance
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    double elapsed_s = elapsed_ms.count() / 1000.0;
    std::cout << "\n=== Performance ===\n"
              << "Execution time: " << elapsed_s << " seconds ("
              << elapsed_ms.count() << " milliseconds)\n";

    return 0;
}
