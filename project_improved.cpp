/*
 * Subset Sum Solver
 * Implements configurable k/l-swap refinement with:
 * - Phase-focused iteration
 * - Pruning optimizations
 * - Multiple starting algorithms (greedy, randomized)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include <atomic>
#include <execution>

namespace SubsetSum
{
    using SelectionMask = std::vector<int>;
    using ValueVector = std::vector<double>;

    enum class StarterAlgorithm
    {
        GreedyBackward,
        UniformRandom,
        ValueWeightedRandom
    };

    struct PhaseConfig
    {
        int remove_elements;
        int add_elements;
    };

    struct SolverConfig
    {
        StarterAlgorithm starter = StarterAlgorithm::GreedyBackward;
        std::vector<PhaseConfig> phases = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
        bool verbose = true; 
    };

    double calculate_sum(const SelectionMask &mask, const ValueVector &values)
    {
        double total = 0;
        for (size_t i = 0; i < mask.size(); ++i)
            if (mask[i])
                total += values[i];
        return total;
    }

    std::vector<std::vector<int>> generate_combinations(const std::vector<int> &elements, int k)
    {
        std::vector<std::vector<int>> combinations;
        std::vector<int> combination(k);
        std::vector<int> indices(k);

        for (int i = 0; i < k; ++i)
            indices[i] = i;

        while (true)
        {
            for (int i = 0; i < k; ++i)
                combination[i] = elements[indices[i]];
            combinations.push_back(combination);

            int i;
            for (i = k - 1; i >= 0 && indices[i] == elements.size() - k + i; --i)
                ;
            if (i < 0)
                break;

            indices[i]++;
            for (int j = i + 1; j < k; ++j)
                indices[j] = indices[j - 1] + 1;
        }
        return combinations;
    }

    void show_progress(size_t current_step, size_t total_steps, bool verbose)
    {
        if (!verbose) return;
        constexpr int bar_width = 40;
        const double progress = static_cast<double>(current_step) / total_steps;
        const int filled = static_cast<int>(bar_width * progress);

        std::cout << "[";
        for (int i = 0; i < bar_width; ++i)
            std::cout << (i < filled ? '=' : (i == filled ? '>' : ' '));
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%\r";
        std::cout.flush();
    }

    void print_error_report(double error, bool verbose)
    {
        if (!verbose) return;
        std::cout << "Current error: " << std::scientific << error << "\n";
        if (error > 0)
        {
            const int precision_digits = static_cast<int>(100 + std::log10(error));
            std::cout << "Precision digits (10^100 scale): " << precision_digits << "\n";
        }
    }

    void print_solution_mask(const SelectionMask &mask, bool verbose)
    {
        if (!verbose) return;
        std::cout << "Solution mask: [";
        for (size_t i = 0; i < mask.size(); ++i)
            std::cout << mask[i] << (i + 1 < mask.size() ? ", " : "");
        std::cout << "]\n";
    }

    SelectionMask create_greedy_start(const ValueVector &values, double target)
    {
        SelectionMask mask(values.size(), 0);
        double current_sum = 0;

        for (int i = static_cast<int>(values.size()) - 1; i >= 0; --i)
        {
            if (current_sum + values[i] <= target)
            {
                mask[i] = 1;
                current_sum += values[i];
            }
        }
        return mask;
    }

    SelectionMask create_uniform_random_start(const ValueVector &values)
    {
        std::mt19937_64 rng(std::random_device{}());
        std::bernoulli_distribution dist(0.5);
        SelectionMask mask(values.size());

        for (size_t i = 0; i < values.size(); ++i)
            mask[i] = dist(rng);

        return mask;
    }

    SelectionMask create_weighted_random_start(const ValueVector &values, double target)
    {
        std::mt19937_64 rng(std::random_device{}());
        const double total = std::accumulate(values.begin(), values.end(), 0.0);
        const double inclusion_prob = (total > 0) ? std::clamp(target / total, 0.0, 1.0) : 0.5;
        std::bernoulli_distribution dist(inclusion_prob);
        SelectionMask mask(values.size());

        for (size_t i = 0; i < values.size(); ++i)
            mask[i] = dist(rng);

        return mask;
    }

    SelectionMask refine_solution(const ValueVector &values,
                                  double target,
                                  const SolverConfig &config)
    {
        SelectionMask current_mask;
        switch (config.starter)
        {
        case StarterAlgorithm::GreedyBackward:
            current_mask = create_greedy_start(values, target);
            break;
        case StarterAlgorithm::UniformRandom:
            current_mask = create_uniform_random_start(values);
            break;
        case StarterAlgorithm::ValueWeightedRandom:
            current_mask = create_weighted_random_start(values, target);
            break;
        }

        double current_sum = calculate_sum(current_mask, values);
        double best_error = std::abs(target - current_sum);

        if (config.verbose)
        {
            std::cout << "=== Starting Refinement Process ===\n";
            print_error_report(best_error, config.verbose);
            print_solution_mask(current_mask, config.verbose);
        }

        for (size_t phase_num = 0; phase_num < config.phases.size(); ++phase_num)
        {
            const auto [k, l] = config.phases[phase_num];
            std::vector<int> included, excluded;
            for (size_t i = 0; i < current_mask.size(); ++i)
                current_mask[i] ? included.push_back(i) : excluded.push_back(i);

            if (included.size() < static_cast<size_t>(k) || excluded.size() < static_cast<size_t>(l))
                continue;

            if (config.verbose)
                std::cout << "\n--- Phase " << (phase_num + 1) << " (Remove " << k << ", Add " << l << ") ---\n";

            const auto removal_candidates = generate_combinations(included, k);
            const auto addition_candidates = generate_combinations(excluded, l);
            std::vector<double> removal_sums;
            for (const auto &removal : removal_candidates)
                removal_sums.push_back(std::accumulate(removal.begin(), removal.end(), 0.0, [&](double s, int i) { return s + values[i]; }));

            double phase_best_error = best_error;
            std::vector<int> best_removal, best_addition;
            std::atomic<double> atomic_best_error(best_error);
            double best_removal_sum = 0, best_addition_sum = 0;

#pragma omp parallel for schedule(dynamic) collapse(1)
            for (size_t rem_idx = 0; rem_idx < removal_candidates.size(); ++rem_idx)
            {
                const auto &removal = removal_candidates[rem_idx];
                const double rem_sum = removal_sums[rem_idx];
                for (const auto &addition : addition_candidates)
                {
                    const double add_sum = std::accumulate(addition.begin(), addition.end(), 0.0, [&](double s, int i) { return s + values[i]; });
                    const double trial_error = std::abs(target - (current_sum - rem_sum + add_sum));

                    if (trial_error < atomic_best_error.load(std::memory_order_relaxed))
                    {
#pragma omp critical
                        if (trial_error < phase_best_error)
                        {
                            phase_best_error = trial_error;
                            best_removal = removal;
                            best_addition = addition;
                            best_removal_sum = rem_sum;
                            best_addition_sum = add_sum;
                            atomic_best_error.store(phase_best_error, std::memory_order_relaxed);
                        }
                    }
                }
            }

            if (phase_best_error < best_error)
            {
                if (config.verbose)
                {
                    std::cout << std::fixed << std::setprecision(6)
                              << "Phase " << (phase_num + 1) << " improvement: " << best_error << " → " << phase_best_error << "\n";
                    print_error_report(phase_best_error, config.verbose);
                    print_solution_mask(current_mask, config.verbose);
                }
                for (int idx : best_removal) current_mask[idx] = 0;
                for (int idx : best_addition) current_mask[idx] = 1;
                current_sum = current_sum - best_removal_sum + best_addition_sum;
                best_error = phase_best_error;
                phase_num--;
            }
        }
        return current_mask;
    }

    std::vector<int> generate_first_n_primes(int n)
    {
        std::vector<int> primes;
        int candidate = 2;
        while (primes.size() < static_cast<size_t>(n))
        {
            bool is_prime = true;
            for (int d = 2; d * d <= candidate; ++d)
                if (candidate % d == 0) { is_prime = false; break; }
            if (is_prime) primes.push_back(candidate);
            ++candidate;
        }
        return primes;
    }

    ValueVector compute_prime_sqrts(const std::vector<int> &primes)
    {
        ValueVector sqrts;
        sqrts.reserve(primes.size());
        for (int prime : primes)
            sqrts.push_back(std::sqrt(static_cast<double>(prime)));
        return sqrts;
    }

    constexpr double calculate_target_value()
    {
        return 7 * 113596441 * 1e-6;
    }
}

int main()
{
    using namespace SubsetSum;

    std::vector<std::vector<PhaseConfig>> all_phase_configs = {
        {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3}, {3, 4}, {4, 3}, {4, 4}},
    };

    SolverConfig config;
    config.starter = StarterAlgorithm::UniformRandom;
    config.verbose = false; 

    const auto primes = generate_first_n_primes(100);
    const ValueVector values = compute_prime_sqrts(primes);
    const double target = calculate_target_value();

    std::ofstream outfile("best_solutions.txt", std::ios::app);
    if (!outfile)
    {
        std::cerr << "Error opening output file.\n";
        return 1;
    }

    double best_error = std::numeric_limits<double>::infinity();
    SelectionMask best_solution;
    const int num_trials_per_config = 100000; 

    auto start_time = std::chrono::high_resolution_clock::now();

    for (const auto& phases : all_phase_configs)
    {
        config.phases = phases;
        std::cout << "=== Testing Phase Config: ";
        for (const auto& phase : phases)
            std::cout << "(" << phase.remove_elements << "," << phase.add_elements << ") ";
        std::cout << "===\n";
        for (int trial = 0; trial < num_trials_per_config; ++trial)
        {
            if (config.verbose)
                std::cout << "=== Trial " << (trial + 1) << " (Phase Config " << (&phases - &all_phase_configs[0] + 1) << ") ===\n";

            const SelectionMask solution = refine_solution(values, target, config);
            const double current_error = std::abs(target - calculate_sum(solution, values));

            if (current_error < best_error)
            {
                best_error = current_error;
                best_solution = solution;
                outfile << "Phase Config: ";
                for (const auto& phase : phases)
                    outfile << "(" << phase.remove_elements << "," << phase.add_elements << ") ";
                outfile << "\nError: " << std::scientific << best_error << "\nSolution Mask: [";
                for (size_t i = 0; i < best_solution.size(); ++i)
                    outfile << best_solution[i] << (i != best_solution.size() - 1 ? ", " : "");
                outfile << "]\n\n";
                outfile.flush();
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n=== Final Best Solution ===\n";
    std::cout << "Error: " << std::scientific << best_error << "\n";
    std::cout << "Time: " << elapsed.count() << " seconds\n";

    return 0;
}
