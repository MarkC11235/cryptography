#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <random>
#include <string>
#include <functional> // Add this header for std::function

using namespace std;

// Calculate sum from binary list
double sum_from_bin_list(const vector<int>& T, const vector<double>& M) {
    double sum = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        if (T[i]) {
            sum += M[i];
        }
    }
    return sum;
}

// Function to generate combinations
void generate_combinations(vector<int>& current, int start, int n, int k, 
                          vector<vector<int>>& result, const vector<int>& indices) {
    if (k == 0) {
        result.push_back(current);
        return;
    }
    
    for (int i = start; i <= n - k; ++i) {
        current.push_back(indices[i]);
        generate_combinations(current, i + 1, n, k - 1, result, indices);
        current.pop_back();
    }
}

// Function to display progress bar
void display_progress_bar(long step, long total_steps, int barWidth = 40) {
    int pct = int((step * 100) / total_steps);
    int pos = barWidth * pct / 100;
    
    cout << "\r[";
    for (int b = 0; b < barWidth; ++b) {
        if (b < pos) cout << "=";
        else if (b == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << setw(3) << pct << "% "
         << "(" << step << "/" << total_steps << ")"
         << flush;
}

// Modified to accept std::function instead of function pointer
vector<int> subset_sum_progressive(const vector<double>& M, 
                                double S, 
                                function<vector<int>(const vector<double>&, double)> subset_sum_function) {
    // Initial subset
    vector<int> T0 = subset_sum_function(M, S);
    double sum0 = sum_from_bin_list(T0, M);
    double err = fabs(S - sum0);
    
    cout << "Initial solution error: " << scientific << err << endl;
    cout << "Target sum: " << S << endl;
    cout << "Initial sum: " << sum0 << endl;
    
    // Progressive refinement strategy with different phases
    vector<vector<pair<int, int>>> phases = {
        // Phase 1: Fine-tuning with small swaps
        {{1, 1}, {1, 2}, {2, 1}},
        
        // Phase 2: Medium adjustments
        {{2, 2}, {2, 3}, {3, 2}},
        
        // Phase 3: Larger structural changes
        {{3, 3}, {3, 4}, {4, 3}},

        // Phase 4: Larger structural changes
        {{4, 4}, {4, 5}, {5, 4}},
        
        // Phase 5: Final adjustments
        {{1, 1}, {1, 2}, {2, 1}}
    };
    
    for (size_t phase = 0; phase < phases.size(); phase++) {
        cout << "\n=== Phase " << phase + 1 << " ===\n";
        
        for (const auto& k_l_pair : phases[phase]) {
            int k = k_l_pair.first;
            int l = k_l_pair.second;
            
            cout << "\nTrying k=" << k << ", l=" << l << endl;
            
            bool improve = true;
            int iteration = 0;
            
            while (improve && iteration < 5) { // Limit iterations per k,l pair
                iteration++;
                improve = false;
                
                // Build remove/add index lists
                vector<int> remove_indices, add_indices;
                for (size_t i = 0; i < T0.size(); ++i) {
                    if (T0[i]) {
                        remove_indices.push_back(i);
                    } else {
                        add_indices.push_back(i);
                    }
                }
                
                if (k > (int)remove_indices.size() || l > (int)add_indices.size()) {
                    cout << "Skipping: not enough elements to remove/add\n";
                    break;
                }
                
                // Precompute all combinations
                vector<vector<int>> remove_combinations, add_combinations;
                vector<int> tmp;
                generate_combinations(tmp, 0, remove_indices.size(), k, 
                                     remove_combinations, remove_indices);
                tmp.clear();
                generate_combinations(tmp, 0, add_indices.size(), l, 
                                     add_combinations, add_indices);
                
                const long total_remove = remove_combinations.size();
                const long total_add = add_combinations.size();
                const long total_steps = total_remove * total_add;
                
                cout << "Checking " << total_steps << " combinations...\n";
                
                // Track best improvement
                double best_err = err;
                vector<int> best_remove, best_add;
                
                long step = 0;
                for (int ri = 0; ri < (int)total_remove && !improve; ++ri) {
                    for (int ai = 0; ai < (int)total_add; ++ai) {
                        ++step;
                        
                        // Update progress bar every 1000 steps or for last step
                        if (step % 1000 == 0 || step == total_steps) {
                            display_progress_bar(step, total_steps);
                        }
                        
                        // Test this remove/add pair
                        double new_sum = sum0;
                        for (int r : remove_combinations[ri]) {
                            new_sum -= M[r];
                        }
                        for (int a : add_combinations[ai]) {
                            new_sum += M[a];
                        }
                        
                        double new_err = fabs(S - new_sum);
                        
                        // Check if this is better than our best so far
                        if (new_err < best_err) {
                            best_err = new_err;
                            best_remove = remove_combinations[ri];
                            best_add = add_combinations[ai];
                            
                            // If it's significantly better, stop early
                            if (best_err < err * 0.5) {
                                improve = true;
                                break;
                            }
                        }
                    }
                }
                
                cout << endl; // End the progress bar line
                
                // Apply the best improvement found (if any)
                if (best_err < err) {
                    // Clear progress bar line
                    cout << "\rImprovement found: " << scientific << err << " → " << best_err << endl;
                    
                    // Apply changes
                    for (int r : best_remove) {
                        T0[r] = 0;
                    }
                    for (int a : best_add) {
                        T0[a] = 1;
                    }
                    
                    sum0 = sum_from_bin_list(T0, M);  // Recalculate for numerical stability
                    err = best_err;
                    improve = true;
                } else {
                    cout << "No improvement found for k=" << k << ", l=" << l << endl;
                }
            }
        }
    }
    
    return T0;
}

// Original subset sum function (backwards greedy approach)
vector<int> subset_sum_backwards(const vector<double>& M, double S) {
    vector<int> T(M.size(), 0);
    double current_sum = 0;
    
    for (int i = M.size() - 1; i >= 0; --i) {
        if (current_sum + M[i] <= S) {
            T[i] = 1;
            current_sum += M[i];
        }
    }
    
    return T;
}

// Forward greedy approach (alternative starting point)
vector<int> subset_sum_forwards(const vector<double>& M, double S) {
    vector<int> T(M.size(), 0);
    double current_sum = 0;
    
    for (int i = 0; i < M.size(); ++i) {
        if (current_sum + M[i] <= S) {
            T[i] = 1;
            current_sum += M[i];
        }
    }
    
    return T;
}

// Value ratio based approach (tries to maximize value per element)
vector<int> subset_sum_value_ratio(const vector<double>& M, double S) {
    vector<pair<double, int>> value_ratio;
    for (int i = 0; i < M.size(); i++) {
        value_ratio.push_back({M[i], i});
    }
    
    // Sort by value (ascending)
    sort(value_ratio.begin(), value_ratio.end());
    
    vector<int> T(M.size(), 0);
    double current_sum = 0;
    
    // Try to add items in order of increasing value
    for (const auto& item : value_ratio) {
        if (current_sum + M[item.second] <= S) {
            T[item.second] = 1;
            current_sum += M[item.second];
        }
    }
    
    return T;
}

int main() {
    // Generate first 100 primes
    vector<int> primes;
    int num = 2;
    while (primes.size() < 100) {
        bool is_prime = true;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                is_prime = false;
                break;
            }
        }
        if (is_prime) {
            primes.push_back(num);
        }
        num++;
    }
    
    // Create M = {sqrt(p_i) | 1 <= i <= 100 where p_i is the ith prime}
    vector<double> M(100);
    for (int i = 0; i < 100; ++i) {
        M[i] = sqrt(static_cast<double>(primes[i]));
    }
    
    // Target sum without 10^100 factor
    double S = 7 * 113596441 * pow(10, -6);
    
    cout << "=================================================\n";
    cout << "Subset Sum Progressive Refinement Algorithm\n";
    cout << "=================================================\n";
    cout << "Target sum: " << S << "\n";
    cout << "Number of elements: " << M.size() << "\n\n";
    
    auto start = chrono::high_resolution_clock::now();
    
    // Try multiple starting points and choose the best result
    cout << "Testing different starting algorithms...\n\n";
    
    vector<vector<int>> starting_solutions = {
        subset_sum_backwards(M, S),
        subset_sum_forwards(M, S),
        subset_sum_value_ratio(M, S)
    };
    
    vector<string> algorithm_names = {
        "Backwards Greedy",
        "Forwards Greedy",
        "Value Ratio Approach"
    };
    
    vector<int> best_solution;
    double best_error = numeric_limits<double>::max();
    int best_algo_index = -1;
    
    // Find the best starting solution
    for (size_t i = 0; i < starting_solutions.size(); i++) {
        double sum = sum_from_bin_list(starting_solutions[i], M);
        double error = fabs(S - sum);
        
        cout << algorithm_names[i] << ":\n";
        cout << "  Sum: " << sum << "\n";
        cout << "  Error: " << scientific << error << "\n\n";
        
        if (error < best_error) {
            best_error = error;
            best_solution = starting_solutions[i];
            best_algo_index = i;
        }
    }
    
    cout << "Using best starting solution: " << algorithm_names[best_algo_index] << "\n";
    cout << "Initial error: " << scientific << best_error << "\n\n";
    
    // Apply progressive refinement to the best starting solution
    vector<int> result = subset_sum_progressive(M, S, 
        [&best_solution](const vector<double>& M, double S) {
            return best_solution;
        });
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    
    cout << "\n=================================================\n";
    cout << "Results:\n";
    cout << "=================================================\n";
    
    double final_sum = sum_from_bin_list(result, M);
    double final_error = fabs(S - final_sum);
    
    cout << "Elements in subset: " << count(result.begin(), result.end(), 1) << "/" << M.size() << endl;
    cout << "Final sum (unscaled): " << final_sum << endl;
    cout << "Target sum (unscaled): " << S << endl;
    cout << "Absolute difference: " << final_error << endl;
    cout << "Relative error: " << (final_error / S) * 100 << "%" << endl;

    // Adding scaling factor back for final report
    cout << "\nWith 10^100 scaling factor:" << endl;
    cout << "Final sum: " << final_sum << " × 10^100" << endl;
    cout << "Target sum: " << S << " × 10^100" << endl;
    cout << "Difference: " << final_error << " × 10^100" << endl;
        
    // Calculate digit precision considering the 10^100 scaling factor
    int precision_digits = 0;
    if (final_error > 0) {
        // For a number scaled by 10^100, we need to consider how many digits after
        // the most significant digit are correctly matched
        precision_digits = static_cast<int>(100 + log10(final_error));
    }
        
    cout << "Matched to approximately " << precision_digits << " digits of precision" << endl;
    cout << "Time taken: " << duration.count() << " seconds" << endl;
    
    // Display elements in the final subset
    cout << "\nFinal subset elements:" << endl;
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i]) {
            cout << "  M[" << i << "] = " << M[i] << " (sqrt of " << primes[i] << ")" << endl;
        }
    }
    
    return 0;
}
