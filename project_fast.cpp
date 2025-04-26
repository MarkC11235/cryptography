#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

using namespace std;

long long sum_from_bin_list(const vector<int>& T, const vector<long long>& M) {
    if (T.size() != M.size()) {
        cerr << "Error: T and M must have the same size." << endl;
        exit(EXIT_FAILURE);
    }
    long long sum = 0;
    for (size_t i = 0; i < T.size(); ++i) {
        if (T[i]) {
            sum += M[i];
        }
    }
    return sum;
}

vector<int> subset_sum_backwards(const vector<long long>& M, long long S) {
    if (M.empty()) {
        cerr << "Error: Input vector M is empty." << endl;
        exit(EXIT_FAILURE);
    }
    vector<int> T(M.size(), 0);
    long long current_sum = 0;
    for (int i = M.size() - 1; i >= 0; --i) {
        if (current_sum + M[i] <= S) {
            T[i] = 1;
            current_sum += M[i];
        }
    }
    return T;
}

int main() {
    cout << "Starting program..." << endl;

    // Example input
    vector<long long> M = {100, 200, 300, 400, 500};
    long long S = 800;

    vector<pair<int, int>> k_l_pairs = {{1, 1}, {2, 1}, {1, 2}};
    int N = 3;

    cout << "Input vector M: ";
    for (auto m : M) {
        cout << m << " ";
    }
    cout << endl;

    vector<int> result = subset_sum_backwards(M, S);

    cout << "Final subset: ";
    for (size_t i = 0; i < result.size(); ++i) {
        if (result[i]) {
            cout << M[i] << " ";
        }
    }
    cout << endl;

    return 0;
}