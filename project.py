from decimal import Decimal, getcontext
import math
from typing import Callable, List
getcontext().prec = 110 # this allows for 110 decimal places when doing the sqrt operation

# bad
def subset_sum_backwards(M: list[int], S: int) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    """
    T = [0] * len(M) # binary list to show if the element from M is in the subset

    # Start at the end of the list and work backwards
    # only add elements to the subset if they don't make the sum exceed S
    for i in range(len(M) - 1, -1, -1):
        if sum(M[j] for j in range(len(M)) if T[j]) + M[i] <= S:
            T[i] = 1 # include this element in the subset

    return T

# bad
def subset_sum_forwards(M: list[int], S: int) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    """
    T = [0] * len(M) # binary list to show if the element from M is in the subset

    # Start at the beginning of the list and work forwards
    # only add elements to the subset if they don't make the sum exceed S
    for i in range(len(M)):
        if sum(M[j] for j in range(len(M)) if T[j]) + M[i] <= S:
            T[i] = 1 # include this element in the subset

    return T

# worse
def subset_sum_alternate(M: list[int], S: int) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    """
    T = [0] * len(M) # binary list to show if the element from M is in the subset

    # Choose the first ele, then the last, then the second, then the second to last, etc.
    # This is a bit of a hack, but it works for this problem.
    for i in range(len(M) // 2):
        if sum(M[j] for j in range(len(M)) if T[j]) + M[i] <= S:
            T[i] = 1
        if sum(M[j] for j in range(len(M)) if T[j]) + M[len(M) - 1 - i] <= S:
            T[len(M) - 1 - i] = 1

    return T

# too slow
def subset_sum_brute_force(M: list[int], S: int) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    """
    T = [0] * len(M) # binary list to show if the element from M is in the subset

    # Check all possible subsets
    print(f"Checking {1 << len(M)} possible subsets...")
    for i in range(1 << len(M)):
        if i % 1000000 == 0:
            print(f"Checking subset {i}...")
        current_sum = sum(M[j] for j in range(len(M)) if (i & (1 << j)) > 0)
        if current_sum <= S:
            T = [(i & (1 << j)) > 0 for j in range(len(M))]

    return T

# worse
def subset_sum_recursive(M: list[int], S: int, index: int, T: list[int]) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    """
    if index == len(M):
        return T

    # Include the current element in the subset
    T[index] = 1
    if sum_from_bin_list(T, M) <= S:
        return subset_sum_recursive(M, S, index + 1, T)

    # Exclude the current element from the subset
    T[index] = 0
    return subset_sum_recursive(M, S, index + 1, T)

def subset_sum_refine(M: list[int], S: int, subset_sum_function: Callable) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    Calls a different function to start this process, then refines the result.
    """

    T0 = subset_sum_function(M, S)
    sum0 = sum_from_bin_list(T0, M)
    err = abs(S - sum0)
    # Refine the result with the hill climbing algorithm
    improve = True
    while improve:
        improve = False
        # foreach M[i] == 1: remove it and try to add M[j] == 0
        for i in range(len(M)):
            if T0[i] == 1:
                for j in range(len(M)):
                    if T0[j] == 0:
                        # remove M[i] and add M[j]
                        new_sum = sum0 - M[i] + M[j]
                        new_err = abs(S - new_sum)
                        if new_err < err:
                            # update the subset and the error
                            print(f"Improving: removing M[{i}] = {M[i]} and adding M[{j}] = {M[j]}")
                            T0[i] = 0
                            T0[j] = 1
                            sum0 = new_sum
                            err = new_err
                            improve = True
                            break
                if improve:
                    break

    return T0

def subset_sum_refine_2(M: list[int], S: int, subset_sum_function: Callable, k=2, l=3) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    Calls a different function to start this process, then refines the result.
    k: Number of elements to remove on each iteration.
    l: Number of elements to add back on each iteration.
    """

    print(f"Refining with k = {k} and l = {l}")
    T0 = subset_sum_function(M, S)
    sum0 = sum_from_bin_list(T0, M)
    err = abs(S - sum0)
    # Refine the result with the hill climbing algorithm
    improve = True
    while improve:
        improve = False
        # Generate all combinations of k elements to remove
        remove_indices = [i for i in range(len(M)) if T0[i] == 1]
        add_indices = [i for i in range(len(M)) if T0[i] == 0]

        from itertools import combinations

        for remove_comb in combinations(remove_indices, k):
            for add_comb in combinations(add_indices, l):
                # Calculate the new sum after removing and adding elements
                new_sum = sum0
                for r in remove_comb:
                    new_sum -= M[r]
                for a in add_comb:
                    new_sum += M[a]

                new_err = abs(S - new_sum)
                if new_err < err:
                    # Update the subset and the error
                    print(f"Improving: removing {remove_comb} and adding {add_comb}")
                    for r in remove_comb:
                        T0[r] = 0
                    for a in add_comb:
                        T0[a] = 1
                    sum0 = new_sum
                    err = new_err
                    improve = True
                    break
            if improve:
                break

    return T0

def subset_sum_refine_3(M: list[int], S: int, subset_sum_function: Callable, k_l_pairs: List[tuple], N=None) -> list[int]:
    """
    M is a list (length 100) of integers and S is the target sum.
    Return the subset that sums the closest to S.
    Calls a different function to start this process, then refines the result.
    k_l_pairs: A list of (k, l) pairs to use for each iteration.
    N: Number of iterations (optional). If not provided, defaults to the length of k_l_pairs.
    """

    if N is None:
        N = len(k_l_pairs)

    print(f"Refining with {N} iterations and varying k, l pairs", flush=True)
    T0 = subset_sum_function(M, S)
    sum0 = sum_from_bin_list(T0, M)
    err = abs(S - sum0)

    for i in range(N):
        k, l = k_l_pairs[i % len(k_l_pairs)]  # Cycle through k_l_pairs if N > len(k_l_pairs)
        print(f"Iteration {i + 1} of {N}: Using k = {k}, l = {l}", flush=True)
        improve = True
        while improve:
            improve = False
            # Generate all combinations of k elements to remove
            remove_indices = [i for i in range(len(M)) if T0[i] == 1]
            add_indices = [i for i in range(len(M)) if T0[i] == 0]

            from itertools import combinations

            for remove_comb in combinations(remove_indices, k):
                for add_comb in combinations(add_indices, l):
                    # Calculate the new sum after removing and adding elements
                    new_sum = sum0
                    for r in remove_comb:
                        new_sum -= M[r]
                    for a in add_comb:
                        new_sum += M[a]

                    new_err = abs(S - new_sum)
                    if new_err < err:
                        # Update the subset and the error
                        print(f"Improving: removing {remove_comb} and adding {add_comb}", flush=True)
                        for r in remove_comb:
                            T0[r] = 0
                        for a in add_comb:
                            T0[a] = 1
                        sum0 = new_sum
                        err = new_err
                        improve = True
                        break
                if improve:
                    break

    return T0

def dp_subset_sum(M, S):
    dp = {0: []}  # Dictionary to store sums and their corresponding subsets
    for num in M:
        for current_sum, subset in list(dp.items()):
            new_sum = current_sum + num
            if new_sum <= S and new_sum not in dp:
                dp[new_sum] = subset + [num]
    closest_sum = max(dp.keys())
    return dp[closest_sum]

def subset_sum_split_recursive(
    M: list[int], 
    S: int, 
    depth: int = 0, 
    subset_function: Callable[[list[int], int], list[int]] = dp_subset_sum,
    max_depth: int = 2,
    num_splits: int = 2,
    corrector_function: Callable[[list[int], int, Callable], list[int]] = subset_sum_refine
) -> list[int]:
    """
    Recursively split the list M into smaller subsets down to a specified depth.
    At the base case, apply the provided subset_function to calculate the subset.
    After combining the results, apply a corrector function to refine the subset.

    Args:
        M: List of integers.
        S: Target sum.
        depth: Depth of recursion.
        subset_function: Function to calculate the subset for smaller lists.
        max_depth: Maximum depth of recursion.
        num_splits: Number of splits at each level.
        corrector_function: Function to refine the combined subset.

    Returns:
        A binary list indicating the subset that sums closest to S.
    """
    print(f"Depth: {depth}, M: {M}, S: {S}")

    if not M:
        return []
    if len(M) == 1:
        return [1] if M[0] <= S else [0]
    if depth == max_depth:
        return subset_function(M, S)

    # Split M into `num_splits` parts
    chunk_size = (len(M) + num_splits - 1) // num_splits  # Calculate chunk size
    chunks = [M[i * chunk_size:(i + 1) * chunk_size] for i in range(num_splits)]

    # Split the target sum approximately evenly among the chunks
    target_splits = [S // num_splits] * num_splits
    for i in range(S % num_splits):
        target_splits[i] += 1  # Distribute the remainder

    # Recursively calculate subsets for each chunk
    subsets = [
        subset_sum_split_recursive(chunks[i], target_splits[i], depth + 1, subset_function, max_depth, num_splits, corrector_function)
        for i in range(len(chunks))
    ]

    # Combine the results into a single binary list
    T = [0] * len(M)
    offset = 0
    for i, subset in enumerate(subsets):
        for j, value in enumerate(subset):
            T[offset + j] = value
        offset += len(chunks[i])

    # Apply the corrector function to refine the combined subset
    T = corrector_function(M, S, lambda m, s: T)

    return T
    

def is_prime(n: int) -> bool:
    """
    Check if n is prime.
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def sum_from_bin_list(T: list[int], M: list[int]) -> int:
    """
    Given a binary list T and a list M, return the sum of the elements in M
    that are included in the subset indicated by T.
    """
    return sum(M[i] for i in range(len(M)) if T[i])

def timer(func: Callable) -> Callable:
    """
    Decorator to time a function.
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timer
def main(subset_sum_function=subset_sum_backwards, ID=113596441, extra_args=[]):
    print(f"Using {subset_sum_function.__name__} with ID {ID}")
    print()
    # generate the first 100 prime numbers
    primes = []
    i = 2
    while len(primes) < 100:
        if is_prime(i):
            primes.append(i)
        i += 1

    # print(len(primes))
    # print(primes)
    # print()

    # M = [10**100 * i**0.5 for i in primes]
    M = [math.floor(Decimal(i).sqrt() * Decimal(10**100)) for i in primes]
    # print(len(M))
    # print(M)

    S = 7 * ID * 10**94

    print("Number of digits in the target sum:", len(str(int(S))))
    print("Target sum:", S)
    print()

    exp_target = sum_from_bin_list(subset_sum_function(M, S, *extra_args), M)
    print("Number of digits in the subset sum:", len(str(int(exp_target))))
    print("Subset sum:", exp_target)
    print()

    print("Number of digits in the difference:", len(str(int(S - exp_target))))
    print("Difference:", S - exp_target) 
    print()


ID = 113596441
# main(subset_sum_function=subset_sum_backwards, ID=ID, extra_args=[])
# main(subset_sum_function=subset_sum_forwards, ID=ID, extra_args=[])
# main(subset_sum_function=subset_sum_alternate, ID=ID, extra_args=[])
# main(subset_sum_function=subset_sum_brute_force, ID=ID, extra_args=[])
# main(subset_sum_function=subset_sum_recursive, ID=ID, extra_args=[0, [0] * 100])
# main(subset_sum_function=subset_sum_refine, ID=ID, extra_args=[subset_sum_backwards])


# subset_sum_refine_2 with subset_sum_backwards as start and k = 2, l = 3, makes the error 94 digits 
# main(subset_sum_function=subset_sum_refine_2, ID=ID, extra_args=[subset_sum_backwards, 2, 3])

# subset_sum_refine_3 with subset_sum_backwards as start and k = 2, l = 3, N = 3 makes the error 94 digits 
main(subset_sum_function=subset_sum_refine_3, ID=ID, extra_args=[subset_sum_alternate, [(1, 1), (1, 2), (2, 3), (3, 3), (3, 2), (2, 1), (1, 1)]])


# main(subset_sum_function=subset_sum_split_recursive, ID=ID, extra_args=[subset_sum_backwards, 2, 3, subset_sum_refine_3, [(1, 1), (1, 2), (2, 3)], 3])