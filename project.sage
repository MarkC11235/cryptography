# Load all of Sage’s symbolic and numeric environment
# from sage.all import Matrix, ZZ, vector, primes, floor, sqrt

# 1) Parameters
n = 100
# First 100 primes
plist = list(primes(2, 2*n))[:n]                             # :contentReference[oaicite:0]{index=0}
# Scaled weights and target (scaled down by 10^94)
M = [floor(10^6 * sqrt(p)) for p in plist]                   # :contentReference[oaicite:1]{index=1}
S = 7 * 113596441                                            # :contentReference[oaicite:2]{index=2}

# 2) Embedding scale (power of two ≥ max(M) for good conditioning)
N = 1 << max(M).bit_length()                                 # :contentReference[oaicite:3]{index=3}

# 3) Build lattice basis: identity + weight column, and the target row
B = Matrix(ZZ, n+1, n+1)
for i in range(n):
    B[i,i] = 1
    B[i,n] = N * M[i]                                        # :contentReference[oaicite:4]{index=4}
B[n] = [0]*n + [-N * S]                                      # :contentReference[oaicite:5]{index=5}

# 4) Lattice reduction: choose either LLL or BKZ
B_lll  = B.LLL(delta=0.99)                                   # :contentReference[oaicite:6]{index=6}
B_bkz  = B.BKZ(block_size=30, proof=False)                   # :contentReference[oaicite:7]{index=7}

# 5) CVP: find the closest lattice vector to the embedding of -N⋅S
tvec = vector(ZZ, [B_bkz[i,n] for i in range(n+1)])
v    = B_bkz.closest_vector(tvec)                           # :contentReference[oaicite:8]{index=8}

# 6) Extract 0/1 subset from the first n coordinates
x = [1 if c > 0 else 0 for c in v[:n]]
chosen = [i for i, xi in enumerate(x) if xi]

# 7) Compute achieved sum and difference
subset_sum = sum(M[i] for i in chosen)
diff       = subset_sum - S

# 8) Output
print("Chosen indices:", chosen)
print("Subset sum =", subset_sum)
print("Target      =", S)
print("Difference  =", diff)
