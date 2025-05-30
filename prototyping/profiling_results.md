# Monte Carlo Profiling Results

This document summarizes the performance profiling of the `mc_price` function in `prototyping/profile.py`, using `kernprof` (line-profiler). We ran 5 repeats for each sample size and measured both end-to-end runtimes and per-line hotspots.

---

## 1. End-to-End Timing

| Number of Paths $n$ | Avg Runtime (ms) |
|:---------------------:|:----------------:|
| 10 000                | 0.32             |
| 100 000               | 2.66             |
| 1 000 000             | 27.20            |
| 10 000 000            | 268.02           |
| 100 000 000           | 3055.68          |

- Runtime scales roughly linearly with $n$.  
- At $10^8$ paths, the pure-Python implementation takes over 3 seconds.

---

## 2. Per-Line Hotspots (n = various)

| Line                                   | Operation                                    | Total Time (μs) | % of Total |
|:--------------------------------------:|:---------------------------------------------|---------------:|-----------:|
| `z = np.random.randn(n_paths)`         | Generate standard normals                    | 8 472 017      | 51.2 %     |
| `ST = S0 * np.exp(... + sigma√T·z)`    | Compute terminal prices via vectorized exp    | 4 314 995      | 26.1 %     |
| `std_error = np.std(...)/√n_paths`     | Compute sample standard deviation & SE        | 1 508 222      | 9.1 %      |
| `payoffs = np.maximum(ST - K, 0.0)`     | Compute payoffs                               | 1 202 269      | 7.3 %      |
| `discounted_payoffs = exp(-rT)*payoffs` | Discount payoffs                              |   774 145      | 4.7 %      |
| `price = np.mean(discounted_payoffs)`  | Compute mean                                  |   271 875      | 1.6 %      |
| Other (overhead, branching)            | T ≤ 0 check, return, loop overhead            |      46        | 0.0 %      |

---

## 3. Analysis

1. **Random Number Generation (51%)**  
   `np.random.randn` is the dominant cost—over half of total runtime.  

2. **Vectorized Exponential (26%)**  
   Computing $\exp\bigl((r - 0.5\sigma^2)T + \sigma\sqrt T\,z\bigr)$ is the second largest contributor.  

3. **Statistics Computation (9% + 1.6%)**  
   - Standard deviation and standard-error calculation take ~9% of the time.  
   - Mean calculation is relatively cheap (~1.6%).  

4. **Payoff & Discount (7.3% + 4.7%)**  
   - The `max` operation and scalar multiplication by $e^{-rT}$ together account for ~12%.  

---

## 4. Next Steps

- **GPU RNG**: Offload random-number generation to **cuRAND** to eliminate the Python/NumPy RNG bottleneck.  
- **CUDA Kernel for Path Simulation**: Implement the `exp` + payoff logic in a single GPU kernel to combine the 26% + 12% costs.  
- **Parallel Reduction**: Replace `np.mean` and `np.std` with a two-stage GPU reduction (block-level + global) to collapse the ~11% spent on statistics.  

These optimizations will target the top 90% of runtime costs and lay the groundwork for a high-throughput GPU pricer.