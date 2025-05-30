import time
from prototype import mc_price

def time_mc(n_paths, repeats=5):
    """Measure average runtime of mc_price for given n_paths."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ , _ = mc_price(S0, K, r, sigma, T, n_paths)
        end = time.perf_counter()
        times.append(end - start)
    avg = sum(times) / len(times)
    print(f"n={n_paths:8,d} → {avg*1e3:8.2f} ms (avg over {repeats})")

if __name__ == "__main__":
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    for n in [10**4, 10**5, 10**6, 10**7, 10**8]:
        time_mc(n)