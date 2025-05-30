import numpy as np
from scipy.stats import norm
import argparse

def bs_price(S0, K, r, sigma, T):
    """Black-Scholes closed-form price for a European call option."""
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price

@profile
def mc_price(S0, K, r, sigma, T, n_paths):
    """Monte Carlo price estimate for a European call option.
    Returns (price, standard_error).
    """
    if T <= 0:
        payoff = np.maximum(S0 - K, 0.0)
        return payoff, 0.0

    # Simulate end stock prices under risk-neutral measure
    z = np.random.randn(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    return price, std_error

def main():
    parser = argparse.ArgumentParser(description="Black-Scholes and Monte Carlo European call pricer")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial stock price")
    parser.add_argument("--K", type=float, default=100.0, help="Strike price")
    parser.add_argument("--r", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility")
    parser.add_argument("--T", type=float, default=1.0, help="Time to maturity (in years)")
    parser.add_argument("--n_paths", type=int, default=100000, help="Number of Monte Carlo paths")
    args = parser.parse_args()

    S0 = args.S0
    K = args.K
    r = args.r
    sigma = args.sigma
    T = args.T
    n_paths = args.n_paths

    bs = bs_price(S0, K, r, sigma, T)
    mc, mc_se = mc_price(S0, K, r, sigma, T, n_paths)
    conf_interval = (mc - 1.96 * mc_se, mc + 1.96 * mc_se)

    print(f"Black-Scholes price: {bs:.6f}")
    print(f"Monte Carlo price:   {mc:.6f} (95% CI: [{conf_interval[0]:.6f}, {conf_interval[1]:.6f}])")

if __name__ == "__main__":
    main()