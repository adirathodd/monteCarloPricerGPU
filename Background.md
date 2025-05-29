# Quantitative & Financial Foundations

---

## 1. Asset Dynamics under the Risk-Neutral Measure

We model the underlying asset price $S_t$ as a **Geometric Brownian Motion** under the _risk-neutral_ measure $Q$:

$$
dS_t = r\,S_t\,dt + \sigma\,S_t\,dW_t^Q
$$

- **$r$**: continuously compounded risk-free interest rate
- **$\sigma$**: annualized volatility of the asset
- **$W_t^Q$**: Brownian motion under the risk-neutral measure

By using $r$ (not the real-world drift) we ensure the _discounted price_ $e^{-rt}S_t$ is a martingale (no-arbitrage condition).

---

## 2. European Option Payoff

A **European call option** gives the right to buy the asset at strike $K$ on maturity $T$. Its payoff is:

$$
H_T = \max(S_T - K,\,0)
$$

A **European put** pays $\max(K - S_T,0)$. Because these depend _only_ on $S_T$, closed-form solutions exist under GBM.

---

## 3. Risk-Neutral Valuation

The no-arbitrage price at time 0 of any payoff $H_T$ is the _discounted expectation_ under $Q$:

$$
V_0 = e^{-rT}\,\mathbb{E}^Q\bigl[H_T\bigr]
$$

**Implementation plan**:

- **Simulate** paths $S_T^{(i)}$ under GBM with drift $r$.
- **Compute** each payoff $H_T^{(i)}$.
- **Discount**: $\tilde H^{(i)} = e^{-rT} H_T^{(i)}$.
- **Average**: $\hat V = \frac{1}{n}\sum_{i=1}^n \tilde H^{(i)}$.

---

## 4. Black-Scholes Closed-Form Benchmark

For a European call:

$$
\begin{aligned}
d_1 &= \frac{\ln(S_0/K) + (r + \frac12\sigma^2)\,T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T},\\
C &= S_0\,N(d_1) - K\,e^{-rT}\,N(d_2),
\end{aligned}
$$

- **$S_0$**: current spot price
- **$K$**: strike price
- **$N(\cdot)$**: standard normal CDF

Use $C$ as the “ground truth” to validate your Monte Carlo estimates.

---

## 5. Monte Carlo Pricing

- **Generate** $n$ samples $Z_i \sim \mathcal{N}(0,1)$.
- **Compute** terminal prices:
$$
S_T^{(i)} = S_0 \exp\bigl((r - \tfrac12\sigma^2)\,T + \sigma\sqrt{T}\,Z_i\bigr).
$$
- **Payoff**: $\pi_i = \max(S_T^{(i)} - K, 0)$.
- **Discount**: $\tilde \pi_i = e^{-rT}\,\pi_i$.
- **Estimate** price: $\hat V = \frac{1}{n}\sum \tilde \pi_i$.

---

## 6. Statistical Error & Convergence

- **Sample variance**:
$$
s^2 = \frac{1}{n-1}\sum(\tilde \pi_i - \hat V)^2
$$
- **Standard error**:
$$
\mathrm{SE} = \frac{s}{\sqrt{n}}
$$
- **95% confidence interval**:
$$
\bigl[\hat V - 1.96\,\mathrm{SE},\; \hat V + 1.96\,\mathrm{SE}\bigr]
$$
- **Convergence check**: plot $\log|\hat V - C|$ vs. $\log n$; slope ≈ –0.5 confirms $O(n^{-1/2})$ error decay.

---