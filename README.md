
# GPU-Accelerated Monte Carlo Pricer

A production-grade engine for pricing European options via Monte Carlo simulation on the NVIDIA Jetson Orin Nano. Demonstrates end-to-end quantitative modeling, GPU acceleration, and integration into Python workflows.

---

## Project Overview

- **Objective:**  
  Leverage the Orin Nano’s GPU to simulate millions of asset-price paths under the risk-neutral measure, compute discounted payoffs, and validate against the analytical Black-Scholes formula.
- **Key Outcomes:**  
  - High throughput (paths/sec) and low latency on embedded hardware  
  - Statistically rigorous estimates with confidence intervals  
  - Python bindings for easy integration into quant workflows  
  - Automated benchmarks and profiling reports

---

## Architecture & Technology Stack

- **Hardware:** NVIDIA Jetson Orin Nano (ARM64 + integrated GPU)  
- **Core Implementation:**  
  - CUDA /cuRAND kernels for path generation and reduction  
  - C++17 host code with modern build (CMake)  
- **Prototyping & Validation:** Python with NumPy, SciPy and Matplotlib  
- **Bindings & CLI:** pybind11 → Python extension + command-line interface  
- **Benchmarking & Profiling:** Shell/Python scripts + NVIDIA Nsight

---

## High-Level Phases

1. **Phase 1 – Prototyping**  
   Implement Black-Scholes formula and Monte Carlo sampler in Python. Validate accuracy, plot error convergence, compute 95% confidence intervals.
2. **Phase 2 – Design**  
   Analyze hot spots, draft GPU kernel layout (thread/block strategy, RNG, reduction, precision).
3. **Phase 3 – CUDA Implementation**  
   Write and test CUDA kernels (random-number generation, path simulation, payoff reduction) and C++ host orchestration.
4. **Phase 4 – Benchmarking & Tuning**  
   Measure throughput/latency, profile with Nsight, optimize grid/block sizes and memory usage, overlap compute and data transfer.
5. **Phase 5 – Python Bindings & CLI**  
   Expose GPU pricer via pybind11, provide a command-line utility for parameter sweeps and batch pricing.

---

## Prerequisites

- Jetson Orin Nano with JetPack (CUDA & cuDNN) installed  
- Ubuntu 20.04 (ARM64) or compatible OS  
- `build-essential`, `cmake`, `git`, `python3-pip`  
- Python packages: `numpy`, `scipy`, `matplotlib`, `pybind11`

---
