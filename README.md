# Pareto Simulator — Anti-Attention README

**Purpose:**

Optimizes computational parameter spaces by iteratively finding Pareto-efficient trade-offs between throughput and error.

## Quick Start

1. **Install:** `pip install numpy matplotlib`
2. **Run:** `python simulator.py`
3. **Observe:** Scatter plot of error vs. throughput.

## Configuration

Edit only these constants at the top of `simulator.py`:

```python
ITERATIONS = 5       # Feedback loops
N_SAMPLES  = 1000    # Data points
DIM_RANGE  = (64,512)# Dimension bounds
NOISE_MAX  = 0.1     # Error ceiling
```

**Our Python simulator:**

* Randomly generates “dimensions” and “noise,” modulated by a continuous manifold factor using `tanh(dim/200)`.
* Computes throughput as `throughput = dim / (1 + noise)` for each sample.
* Runs an iterative feedback loop: it finds the Pareto front (highest throughput with lowest error) and narrows the dimension range toward those optimal regions.
* Finally, it plots a scatter chart of error vs. throughput.

**Possible practical applications:**

* Parameter tuning in data or streaming pipelines (buffer sizes, batch sizes).
* QoS optimization in networks (latency vs. packet loss trade-off).
* Hyperparameter optimization in ML (accuracy vs. inference speed).
* Hardware benchmarking (compute cores vs. power consumption).
* IoT/edge system design (performance vs. signal noise).
* Compression or encoding analysis (bitrate vs. distortion).
