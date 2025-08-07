# app.py

from fastapi import FastAPI, Query
import numpy as np
import requests
import random
import asyncio
from typing import Callable, Dict, List
import uvicorn

app = FastAPI(title="Pareto Simulator Service")

# --- plugin architecture: manifold functions ---
def tanh_manifold(dims: np.ndarray) -> np.ndarray:
    return np.tanh(dims / 200.0)

def graph_laplacian_manifold(dims: np.ndarray) -> np.ndarray:
    μ, σ = np.mean(dims), np.std(dims) + 1e-6
    return np.exp(-((dims - μ)**2) / (2 * σ**2))

MANIFOLDS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": tanh_manifold,
    "graph": graph_laplacian_manifold,
    # add more here...
}

# --- real-world continuum data ingestion (Coingecko free API) ---
def fetch_continuum_data(coin: str = "bitcoin", days: int = 1) -> List[float]:
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        f"?vs_currency=usd&days={days}&interval=hourly"
    )
    r = requests.get(url, timeout=5)
    data = r.json().get("prices", [])
    # extract just the price values
    return [p[1] for p in data]

# --- core simulation & optimization ---
def simulate(
    n_samples: int,
    dim_min: float,
    dim_max: float,
    noise_max: float,
    manifold_fn: Callable[[np.ndarray], np.ndarray]
):
    dims = np.random.uniform(dim_min, dim_max, n_samples)
    base_noise = np.random.uniform(0, noise_max, n_samples)
    mf = manifold_fn(dims)
    noise = base_noise * mf
    throughput = dims / (1 + noise)
    return dims, throughput, noise

def optimize(
    iterations: int,
    n_samples: int,
    dim_min: float,
    dim_max: float,
    noise_max: float,
    manifold_fn: Callable[[np.ndarray], np.ndarray],
):
    low, high = dim_min, dim_max
    for _ in range(iterations):
        dims, thr, err = simulate(n_samples, low, high, noise_max, manifold_fn)
        idxs = np.argsort(-thr)
        best_err = float("inf")
        pareto = []
        for idx in idxs:
            if err[idx] < best_err:
                pareto.append(dims[idx])
                best_err = err[idx]
        if pareto:
            low, high = max(dim_min, min(pareto)), min(dim_max, max(pareto))
    return {"dims": dims.tolist(), "throughput": thr.tolist(), "error": err.tolist()}

# --- lightweight evolutionary hyperparam tuning ---
async def evolutionary_tune(
    pop_size: int,
    generations: int,
    base_params: dict,
    manifold_key: str
):
    population = [base_params["noise_max"] * random.uniform(0.5, 1.5)
                  for _ in range(pop_size)]
    best = {"noise_max": base_params["noise_max"], "score": -1}
    for _ in range(generations):
        scored = []
        for noise in population:
            res = optimize(
                iterations=base_params["iterations"],
                n_samples=base_params["n_samples"],
                dim_min=base_params["dim_min"],
                dim_max=base_params["dim_max"],
                noise_max=noise,
                manifold_fn=MANIFOLDS[manifold_key],
            )
            avg_throughput = np.mean(res["throughput"])
            avg_error = np.mean(res["error"])
            score = avg_throughput / (1 + avg_error)
            scored.append((noise, score))
            if score > best["score"]:
                best = {"noise_max": noise, "score": score}
        # breed next gen
        scored.sort(key=lambda x: -x[1])
        top = [n for n, _ in scored[: pop_size // 2]]
        population = [n + random.uniform(-0.01, 0.01) for n in top for _ in (0, 1)]
        await asyncio.sleep(0)  # cooperative multitasking
    return best

# --- endpoints ---
@app.get("/simulate")
def simulate_endpoint(
    iterations: int = Query(5, ge=1, le=20),
    n_samples: int = Query(1000, ge=100, le=10000),
    dim_min: float = 64.0,
    dim_max: float = 512.0,
    noise_max: float = 0.1,
    manifold: str = Query("tanh", enum=list(MANIFOLDS.keys())),
    real_data: bool = False
):
    if real_data:
        continuum = fetch_continuum_data()
        noise_max = max(noise_max, np.std(continuum) / max(continuum))
    return optimize(iterations, n_samples, dim_min, dim_max, noise_max, MANIFOLDS[manifold])

@app.get("/tune")
async def tune_endpoint(
    generations: int = Query(5, ge=1, le=50),
    pop_size: int = Query(10, ge=4, le=100),
    iterations: int = 5,
    n_samples: int = 1000,
    dim_min: float = 64.0,
    dim_max: float = 512.0,
    manifold: str = Query("tanh", enum=list(MANIFOLDS.keys()))
):
    base = {
        "iterations": iterations,
        "n_samples": n_samples,
        "dim_min": dim_min,
        "dim_max": dim_max,
        "noise_max": 0.1,
    }
    best = await evolutionary_tune(pop_size, generations, base, manifold)
    return {"best_noise_max": best["noise_max"], "score": best["score"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
