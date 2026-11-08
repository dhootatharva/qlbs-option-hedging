"""Closed-form Black-Scholes prices and deltas, plus a risk-neutral Monte Carlo pricer."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, tau):
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return d1, d1 - sigma * np.sqrt(tau)


def bs_price(S, K, r, sigma, tau, kind: str = "put"):
    """Black-Scholes price of a European option with time to maturity ``tau``."""
    d1, d2 = _d1_d2(S, K, r, sigma, tau)
    if kind == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    if kind == "put":
        return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError(f"unknown option kind: {kind}")


def bs_delta(S, K, r, sigma, tau, kind: str = "put"):
    """Black-Scholes delta (number of shares held by the hedger)."""
    d1, _ = _d1_d2(S, K, r, sigma, tau)
    return norm.cdf(d1) - (1.0 if kind == "put" else 0.0)


def mc_price(s0, K, r, sigma, T, kind="put", paths=200_000, seed=0):
    """Risk-neutral Monte Carlo price. Returns (price, standard error)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths)
    ST = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(K - ST, 0.0) if kind == "put" else np.maximum(ST - K, 0.0)
    disc = np.exp(-r * T) * payoff
    return float(disc.mean()), float(disc.std(ddof=1) / np.sqrt(paths))
