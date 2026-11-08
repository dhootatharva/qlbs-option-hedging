"""Monte Carlo paths under the real-world measure and the B-spline state features used by QLBS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class Market:
    s0: float = 100.0
    strike: float = 100.0
    mu: float = 0.05  # real-world drift
    sigma: float = 0.15
    r: float = 0.03
    T: float = 1.0


def knot_vector(lo: float, hi: float, n_basis: int = 12, degree: int = 4) -> np.ndarray:
    """Clamped knot vector with ``n_basis`` functions on [lo, hi].

    Same construction as ``splinelab.aptknt`` (endpoints repeated, interior knots are
    running averages of evenly spaced sites), which the original notebook used.
    """
    tau = np.linspace(lo, hi, n_basis)
    k = degree + 1
    interior = np.array([tau[i : i + k - 1].mean() for i in range(1, n_basis - k + 1)])
    return np.r_[np.full(k, lo), interior, np.full(k, hi)]


def basis_matrix(x: np.ndarray, knots: np.ndarray, degree: int = 4) -> np.ndarray:
    """Evaluate all B-spline basis functions at ``x``. Shape (len(x), n_basis).

    Points outside the knot range are clipped to it, so features are defined for
    out-of-sample paths too.
    """
    lo, hi = knots[0], knots[-1]
    x = np.clip(x, lo, np.nextafter(hi, -np.inf))
    return BSpline.design_matrix(x, knots, degree).toarray()


class Simulation:
    """Simulated stock paths plus everything the QLBS recursions need.

    Arrays are indexed ``[path, time]`` with ``time = 0..steps``.
    """

    def __init__(self, market: Market, steps: int, paths: int, seed: int | None = 42,
                 n_basis: int = 12, degree: int = 4, knots: np.ndarray | None = None):
        self.market, self.steps, self.paths = market, steps, paths
        self.dt = market.T / steps
        self.gamma = np.exp(-market.r * self.dt)  # one-step discount factor

        # RandomState keeps the exact random stream of the original notebook (np.random.seed(42)).
        z = np.random.RandomState(seed).randn(paths, steps)
        m = market
        log_inc = (m.mu - 0.5 * m.sigma**2) * self.dt + m.sigma * np.sqrt(self.dt) * z
        self.S = m.s0 * np.exp(np.c_[np.zeros(paths), np.cumsum(log_inc, axis=1)])

        # Excess one-step stock move and its cross-sectional centred version.
        self.dS = self.S[:, 1:] - np.exp(m.r * self.dt) * self.S[:, :-1]
        self.dS_hat = self.dS - self.dS.mean(axis=0)

        # Drift-adjusted log price: a stationary-ish state variable.
        self.X = np.log(self.S) - (m.mu - 0.5 * m.sigma**2) * np.arange(steps + 1) * self.dt

        self.degree = degree
        self.knots = knots if knots is not None else knot_vector(self.X.min(), self.X.max(), n_basis, degree)
        self.features = np.stack([basis_matrix(self.X[:, t], self.knots, degree) for t in range(steps + 1)])

    def payoff(self, kind: str = "put") -> np.ndarray:
        ST, K = self.S[:, -1], self.market.strike
        return np.maximum(K - ST, 0.0) if kind == "put" else np.maximum(ST - K, 0.0)
