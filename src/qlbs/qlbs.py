"""QLBS: option pricing and hedging as a Markov decision process (Halperin, 2017).

Two solvers on the same simulated data:

* ``solve_dp``: model-based dynamic programming. The hedge and the Q-function are
  expanded in B-spline features and fitted backward in time by regularised least squares.
* ``solve_fqi``: model-free fitted Q iteration. Trained only on off-policy data
  (noisy hedges and the rewards they produced), with the Q-function quadratic in the action.

Option price under QLBS = -Q_0(X_0, a_0*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulation import Simulation


@dataclass
class QLBSResult:
    price: float
    hedge: np.ndarray       # (paths, steps+1) optimal action a_t*
    portfolio: np.ndarray   # (paths, steps+1) hedge portfolio value Pi_t
    reward: np.ndarray      # (paths, steps+1)
    Q: np.ndarray           # (paths, steps+1)
    hedge_coef: np.ndarray  # (steps, n_basis): a_t*(x) = features(x) @ hedge_coef[t]


def _ridge(A: np.ndarray, b: np.ndarray, reg: float) -> np.ndarray:
    """Ridge solve with the penalty scaled to the average diagonal of A.

    A scale-free penalty keeps the fit stable when feature magnitudes change with the
    horizon and volatility; a fixed absolute penalty let fitted Q iteration diverge on
    long horizons (24+ steps).
    """
    scale = np.trace(A) / A.shape[0]
    return np.linalg.solve(A + reg * scale * np.eye(A.shape[0]), b)


def _rewards(sim: Simulation, hedge, portfolio, risk_lambda):
    """R_t = gamma * a_t * dS_t - lambda * Var[Pi_t];  R_T = -lambda * Var[Pi_T]."""
    R = np.empty_like(portfolio)
    R[:, -1] = -risk_lambda * portfolio[:, -1].var()
    R[:, :-1] = sim.gamma * hedge[:, :-1] * sim.dS - risk_lambda * portfolio[:, :-1].var(axis=0)
    return R


def _hedge_step(sim: Simulation, t: int, pi_hat_next: np.ndarray, reg: float):
    """Variance-minimising hedge at time t: a_t = argmin Var[Pi_t | F_t].

    Note: Halperin's full formula adds a 1/(2*gamma*lambda) * dS_t term (the mean-variance
    drift term). Like the original implementation, we use the pure variance-minimising hedge.
    """
    Phi, dSh = sim.features[t], sim.dS_hat[:, t]
    A = Phi.T @ (Phi * dSh[:, None] ** 2)
    B = Phi.T @ (pi_hat_next * dSh)
    return _ridge(A, B, reg)


def _roll_portfolio(sim: Simulation, payoff, hedge_fn):
    """Backward recursion Pi_t = gamma * (Pi_{t+1} - a_t * dS_t), Pi_T = payoff."""
    n, T = sim.paths, sim.steps
    a, pi = np.zeros((n, T + 1)), np.zeros((n, T + 1))
    pi[:, -1] = payoff
    for t in range(T - 1, -1, -1):
        a[:, t] = hedge_fn(t, pi[:, t + 1] - pi[:, t + 1].mean())
        pi[:, t] = sim.gamma * (pi[:, t + 1] - a[:, t] * sim.dS[:, t])
    return a, pi


def solve_dp(sim: Simulation, kind: str = "put", risk_lambda: float = 0.001, reg: float = 1e-4) -> QLBSResult:
    coef = np.zeros((sim.steps, sim.features.shape[2]))

    def hedge_fn(t, pi_hat_next):
        coef[t] = _hedge_step(sim, t, pi_hat_next, reg)
        return sim.features[t] @ coef[t]

    a, pi = _roll_portfolio(sim, sim.payoff(kind), hedge_fn)
    R = _rewards(sim, a, pi, risk_lambda)

    Q = np.zeros_like(pi)
    Q[:, -1] = -pi[:, -1] - risk_lambda * pi[:, -1].var()
    for t in range(sim.steps - 1, -1, -1):
        Phi = sim.features[t]
        omega = _ridge(Phi.T @ Phi, Phi.T @ (R[:, t] + sim.gamma * Q[:, t + 1]), reg)
        Q[:, t] = Phi @ omega

    return QLBSResult(float(-Q[:, 0].mean()), a, pi, R, Q, coef)


def off_policy_data(sim: Simulation, optimal_hedge: np.ndarray, kind: str = "put",
                    risk_lambda: float = 0.001, eta: float = 0.5, seed: int | None = 42):
    """Behaviour policy: optimal hedge times U[1-eta, 1+eta] noise. Returns (actions, portfolio, rewards)."""
    rng = np.random.RandomState(seed)

    def noisy(t, _):
        return optimal_hedge[:, t] * rng.uniform(1 - eta, 1 + eta, size=sim.paths)

    a, pi = _roll_portfolio(sim, sim.payoff(kind), noisy)
    return a, pi, _rewards(sim, a, pi, risk_lambda)


def solve_fqi(sim: Simulation, actions, portfolio, rewards, risk_lambda: float = 0.001,
              reg: float = 1e-4) -> QLBSResult:
    """Fitted Q iteration on off-policy data.

    Q_t(x, a) = W_t . Psi(x, a),  Psi = (1, a, a^2/2) (x) Phi(x), fitted by least squares on the
    Bellman target R_t + gamma * Q_{t+1}(X_{t+1}, a*_{t+1}). The greedy action a*_t is the
    variance-minimising hedge estimated from the same off-policy portfolio values.
    """
    n, T, nb = sim.paths, sim.steps, sim.features.shape[2]
    Q = np.zeros((n, T + 1))
    Q[:, -1] = -portfolio[:, -1] - risk_lambda * portfolio[:, -1].var()
    a_star = np.zeros((n, T + 1))
    coef = np.zeros((T, nb))

    for t in range(T - 1, -1, -1):
        Phi, a = sim.features[t], actions[:, t]
        act = np.stack([np.ones(n), a, 0.5 * a**2], axis=1)      # (n, 3)
        Psi = (Phi[:, :, None] * act[:, None, :]).reshape(n, -1)  # (n, 3*nb)
        W = _ridge(Psi.T @ Psi, Psi.T @ (rewards[:, t] + sim.gamma * Q[:, t + 1]), reg)
        U = (Phi @ W.reshape(nb, 3)).T                            # (3, n): U0, U1, U2

        pi_next = portfolio[:, t + 1]
        coef[t] = _hedge_step(sim, t, pi_next - pi_next.mean(), reg)
        a_star[:, t] = Phi @ coef[t]
        Q[:, t] = U[0] + a_star[:, t] * U[1] + 0.5 * a_star[:, t] ** 2 * U[2]

    return QLBSResult(float(-Q[:, 0].mean()), a_star, portfolio, rewards, Q, coef)


def replication_error(sim: Simulation, hedge: np.ndarray, price: float, kind: str = "put") -> np.ndarray:
    """Terminal P&L of a self-financing hedge started with ``price`` in cash.

    V_{t+1} = e^{r dt} V_t + a_t * dS_t. Returns V_T - payoff per path (0 = perfect replication).
    ``hedge`` is the number of shares in the replicating portfolio (QLBS a_t* or the
    Black-Scholes delta), so a put hedge is negative.
    """
    growth = np.exp(sim.market.r * sim.dt)
    V = np.full(sim.paths, price, dtype=float)
    for t in range(sim.steps):
        V = growth * V + hedge[:, t] * sim.dS[:, t]
    return V - sim.payoff(kind)
