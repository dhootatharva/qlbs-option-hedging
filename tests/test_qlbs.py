import numpy as np
import pytest

from qlbs import Market, Simulation, bs_delta, bs_price, off_policy_data, solve_dp, solve_fqi
from qlbs.qlbs import _roll_portfolio

M = Market()


def test_put_call_parity():
    c = bs_price(M.s0, M.strike, M.r, M.sigma, M.T, "call")
    p = bs_price(M.s0, M.strike, M.r, M.sigma, M.T, "put")
    assert c - p == pytest.approx(M.s0 - M.strike * np.exp(-M.r * M.T))


def test_basis_is_partition_of_unity():
    sim = Simulation(M, steps=6, paths=2_000, seed=0)
    np.testing.assert_allclose(sim.features.sum(axis=2), 1.0, atol=1e-12)


def test_backward_recursion_with_bs_delta_recovers_bs_price():
    steps = 24
    sim = Simulation(M, steps=steps, paths=50_000, seed=1)
    tau = M.T - np.arange(steps) * sim.dt
    delta = bs_delta(sim.S[:, :steps], M.strike, M.r, M.sigma, tau, "put")
    _, pi = _roll_portfolio(sim, sim.payoff("put"), lambda t, _: delta[:, t])
    assert pi[:, 0].mean() == pytest.approx(float(bs_price(M.s0, M.strike, M.r, M.sigma, M.T, "put")), rel=0.01)


def test_dp_price_close_to_bs_without_risk_aversion():
    sim = Simulation(M, steps=6, paths=20_000, seed=2)
    dp = solve_dp(sim, "put", risk_lambda=0.0)
    assert dp.price == pytest.approx(float(bs_price(M.s0, M.strike, M.r, M.sigma, M.T, "put")), rel=0.02)


def test_risk_aversion_raises_price():
    sim = Simulation(M, steps=6, paths=10_000, seed=3)
    assert solve_dp(sim, "put", 0.005).price > solve_dp(sim, "put", 0.0).price


def test_fqi_learns_dp_price_from_off_policy_data():
    sim = Simulation(M, steps=6, paths=20_000, seed=4)
    dp = solve_dp(sim, "put")
    a, pi, R = off_policy_data(sim, dp.hedge, "put", seed=4)
    assert solve_fqi(sim, a, pi, R).price == pytest.approx(dp.price, rel=0.03)
