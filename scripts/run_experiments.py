"""Reproduce every table and figure in the README.

    python scripts/run_experiments.py            # full run (a few minutes)
    python scripts/run_experiments.py --quick    # smaller grids for a smoke test

Outputs go to results/ (CSV tables, a Markdown summary and PNG figures).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qlbs import (Market, Simulation, bs_delta, bs_price, mc_price, off_policy_data,  # noqa: E402
                  replication_error, solve_dp, solve_fqi)

OUT = ROOT / "results"
BASE = Market(s0=100, strike=100, mu=0.05, sigma=0.15, r=0.03, T=1.0)
LAMBDA, ETA = 0.001, 0.5


def fit(market, steps, paths, seed, kind="put", risk_lambda=LAMBDA):
    sim = Simulation(market, steps, paths, seed=seed)
    dp = solve_dp(sim, kind, risk_lambda)
    a, pi, R = off_policy_data(sim, dp.hedge, kind, risk_lambda, ETA, seed=seed)
    fqi = solve_fqi(sim, a, pi, R, risk_lambda)
    return sim, dp, fqi


def out_of_sample_errors(market, steps, train_sim, dp, fqi, kind, test_paths, seed):
    """Apply hedges learned on training paths to fresh paths; compare with BS delta hedging."""
    test = Simulation(market, steps, test_paths, seed=seed, knots=train_sim.knots)
    m, T = market, steps
    tau = m.T - np.arange(T) * test.dt
    bs_hedge = np.c_[bs_delta(test.S[:, :T], m.strike, m.r, m.sigma, tau, kind), np.zeros(test_paths)]
    learned = lambda coef: np.c_[np.stack([test.features[t] @ coef[t] for t in range(T)], axis=1), np.zeros(test_paths)]
    bs0 = float(bs_price(m.s0, m.strike, m.r, m.sigma, m.T, kind))
    return {
        "QLBS-DP": replication_error(test, learned(dp.hedge_coef), dp.price, kind),
        "QLBS-FQI": replication_error(test, learned(fqi.hedge_coef), fqi.price, kind),
        "BS delta": replication_error(test, bs_hedge, bs0, kind),
    }


def base_case(paths):
    rows = []
    for kind in ("put", "call"):
        _, dp, fqi = fit(BASE, 6, paths, 42, kind)
        mc, se = mc_price(BASE.s0, BASE.strike, BASE.r, BASE.sigma, BASE.T, kind)
        rows.append({"option": kind, "Black-Scholes": float(bs_price(BASE.s0, BASE.strike, BASE.r, BASE.sigma, BASE.T, kind)),
                     "Monte Carlo (risk-neutral)": mc, "MC std. error": se,
                     "QLBS DP": dp.price, "QLBS Q-learning (FQI)": fqi.price})
    return pd.DataFrame(rows)


def sweep(param, values, paths, test_paths, seeds):
    rows = []
    for v in values:
        market = replace(BASE, sigma=v) if param == "sigma" else BASE
        steps = v if param == "steps" else 24
        for seed in seeds:
            sim, dp, fqi = fit(market, steps, paths, seed)
            errs = out_of_sample_errors(market, steps, sim, dp, fqi, "put", test_paths, seed + 10_000)
            bs0 = float(bs_price(market.s0, market.strike, market.r, market.sigma, market.T, "put"))
            dp0 = solve_dp(sim, "put", 0.0).price
            row = {param: v, "seed": seed, "BS price": bs0, "QLBS DP price": dp.price, "QLBS FQI price": fqi.price,
                   "QLBS DP price (lambda=0)": dp0}
            for name, e in errs.items():
                row[f"{name} hedge error std"] = e.std()
                row[f"{name} hedge error mean"] = e.mean()
            rows.append(row)
    return pd.DataFrame(rows).groupby(param).mean(numeric_only=True).drop(columns="seed").reset_index()


def convergence(path_grid, seeds):
    bs0 = float(bs_price(BASE.s0, BASE.strike, BASE.r, BASE.sigma, BASE.T, "put"))
    rows = []
    for n in path_grid:
        prices = np.array([[r.price for r in fit(BASE, 6, n, s)[1:]] for s in seeds])
        rows.append({"paths": n, "DP mean": prices[:, 0].mean(), "DP std": prices[:, 0].std(ddof=1),
                     "FQI mean": prices[:, 1].mean(), "FQI std": prices[:, 1].std(ddof=1), "BS": bs0})
    return pd.DataFrame(rows)


def lambda_sweep(lambdas, paths):
    rows = []
    for lam in lambdas:
        _, dp, fqi = fit(BASE, 6, paths, 42, "put", lam)
        rows.append({"risk aversion lambda": lam, "QLBS DP price": dp.price, "QLBS FQI price": fqi.price})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_paths(paths):
    sim, dp, _ = fit(BASE, 6, paths, 42)
    idx = np.linspace(0, paths - 1, 12, dtype=int)
    panels = [(sim.S, r"Stock price $S_t$"), (sim.X, r"State $X_t$"), (dp.hedge, r"Optimal hedge $a_t^\star$"),
              (dp.portfolio, r"Hedge portfolio $\Pi_t$"), (dp.reward, r"Reward $R_t$"), (dp.Q, r"Optimal Q-function $Q_t^\star$")]
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (arr, title) in zip(axes.flat, panels):
        ax.plot(arr[idx].T, lw=1)
        ax.set_title(title)
        ax.set_xlabel("time step")
    fig.savefig(OUT / "paths_put.png", dpi=150)
    plt.close(fig)


def fig_lambda(df):
    bs0 = float(bs_price(BASE.s0, BASE.strike, BASE.r, BASE.sigma, BASE.T, "put"))
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(df["risk aversion lambda"], df["QLBS DP price"], "o-", label="QLBS DP")
    ax.plot(df["risk aversion lambda"], df["QLBS FQI price"], "s--", label="QLBS Q-learning (FQI)")
    ax.axhline(bs0, color="k", lw=1, ls=":", label=f"Black-Scholes = {bs0:.3f}")
    ax.set(xlabel=r"risk aversion $\lambda$", ylabel="put price", title="Price vs risk aversion")
    ax.legend()
    fig.savefig(OUT / "price_vs_lambda.png", dpi=150)
    plt.close(fig)


def fig_convergence(df):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for col, style in (("DP", "o-"), ("FQI", "s--")):
        ax.errorbar(df["paths"], df[f"{col} mean"], yerr=df[f"{col} std"], fmt=style, capsize=3, label=f"QLBS {col}")
    ax.axhline(df["BS"].iloc[0], color="k", lw=1, ls=":", label="Black-Scholes")
    ax.set(xscale="log", xlabel="Monte Carlo paths", ylabel="put price (mean ± 1 sd over seeds)",
           title="Convergence in number of paths")
    ax.legend()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)


def fig_hedge_error(df, param, xlabel, fname):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name, style in (("QLBS-DP", "o-"), ("QLBS-FQI", "s--"), ("BS delta", "^:")):
        ax.plot(df[param], df[f"{name} hedge error std"], style, label=name)
    ax.set(xlabel=xlabel, ylabel="std of terminal hedging error", title="Out-of-sample replication error (put)")
    if param == "steps":
        ax.set_xscale("log")
        ax.set_xticks(df[param], [str(v) for v in df[param]])
    ax.legend()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.quick:
        paths, sweep_paths, test_paths, seeds, path_grid, conv_seeds = 2_000, 5_000, 2_000, [0], [1_000, 5_000], [0, 1]
    else:
        paths, sweep_paths, test_paths, seeds = 10_000, 50_000, 20_000, list(range(5))
        path_grid, conv_seeds = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000], list(range(10))

    tables = {
        "base_case": base_case(paths),
        "risk_aversion": lambda_sweep([0.0, 0.001, 0.002, 0.005, 0.01], paths),
        "volatility_regimes": sweep("sigma", [0.10, 0.15, 0.25, 0.40], sweep_paths, test_paths, seeds),
        "rebalancing_frequency": sweep("steps", [6, 12, 24, 52], sweep_paths, test_paths, seeds),
        "convergence": convergence(path_grid, conv_seeds),
    }
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)

    fig_paths(paths)
    fig_lambda(tables["risk_aversion"])
    fig_convergence(tables["convergence"])
    fig_hedge_error(tables["volatility_regimes"], "sigma", r"volatility $\sigma$", "hedge_error_vs_volatility.png")
    fig_hedge_error(tables["rebalancing_frequency"], "steps", "rebalancing steps per year", "hedge_error_vs_rebalancing.png")

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Results ({'quick' if args.quick else 'full'} run)\n\n")
        for name, df in tables.items():
            f.write(f"## {name}\n\n{df.to_markdown(index=False, floatfmt='.4f')}\n\n")
    print((OUT / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
