from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate


def evaluate_grid_mean_pfail(
    base_scenario: dict,
    policy: str,
    param_name: str,
    param_values: list[float],
    n_runs: int,
    base_seed: int,
) -> pd.DataFrame:
    """
    Evaluate mean_cost and p_fail for a grid of policy parameters, for one scenario instance.
    Uses common random numbers across param values: run i uses seed = base_seed + i.
    """
    rows = []

    for v in param_values:
        scenario = deepcopy(base_scenario)

        if policy == "TBM":
            scenario["pm"]["interval_days"] = int(v)
        elif policy == "CBM":
            scenario["cbm"]["threshold_days"] = float(v)
        else:
            raise ValueError("policy must be 'TBM' or 'CBM'")

        costs = np.empty(n_runs, dtype=float)
        had_fail = np.empty(n_runs, dtype=bool)

        for i in range(n_runs):
            rng = np.random.default_rng(base_seed + i)
            out = simulate(policy=policy, scenario=scenario, rng=rng)
            costs[i] = float(out["total_cost"])
            had_fail[i] = int(out["num_failures"]) >= 1

        rows.append(
            {
                "policy": policy,
                param_name: float(v),
                "mean_cost": float(costs.mean()),
                "p_fail": float(had_fail.mean()),
            }
        )

    return pd.DataFrame(rows)


def best_for_lambda(df: pd.DataFrame, param_col: str, lam: float) -> dict:
    """
    For a fixed lambda, compute objective = mean_cost + lam * p_fail,
    return best row info.
    """
    obj = df["mean_cost"].to_numpy() + lam * df["p_fail"].to_numpy()
    idx = int(np.argmin(obj))
    return {
        "best_objective": float(obj[idx]),
        "best_param": float(df.iloc[idx][param_col]),
        "best_mean_cost": float(df.iloc[idx]["mean_cost"]),
        "best_p_fail": float(df.iloc[idx]["p_fail"]),
    }


def main() -> None:
    scenario_path = Path("data/generated/v2/scenario.json").resolve()
    scenario0 = load_scenario(scenario_path)

    out_dir = Path("outputs/shape_k_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = scenario0.get("monte_carlo", {}) or {}
    opt = scenario0.get("optimization", {}) or {}

    # Risk preference (fixed during sweep)
    lam = float(opt.get("risk_lambda", 20000.0))

    # Monte Carlo budget per grid point (reduce if too slow)
    n_runs = int(opt.get("shape_k_sweep_n_runs", mc.get("n_runs", 3000)))
    base_seed = int(opt.get("seed", mc.get("seed", 42)))

    # Policy parameter grids (reuse if present)
    tbm_grid = opt.get("tbm_interval_grid", list(range(10, 121, 5)))
    cbm_grid = opt.get("cbm_threshold_grid", list(range(0, 201, 5)))

    # Sweep shape_k
    k_min = float(opt.get("shape_k_min", 0.8))
    k_max = float(opt.get("shape_k_max", 2.5))
    k_step = float(opt.get("shape_k_step", 0.2))
    k_values = np.arange(k_min, k_max + 1e-9, k_step)

    # Keep lambda scale fixed; keep scale_lambda_days fixed unless you explicitly change it
    fail0 = scenario0.get("failure", {}) or {}
    k0 = fail0.get("shape_k", None)

    print("\n=== shape_k sweep (CBM vs TBM) ===")
    print(f"Scenario: {scenario_path}")
    print(f"Objective: mean_cost + lambda * p_fail  with lambda={lam}")
    print(f"n_runs per grid point: {n_runs}")
    print(f"TBM grid size: {len(tbm_grid)} | CBM grid size: {len(cbm_grid)}")
    print(f"shape_k sweep: {k_min}..{k_max} step {k_step} (n={len(k_values)})")
    print(f"Base shape_k in scenario: {k0}")

    rows = []

    for k in k_values:
        scenario = deepcopy(scenario0)
        scenario["failure"]["shape_k"] = float(k)

        # Evaluate TBM and CBM grids for this k
        df_tbm = evaluate_grid_mean_pfail(
            base_scenario=scenario,
            policy="TBM",
            param_name="pm_interval_days",
            param_values=[int(x) for x in tbm_grid],
            n_runs=n_runs,
            base_seed=base_seed,
        )
        df_cbm = evaluate_grid_mean_pfail(
            base_scenario=scenario,
            policy="CBM",
            param_name="threshold_days",
            param_values=[float(x) for x in cbm_grid],
            n_runs=n_runs,
            base_seed=base_seed,
        )

        tbm_best = best_for_lambda(df_tbm, "pm_interval_days", lam)
        cbm_best = best_for_lambda(df_cbm, "threshold_days", lam)

        rows.append(
            {
                "shape_k": float(k),
                "lambda": lam,
                "tbm_best_objective": tbm_best["best_objective"],
                "tbm_best_interval_days": tbm_best["best_param"],
                "tbm_best_mean_cost": tbm_best["best_mean_cost"],
                "tbm_best_p_fail": tbm_best["best_p_fail"],
                "cbm_best_objective": cbm_best["best_objective"],
                "cbm_best_threshold_days": cbm_best["best_param"],
                "cbm_best_mean_cost": cbm_best["best_mean_cost"],
                "cbm_best_p_fail": cbm_best["best_p_fail"],
                "cbm_minus_tbm": cbm_best["best_objective"] - tbm_best["best_objective"],
            }
        )

        print(
            f"k={k:>4.1f} | "
            f"TBM obj={tbm_best['best_objective']:>10.1f} (int={tbm_best['best_param']:>5.0f}, p_fail={tbm_best['best_p_fail']:.3f}) | "
            f"CBM obj={cbm_best['best_objective']:>10.1f} (thr={cbm_best['best_param']:>6.1f}, p_fail={cbm_best['best_p_fail']:.3f}) | "
            f"CBM-TBM={cbm_best['best_objective'] - tbm_best['best_objective']:>9.1f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "shape_k_sweep.csv", index=False)

    # Find crossover: first k where CBM beats TBM (cbm_minus_tbm < 0)
    crossover_k = None
    mask = df["cbm_minus_tbm"] < 0
    if mask.any():
        crossover_k = float(df.loc[mask, "shape_k"].iloc[0])

    # Plots
    plt.figure()
    plt.plot(df["shape_k"], df["tbm_best_objective"], label="Best TBM objective")
    plt.plot(df["shape_k"], df["cbm_best_objective"], label="Best CBM objective")
    if crossover_k is not None:
        plt.axvline(crossover_k, linestyle="--", label=f"Crossover k ≈ {crossover_k:.2f}")
    plt.xlabel("Weibull shape k")
    plt.ylabel("Best objective value")
    plt.title(f"Best TBM vs Best CBM across Weibull shape k (λ={lam:g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_objective_vs_shape_k.png")
    plt.close()

    plt.figure()
    plt.plot(df["shape_k"], df["cbm_minus_tbm"])
    plt.axhline(0, linestyle="--")
    if crossover_k is not None:
        plt.axvline(crossover_k, linestyle="--")
    plt.xlabel("Weibull shape k")
    plt.ylabel("Objective difference (CBM - TBM)")
    plt.title("When does CBM beat TBM? (negative = CBM better)")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_minus_tbm_vs_shape_k.png")
    plt.close()

    # Best parameter vs k
    plt.figure()
    plt.plot(df["shape_k"], df["tbm_best_interval_days"])
    plt.xlabel("Weibull shape k")
    plt.ylabel("Best TBM interval (days)")
    plt.title("Optimal TBM interval vs Weibull shape k")
    plt.tight_layout()
    plt.savefig(out_dir / "best_tbm_interval_vs_shape_k.png")
    plt.close()

    plt.figure()
    plt.plot(df["shape_k"], df["cbm_best_threshold_days"])
    plt.xlabel("Weibull shape k")
    plt.ylabel("Best CBM threshold (days)")
    plt.title("Optimal CBM threshold vs Weibull shape k")
    plt.tight_layout()
    plt.savefig(out_dir / "best_cbm_threshold_vs_shape_k.png")
    plt.close()

    print("\nSaved to:", out_dir.resolve())
    if crossover_k is None:
        print("Crossover: CBM never beats TBM in the tested k range.")
    else:
        print(f"Crossover: CBM becomes better than TBM at approximately k ≈ {crossover_k:.2f} (for λ={lam:g}).")


if __name__ == "__main__":
    main()
