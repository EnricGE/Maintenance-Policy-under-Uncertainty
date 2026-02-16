from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate


def evaluate_policy_grid(
    base_scenario: dict,
    policy: str,
    param_name: str,
    param_values: list[float],
    n_runs: int,
    base_seed: int,
) -> pd.DataFrame:
    """
    Evaluate (mean_cost, p_fail) for a grid of policy parameters.
    Uses common random numbers: run i uses seed = base_seed + i for every param.
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


def best_objective_for_lambda(df: pd.DataFrame, param_col: str, lam: float) -> dict:
    """
    For a given lambda, compute objective for all rows and return the best one.
    """
    obj = df["mean_cost"].to_numpy() + lam * df["p_fail"].to_numpy()
    idx = int(np.argmin(obj))
    return {
        "lambda": float(lam),
        "best_objective": float(obj[idx]),
        "best_param": float(df.iloc[idx][param_col]),
        "best_mean_cost": float(df.iloc[idx]["mean_cost"]),
        "best_p_fail": float(df.iloc[idx]["p_fail"]),
    }


def main() -> None:
    scenario_path = Path("data/generated/v2/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

    out_dir = Path("outputs/risk_aversion_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = scenario.get("monte_carlo", {}) or {}
    opt = scenario.get("optimization", {}) or {}

    # One-time MC effort (per parameter candidate)
    n_runs = int(opt.get("crossover_n_runs", mc.get("n_runs", 5000)))
    base_seed = int(opt.get("seed", mc.get("seed", 42)))

    # Parameter grids (keep simple; adjust as needed)
    tbm_grid = opt.get("tbm_interval_grid", list(range(10, 121, 5)))
    cbm_grid = opt.get("cbm_threshold_grid", list(range(0, 201, 5)))

    # Lambda sweep
    lam_min = float(opt.get("lambda_min", 0))
    lam_max = float(opt.get("lambda_max", 50000))
    lam_step = float(opt.get("lambda_step", 250))
    lambdas = np.arange(lam_min, lam_max + 1e-9, lam_step)

    print("\n=== Crossover lambda sweep (CBM vs TBM) ===")
    print(f"Scenario: {scenario_path}")
    print(f"n_runs per grid point: {n_runs}")
    print(f"TBM grid size: {len(tbm_grid)}  | CBM grid size: {len(cbm_grid)}")
    print(f"Lambda sweep: {lam_min}..{lam_max} step {lam_step} (n={len(lambdas)})")

    # 1) Evaluate grids ONCE
    df_tbm = evaluate_policy_grid(
        base_scenario=scenario,
        policy="TBM",
        param_name="pm_interval_days",
        param_values=[int(x) for x in tbm_grid],
        n_runs=n_runs,
        base_seed=base_seed,
    )
    df_cbm = evaluate_policy_grid(
        base_scenario=scenario,
        policy="CBM",
        param_name="threshold_days",
        param_values=[float(x) for x in cbm_grid],
        n_runs=n_runs,
        base_seed=base_seed,
    )

    df_tbm.to_csv(out_dir / "tbm_grid_mean_pfail.csv", index=False)
    df_cbm.to_csv(out_dir / "cbm_grid_mean_pfail.csv", index=False)

    # 2) Sweep lambda and record best TBM and best CBM
    tbm_best_rows = []
    cbm_best_rows = []
    for lam in lambdas:
        tbm_best_rows.append(best_objective_for_lambda(df_tbm, "pm_interval_days", lam))
        cbm_best_rows.append(best_objective_for_lambda(df_cbm, "threshold_days", lam))

    best_tbm = pd.DataFrame(tbm_best_rows).rename(
        columns={
            "best_objective": "tbm_best_objective",
            "best_param": "tbm_best_interval_days",
            "best_mean_cost": "tbm_best_mean_cost",
            "best_p_fail": "tbm_best_p_fail",
        }
    )
    best_cbm = pd.DataFrame(cbm_best_rows).rename(
        columns={
            "best_objective": "cbm_best_objective",
            "best_param": "cbm_best_threshold_days",
            "best_mean_cost": "cbm_best_mean_cost",
            "best_p_fail": "cbm_best_p_fail",
        }
    )

    df_sweep = best_tbm.merge(best_cbm, on="lambda")
    df_sweep["cbm_minus_tbm"] = df_sweep["cbm_best_objective"] - df_sweep["tbm_best_objective"]
    df_sweep.to_csv(out_dir / "lambda_sweep_best_policies.csv", index=False)

    # 3) Find crossover (first lambda where CBM beats TBM)
    crossover = None
    mask = df_sweep["cbm_minus_tbm"] < 0
    if mask.any():
        crossover = float(df_sweep.loc[mask, "lambda"].iloc[0])

    # 4) Plots (saved)
    # Objective vs lambda
    plt.figure()
    plt.plot(df_sweep["lambda"], df_sweep["tbm_best_objective"], label="Best TBM objective")
    plt.plot(df_sweep["lambda"], df_sweep["cbm_best_objective"], label="Best CBM objective")
    if crossover is not None:
        plt.axvline(crossover, linestyle="--", label=f"Crossover λ ≈ {crossover:g}")
    plt.xlabel("Risk aversion λ")
    plt.ylabel("Best objective value")
    plt.title("Best TBM vs Best CBM objective across λ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_objective_vs_lambda.png")
    plt.close()

    # Difference vs lambda (CBM - TBM)
    plt.figure()
    plt.plot(df_sweep["lambda"], df_sweep["cbm_minus_tbm"])
    plt.axhline(0, linestyle="--")
    if crossover is not None:
        plt.axvline(crossover, linestyle="--")
    plt.xlabel("Risk aversion λ")
    plt.ylabel("Objective difference (CBM - TBM)")
    plt.title("When does CBM beat TBM? (negative = CBM better)")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_minus_tbm_vs_lambda.png")
    plt.close()

    # Best parameter vs lambda (piecewise constant)
    plt.figure()
    plt.plot(df_sweep["lambda"], df_sweep["tbm_best_interval_days"])
    plt.xlabel("Risk aversion λ")
    plt.ylabel("Best TBM interval (days)")
    plt.title("Optimal TBM interval vs λ")
    plt.tight_layout()
    plt.savefig(out_dir / "best_tbm_interval_vs_lambda.png")
    plt.close()

    plt.figure()
    plt.plot(df_sweep["lambda"], df_sweep["cbm_best_threshold_days"])
    plt.xlabel("Risk aversion λ")
    plt.ylabel("Best CBM threshold (days)")
    plt.title("Optimal CBM threshold vs λ")
    plt.tight_layout()
    plt.savefig(out_dir / "best_cbm_threshold_vs_lambda.png")
    plt.close()

    # 5) Print result
    print("\nSaved to:", out_dir.resolve())
    if crossover is None:
        print("Crossover: CBM never beats TBM in the tested λ range.")
    else:
        print(f"Crossover: CBM becomes better than TBM at approximately λ ≈ {crossover:g}")


if __name__ == "__main__":
    main()
