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

    out_dir = Path("outputs/repair_cost_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = scenario0.get("monte_carlo", {}) or {}
    opt = scenario0.get("optimization", {}) or {}
    costs0 = scenario0.get("costs", {}) or {}

    # Fixed risk preference (use your chosen lambda)
    lam = float(opt.get("risk_lambda", 20000.0))

    # Monte Carlo budget per grid point
    n_runs = int(opt.get("repair_sweep_n_runs", mc.get("n_runs", 3000)))
    base_seed = int(opt.get("seed", mc.get("seed", 42)))

    # Policy parameter grids
    tbm_grid = opt.get("tbm_interval_grid", list(range(10, 121, 5)))
    cbm_grid = opt.get("cbm_threshold_grid", list(range(0, 201, 5)))

    # Repair cost sweep
    rc_min = float(opt.get("repair_cost_min", 2000))
    rc_max = float(opt.get("repair_cost_max", 20000))
    rc_step = float(opt.get("repair_cost_step", 2000))
    repair_cost_values = np.arange(rc_min, rc_max + 1e-9, rc_step)

    print("\n=== Repair cost crossover sweep (CBM vs TBM) ===")
    print(f"Scenario: {scenario_path}")
    print(f"Objective: mean_cost + lambda * p_fail  with lambda={lam}")
    print(f"n_runs per grid point: {n_runs}")
    print(f"TBM grid size: {len(tbm_grid)} | CBM grid size: {len(cbm_grid)}")
    print(f"Repair cost sweep: {rc_min}..{rc_max} step {rc_step} (n={len(repair_cost_values)})")
    print(f"Base repair_cost in scenario: {costs0.get('repair_cost')}")

    rows = []

    for rc in repair_cost_values:
        scenario = deepcopy(scenario0)
        scenario["costs"]["repair_cost"] = float(rc)

        # Evaluate TBM and CBM grids for this repair cost
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
                "repair_cost": float(rc),
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
            f"repair_cost={rc:>7.0f} | "
            f"TBM obj={tbm_best['best_objective']:>10.1f} (int={tbm_best['best_param']:>5.0f}, p_fail={tbm_best['best_p_fail']:.3f}) | "
            f"CBM obj={cbm_best['best_objective']:>10.1f} (thr={cbm_best['best_param']:>6.1f}, p_fail={cbm_best['best_p_fail']:.3f}) | "
            f"CBM-TBM={cbm_best['best_objective'] - tbm_best['best_objective']:>9.1f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "repair_cost_sweep.csv", index=False)

    # Find first repair_cost where CBM beats TBM (cbm_minus_tbm < 0)
    crossover = None
    mask = df["cbm_minus_tbm"] < 0
    if mask.any():
        crossover = float(df.loc[mask, "repair_cost"].iloc[0])

    # Plots
    plt.figure()
    plt.plot(df["repair_cost"], df["tbm_best_objective"], label="Best TBM objective")
    plt.plot(df["repair_cost"], df["cbm_best_objective"], label="Best CBM objective")
    if crossover is not None:
        plt.axvline(crossover, linestyle="--", label=f"Crossover repair_cost ≈ {crossover:.0f}")
    plt.xlabel("Repair cost")
    plt.ylabel("Best objective value")
    plt.title(f"Best TBM vs Best CBM across repair_cost (λ={lam:g})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "best_objective_vs_repair_cost.png")
    plt.close()

    plt.figure()
    plt.plot(df["repair_cost"], df["cbm_minus_tbm"])
    plt.axhline(0, linestyle="--")
    if crossover is not None:
        plt.axvline(crossover, linestyle="--")
    plt.xlabel("Repair cost")
    plt.ylabel("Objective difference (CBM - TBM)")
    plt.title("When does CBM beat TBM? (negative = CBM better)")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_minus_tbm_vs_repair_cost.png")
    plt.close()

    print("\nSaved to:", out_dir.resolve())
    if crossover is None:
        print("Crossover: CBM never beats TBM in the tested repair_cost range.")
    else:
        print(f"Crossover: CBM becomes better than TBM at repair_cost ≈ {crossover:.0f} (for λ={lam:g}).")


if __name__ == "__main__":
    main()
