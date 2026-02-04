from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate


def evaluate_tbm_interval(
    base_scenario: dict,
    pm_interval_days: int,
    n_runs: int,
    base_seed: int,
    risk_lambda: float,
) -> dict:
    """
    Evaluate TBM for a given PM interval using Monte Carlo.

    We use "common random numbers" to reduce noise:
      - run i uses seed = base_seed + i
      - so every candidate interval is tested on the same randomness
    """
    scenario = deepcopy(base_scenario)
    scenario["pm"]["interval_days"] = int(pm_interval_days)

    total_costs = np.empty(n_runs, dtype=float)
    had_failure = np.empty(n_runs, dtype=bool)

    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        out = simulate(policy="TBM", scenario=scenario, rng=rng)

        total_costs[i] = float(out["total_cost"])
        had_failure[i] = int(out["num_failures"]) >= 1

    mean_cost = float(total_costs.mean())
    p_fail = float(had_failure.mean())
    objective = mean_cost + risk_lambda * p_fail

    return {
        "pm_interval_days": int(pm_interval_days),
        "mean_cost": mean_cost,
        "p_fail": p_fail,
        "objective": objective,
    }


def run_grid(
    scenario: dict,
    intervals: list[int],
    n_runs: int,
    base_seed: int,
    risk_lambda: float,
) -> pd.DataFrame:
    """
    Run evaluation over a list of candidate intervals and return a DataFrame.
    """
    rows = []
    for interval in intervals:
        res = evaluate_tbm_interval(
            base_scenario=scenario,
            pm_interval_days=interval,
            n_runs=n_runs,
            base_seed=base_seed,
            risk_lambda=risk_lambda,
        )
        rows.append(res)
        print(
            f"interval={res['pm_interval_days']:>4d} | "
            f"mean_cost={res['mean_cost']:>10.1f} | "
            f"p_fail={res['p_fail']:.3f} | "
            f"objective={res['objective']:>10.1f}"
        )
    return pd.DataFrame(rows).sort_values("pm_interval_days")


def build_refined_grid(best_interval: int, step: int, half_width: int) -> list[int]:
    """
    Create a refined grid centered around best_interval.
    """
    lo = max(1, best_interval - half_width)
    hi = best_interval + half_width
    return list(range(lo, hi + 1, step))


def main() -> None:
    scenario_path = Path("data/generated/v2/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

    # Read defaults / config
    costs = scenario.get("costs", {}) or {}
    mc = scenario.get("monte_carlo", {}) or {}
    opt = scenario.get("optimization", {}) or {}

    risk_lambda = float(opt.get("risk_lambda", costs.get("failure_penalty", 0.0)))
    base_seed = int(opt.get("seed", mc.get("seed", 42)))

    # Stage 1 (coarse)
    coarse_n_runs = int(opt.get("coarse_n_runs", 2000))
    coarse_step = int(opt.get("coarse_step_days", 10))
    coarse_min = int(opt.get("coarse_min_days", 10))
    coarse_max = int(opt.get("coarse_max_days", 120))
    coarse_grid = list(range(coarse_min, coarse_max + 1, coarse_step))

    print("\n=== TBM OPTIMIZATION (COARSE -> REFINE) ===")
    print("Scenario:", scenario_path)
    print("Objective: mean_cost + lambda * p_fail")
    print(f"lambda = {risk_lambda}")
    print("\n--- Stage 1: Coarse grid ---")
    print(f"grid = {coarse_grid}")
    print(f"n_runs per point = {coarse_n_runs}")

    df_coarse = run_grid(
        scenario=scenario,
        intervals=coarse_grid,
        n_runs=coarse_n_runs,
        base_seed=base_seed,
        risk_lambda=risk_lambda,
    )

    best_coarse_interval = int(df_coarse.loc[df_coarse["objective"].idxmin(), "pm_interval_days"])

    # Stage 2 (refine around best)
    refine_n_runs = int(opt.get("refine_n_runs", 5000))
    refine_step = int(opt.get("refine_step_days", 2))
    refine_half_width = int(opt.get("refine_half_width_days", 10))
    refined_grid = build_refined_grid(best_coarse_interval, step=refine_step, half_width=refine_half_width)

    print("\n--- Stage 2: Refined grid ---")
    print(f"best coarse interval = {best_coarse_interval}")
    print(f"refined grid = {refined_grid}")
    print(f"n_runs per point = {refine_n_runs}")

    df_refined = run_grid(
        scenario=scenario,
        intervals=refined_grid,
        n_runs=refine_n_runs,
        base_seed=base_seed,  # same seeds for fairness
        risk_lambda=risk_lambda,
    )

    best_row = df_refined.loc[df_refined["objective"].idxmin()]

    # Save results
    out_dir = Path("outputs/optimization")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_coarse.to_csv(out_dir / "tbm_grid_coarse.csv", index=False)
    df_refined.to_csv(out_dir / "tbm_grid_refined.csv", index=False)

    # Plot objective curves
    plt.figure()
    plt.plot(df_coarse["pm_interval_days"], df_coarse["objective"], marker="o")
    plt.xlabel("TBM interval (days)")
    plt.ylabel("Objective")
    plt.title("TBM coarse grid: objective vs interval")
    plt.tight_layout()
    plt.savefig(out_dir / "tbm_objective_coarse.png")
    plt.close()

    plt.figure()
    plt.plot(df_refined["pm_interval_days"], df_refined["objective"], marker="o")
    plt.xlabel("TBM interval (days)")
    plt.ylabel("Objective")
    plt.title("TBM refined grid: objective vs interval")
    plt.tight_layout()
    plt.savefig(out_dir / "tbm_objective_refined.png")
    plt.close()

    print("\n=== BEST TBM INTERVAL (REFINED) ===")
    print(f"pm_interval_days = {int(best_row['pm_interval_days'])}")
    print(f"mean_cost        = {best_row['mean_cost']:.2f}")
    print(f"p_fail           = {best_row['p_fail']:.4f}")
    print(f"objective        = {best_row['objective']:.2f}")

    print(f"\nSaved: {out_dir / 'tbm_grid_coarse.csv'}")
    print(f"Saved: {out_dir / 'tbm_grid_refined.csv'}")
    print(f"Saved: {out_dir / 'tbm_objective_coarse.png'}")
    print(f"Saved: {out_dir / 'tbm_objective_refined.png'}")


if __name__ == "__main__":
    main()