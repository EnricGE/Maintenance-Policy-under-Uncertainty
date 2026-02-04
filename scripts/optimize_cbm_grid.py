from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate


def evaluate_cbm_threshold(
    base_scenario: dict,
    threshold_days: float,
    n_runs: int,
    base_seed: int,
    risk_lambda: float,
) -> dict:
    """
    Evaluate CBM performance for a given threshold using Monte Carlo.

    Uses common random numbers:
      run i uses seed = base_seed + i
      so all thresholds face the same randomness (less noise in comparisons).
    """
    scenario = deepcopy(base_scenario)

    cbm_cfg = scenario.get("cbm") or {}
    if not cbm_cfg:
        raise ValueError("CBM optimization requires a 'cbm' block in scenario.json")

    scenario["cbm"]["threshold_days"] = float(threshold_days)

    total_costs = np.empty(n_runs, dtype=float)
    had_failure = np.empty(n_runs, dtype=bool)
    cbm_actions = np.empty(n_runs, dtype=float)
    inspections = np.empty(n_runs, dtype=float)

    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        out = simulate(policy="CBM", scenario=scenario, rng=rng)

        total_costs[i] = float(out["total_cost"])
        had_failure[i] = int(out["num_failures"]) >= 1
        cbm_actions[i] = float(out.get("num_cbm_actions", 0))
        inspections[i] = float(out.get("num_inspections", 0))

    mean_cost = float(total_costs.mean())
    p_fail = float(had_failure.mean())
    objective = mean_cost + risk_lambda * p_fail

    return {
        "threshold_days": float(threshold_days),
        "mean_cost": mean_cost,
        "p_fail": p_fail,
        "objective": objective,
        "mean_cbm_actions": float(cbm_actions.mean()),
        "mean_inspections": float(inspections.mean()),
    }


def run_grid(
    scenario: dict,
    thresholds: list[float],
    n_runs: int,
    base_seed: int,
    risk_lambda: float,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        res = evaluate_cbm_threshold(
            base_scenario=scenario,
            threshold_days=float(thr),
            n_runs=n_runs,
            base_seed=base_seed,
            risk_lambda=risk_lambda,
        )
        rows.append(res)
        print(
            f"thr={res['threshold_days']:>6.1f} | "
            f"mean_cost={res['mean_cost']:>10.1f} | "
            f"p_fail={res['p_fail']:.3f} | "
            f"obj={res['objective']:>10.1f} | "
            f"actions={res['mean_cbm_actions']:.2f} | "
            f"inspections={res['mean_inspections']:.1f}"
        )
    return pd.DataFrame(rows).sort_values("threshold_days")


def build_refined_grid(best_thr: float, step: float, half_width: float) -> list[float]:
    lo = max(0.0, best_thr - half_width)
    hi = best_thr + half_width
    # keep nice rounded thresholds
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def main() -> None:
    scenario_path = Path("data/generated/v2/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

    costs = scenario.get("costs", {}) or {}
    mc = scenario.get("monte_carlo", {}) or {}
    opt = scenario.get("optimization", {}) or {}

    risk_lambda = float(opt.get("risk_lambda", costs.get("failure_penalty", 0.0)))
    base_seed = int(opt.get("seed", mc.get("seed", 42)))

    cbm_cfg = scenario.get("cbm") or {}
    if not cbm_cfg:
        raise ValueError("scenario.json must contain a 'cbm' block to optimize CBM.")

    print("\n=== CBM OPTIMIZATION (COARSE -> REFINE) ===")
    print("Scenario:", scenario_path)
    print("Objective: mean_cost + lambda * p_fail")
    print(f"lambda = {risk_lambda}")

    # ---- Stage 1: coarse thresholds
    coarse_n_runs = int(opt.get("coarse_n_runs", 2000))
    coarse_step = float(opt.get("cbm_coarse_step_days", 10))
    coarse_min = float(opt.get("cbm_coarse_min_days", 20))
    coarse_max = float(opt.get("cbm_coarse_max_days", 200))
    coarse_grid = list(np.arange(coarse_min, coarse_max + 1e-9, coarse_step))

    print("\n--- Stage 1: Coarse grid ---")
    print(f"threshold grid = [{coarse_min}..{coarse_max}] step {coarse_step}")
    print(f"n_runs per point = {coarse_n_runs}")

    df_coarse = run_grid(
        scenario=scenario,
        thresholds=coarse_grid,
        n_runs=coarse_n_runs,
        base_seed=base_seed,
        risk_lambda=risk_lambda,
    )
    best_coarse_thr = float(df_coarse.loc[df_coarse["objective"].idxmin(), "threshold_days"])

    # ---- Stage 2: refined around best
    refine_n_runs = int(opt.get("refine_n_runs", 5000))
    refine_step = float(opt.get("cbm_refine_step_days", 2))
    refine_half_width = float(opt.get("cbm_refine_half_width_days", 20))
    refined_grid = build_refined_grid(best_coarse_thr, step=refine_step, half_width=refine_half_width)

    print("\n--- Stage 2: Refined grid ---")
    print(f"best coarse threshold = {best_coarse_thr:.1f}")
    print(f"refined grid = [{refined_grid[0]:.1f}..{refined_grid[-1]:.1f}] step {refine_step}")
    print(f"n_runs per point = {refine_n_runs}")

    df_refined = run_grid(
        scenario=scenario,
        thresholds=refined_grid,
        n_runs=refine_n_runs,
        base_seed=base_seed,
        risk_lambda=risk_lambda,
    )

    best_row = df_refined.loc[df_refined["objective"].idxmin()]

    # ---- Save results + plots
    out_dir = Path("outputs/optimization")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_coarse.to_csv(out_dir / "cbm_grid_coarse.csv", index=False)
    df_refined.to_csv(out_dir / "cbm_grid_refined.csv", index=False)

    plt.figure()
    plt.plot(df_coarse["threshold_days"], df_coarse["objective"], marker="o")
    plt.xlabel("CBM threshold_days")
    plt.ylabel("Objective")
    plt.title("CBM coarse grid: objective vs threshold")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_objective_coarse.png")
    plt.close()

    plt.figure()
    plt.plot(df_refined["threshold_days"], df_refined["objective"], marker="o")
    plt.xlabel("CBM threshold_days")
    plt.ylabel("Objective")
    plt.title("CBM refined grid: objective vs threshold")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_objective_refined.png")
    plt.close()

    # Useful diagnostic plots
    plt.figure()
    plt.plot(df_refined["threshold_days"], df_refined["p_fail"], marker="o")
    plt.xlabel("CBM threshold_days")
    plt.ylabel("P(at least one failure)")
    plt.title("CBM: failure probability vs threshold")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_p_fail_vs_threshold.png")
    plt.close()

    plt.figure()
    plt.plot(df_refined["threshold_days"], df_refined["mean_cbm_actions"], marker="o")
    plt.xlabel("CBM threshold_days")
    plt.ylabel("Mean CBM actions per horizon")
    plt.title("CBM: actions vs threshold")
    plt.tight_layout()
    plt.savefig(out_dir / "cbm_actions_vs_threshold.png")
    plt.close()

    print("\n=== BEST CBM THRESHOLD (REFINED) ===")
    print(f"threshold_days     = {best_row['threshold_days']:.1f}")
    print(f"mean_cost          = {best_row['mean_cost']:.2f}")
    print(f"p_fail             = {best_row['p_fail']:.4f}")
    print(f"objective          = {best_row['objective']:.2f}")
    print(f"mean_cbm_actions   = {best_row['mean_cbm_actions']:.2f}")
    print(f"mean_inspections   = {best_row['mean_inspections']:.2f}")

    print(f"\nSaved: {out_dir / 'cbm_grid_coarse.csv'}")
    print(f"Saved: {out_dir / 'cbm_grid_refined.csv'}")
    print(f"Saved: {out_dir / 'cbm_objective_coarse.png'}")
    print(f"Saved: {out_dir / 'cbm_objective_refined.png'}")
    print(f"Saved: {out_dir / 'cbm_p_fail_vs_threshold.png'}")
    print(f"Saved: {out_dir / 'cbm_actions_vs_threshold.png'}")


if __name__ == "__main__":
    main()
