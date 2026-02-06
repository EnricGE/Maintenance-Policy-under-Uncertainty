from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate
from maintenance_policy.visualization.results_plots import plot_boxplots, plot_risk_reward


def _best_from_optimization_csv(path: Path, param_col: str) -> float:
    """
    Read a refined optimization CSV and return the parameter value for the minimum objective row.
    Expected columns: objective, <param_col>
    """
    df = pd.read_csv(path)
    if "objective" not in df.columns:
        raise ValueError(f"Missing 'objective' column in {path}")
    if param_col not in df.columns:
        raise ValueError(f"Missing '{param_col}' column in {path}")
    best_row = df.loc[df["objective"].idxmin()]
    return float(best_row[param_col])


def _run_mc(policy: str, scenario: dict, n_runs: int, seed: int) -> pd.DataFrame:
    """
    Run Monte Carlo for a single policy using your existing simulate().
    Uses common random numbers: seed+i for run i.
    """
    rows: list[dict] = []
    for i in range(n_runs):
        rng = np.random.default_rng(seed + i)
        out = simulate(policy=policy, scenario=scenario, rng=rng)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    # ---- Inputs
    scenario_path = Path("data/generated/v2/scenario.json")
    scenario = load_scenario(scenario_path)

    # Use same MC settings you already have (or override via scenario["optimization"]["compare_n_runs"])
    mc = scenario.get("monte_carlo", {}) or {}
    opt = scenario.get("optimization", {}) or {}

    n_runs = int(opt.get("compare_n_runs", mc.get("n_runs", 5000)))
    seed = int(opt.get("seed", mc.get("seed", 42)))

    # Read best parameters from your existing optimization outputs
    tbm_refined_csv = Path("outputs/optimization/tbm_grid_refined.csv")
    cbm_refined_csv = Path("outputs/optimization/cbm_grid_refined.csv")

    best_tbm_interval = int(_best_from_optimization_csv(tbm_refined_csv, "pm_interval_days"))
    best_cbm_threshold = float(_best_from_optimization_csv(cbm_refined_csv, "threshold_days"))

    print("\n=== Compare optimized policies ===")
    print(f"Scenario: {scenario_path.resolve()}")
    print(f"n_runs: {n_runs}  seed: {seed}")
    print(f"TBM best interval_days: {best_tbm_interval}")
    print(f"CBM best threshold_days: {best_cbm_threshold}")

    # ---- Build policy-specific scenarios (reuse same base scenario)
    scenario_rtf = deepcopy(scenario)

    scenario_tbm = deepcopy(scenario)
    scenario_tbm["pm"]["interval_days"] = best_tbm_interval

    scenario_cbm = deepcopy(scenario)
    scenario_cbm["cbm"]["threshold_days"] = best_cbm_threshold

    # ---- Monte Carlo runs
    df_rtf = _run_mc("RTF", scenario_rtf, n_runs=n_runs, seed=seed)
    df_tbm = _run_mc("TBM", scenario_tbm, n_runs=n_runs, seed=seed)
    df_cbm = _run_mc("CBM", scenario_cbm, n_runs=n_runs, seed=seed)

    df = pd.concat([df_rtf, df_tbm, df_cbm], ignore_index=True)

    # ---- Outputs
    out_dir = Path("outputs/compare_optimized")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "compare_runs.csv", index=False)

    # Reuse your existing plotting utilities
    plot_risk_reward(df, out_dir)

    plot_boxplots(
        df,
        metric="total_cost",
        ylabel="Total cost",
        title="Total cost (optimized policies)",
        out_path=out_dir / "cost_boxplot_optimized.png",
    )

    plot_boxplots(
        df,
        metric="downtime_hours",
        ylabel="Downtime (hours)",
        title="Downtime (optimized policies)",
        out_path=out_dir / "downtime_boxplot_optimized.png",
    )

    # Decision summary table (very useful for README / decision brief)
    summary = (
        df.groupby("policy")
        .agg(
            mean_cost=("total_cost", "mean"),
            p50_cost=("total_cost", lambda x: float(np.percentile(x, 50))),
            p95_cost=("total_cost", lambda x: float(np.percentile(x, 95))),
            p_fail=("num_failures", lambda s: float((s >= 1).mean())),
            mean_downtime=("downtime_hours", "mean"),
            p95_downtime=("downtime_hours", lambda x: float(np.percentile(x, 95))),
            mean_failures=("num_failures", "mean"),
            mean_pm=("num_pm", "mean"),
            mean_cbm_actions=("num_cbm_actions", "mean") if "num_cbm_actions" in df.columns else ("num_pm", "mean"),
            mean_inspections=("num_inspections", "mean") if "num_inspections" in df.columns else ("num_pm", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(out_dir / "compare_summary.csv", index=False)

    print("\n=== Summary (saved to outputs/compare_optimized/compare_summary.csv) ===")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"  {out_dir / 'compare_runs.csv'}")
    print(f"  {out_dir / 'compare_summary.csv'}")
    print(f"  {out_dir / 'risk_reward_p95.png'} (or similarly named, depending on your plot function)")
    print(f"  {out_dir / 'cost_boxplot_optimized.png'}")
    print(f"  {out_dir / 'downtime_boxplot_optimized.png'}")


if __name__ == "__main__":
    main()
