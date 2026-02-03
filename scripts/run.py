from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from maintenance_policy.preprocessing.loaders import load_scenario
from maintenance_policy.simulation.simulator import simulate
from maintenance_policy.visualization.results_plots import (
    plot_cost_distribution,
    plot_downtime_distribution,
    plot_risk_reward,
    plot_boxplots
)

plots_dir = Path("outputs/results_plots")

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def p95(x: pd.Series) -> float:
        return float(np.percentile(x.to_numpy(), 95))

    out = (
        df.groupby("policy")
        .agg(
            n_runs=("total_cost", "size"),
            mean_cost=("total_cost", "mean"),
            p95_cost=("total_cost", p95),
            mean_downtime_h=("downtime_hours", "mean"),
            p95_downtime_h=("downtime_hours", p95),
            mean_failures=("num_failures", "mean"),
            prob_ge_1_failure=("num_failures", lambda s: float((s >= 1).mean())),
        )
        .reset_index()
    )
    return out


def main() -> None:
    scenario_path = Path("data/generated/v0/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

    policies = scenario.get("policies", ["RTF", "TBM"])
    mc = scenario.get("monte_carlo", {})
    n_runs = int(mc.get("n_runs", 10_000))
    seed = int(mc.get("seed", 42))

    rng = np.random.default_rng(seed)

    rows = []
    for policy in policies:
        for _ in range(n_runs):
            rows.append(simulate(policy=policy, scenario=scenario, rng=rng))

    df = pd.DataFrame(rows)

    # Print summary
    summary = summarize(df)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 50)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    # Save outputs
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df.to_csv(out_dir / "sim_runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nSaved: {out_dir/'sim_runs.csv'}")
    print(f"Saved: {out_dir/'summary.csv'}")

    plot_cost_distribution(df, plots_dir)
    plot_downtime_distribution(df, plots_dir)
    plot_risk_reward(df, plots_dir)

    plot_boxplots(
        df,
        metric="total_cost",
        ylabel="Total cost",
        title="Total cost by maintenance policy",
        out_path=plots_dir / "cost_boxplot.png",
    )

    plot_boxplots(
        df,
        metric="downtime_hours",
        ylabel="Downtime (hours)",
        title="Downtime by maintenance policy",
        out_path=plots_dir / "downtime_boxplot.png",
    )

if __name__ == "__main__":
    main()
