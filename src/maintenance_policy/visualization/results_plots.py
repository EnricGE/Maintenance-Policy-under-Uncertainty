from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_cost_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    bins: int = 50,
) -> None:
    """
    Histogram of total cost per policy.
    """
    plt.figure()

    for policy, g in df.groupby("policy"):
        plt.hist(
            g["total_cost"],
            bins=bins,
            density=True,
            alpha=0.6,
            label=policy,
        )

    plt.xlabel("Total cost")
    plt.ylabel("Density")
    plt.title("Total cost distribution by policy")
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "cost_distribution.png")
    plt.close()


def plot_downtime_distribution(
    df: pd.DataFrame,
    out_dir: Path,
    bins: int = 50,
) -> None:
    """
    Histogram of downtime per policy.
    """
    plt.figure()

    for policy, g in df.groupby("policy"):
        plt.hist(
            g["downtime_hours"],
            bins=bins,
            density=True,
            alpha=0.6,
            label=policy,
        )

    plt.xlabel("Downtime (hours)")
    plt.ylabel("Density")
    plt.title("Downtime distribution by policy")
    plt.legend()
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "downtime_distribution.png")
    plt.close()


def plot_risk_reward(
    df: pd.DataFrame,
    out_dir: Path,
) -> None:
    summary = (
        df.groupby("policy")
        .agg(
            mean_cost=("total_cost", "mean"),
            p95_cost=("total_cost", lambda x: np.percentile(x, 95)),
            prob_ge_1_failure=("num_failures", lambda s: (s >= 1).mean()),
        )
        .reset_index()
    )

    plt.figure()

    x_min, x_max = summary["mean_cost"].min(), summary["mean_cost"].max()
    y_min, y_max = summary["p95_cost"].min(), summary["p95_cost"].max()

    # Dominated region (top-right)
    plt.fill_between(
        [x_min, x_max],
        y_max,
        y_max * 1.1,
        color="red",
        alpha=0.05,
        label="Dominated region",
    )

    plt.scatter(
        summary["mean_cost"],
        summary["p95_cost"],
        s=100,
    )

    for _, row in summary.iterrows():
        label = (
            f"{row['policy']}\n"
            f"P(fail ≥1) = {row['prob_ge_1_failure']:.2f}"
        )
        plt.annotate(
            label,
            (row["mean_cost"], row["p95_cost"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Expected total cost")
    plt.ylabel("p95 total cost")
    plt.title("Risk–reward comparison (mean vs tail risk)")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "risk_reward_p95.png")
    plt.close()


def plot_boxplots(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """
    Side-by-side boxplots per policy.
    """
    data = [g[metric].values for _, g in df.groupby("policy")]
    labels = [policy for policy, _ in df.groupby("policy")]

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    for i, (_, g) in enumerate(df.groupby("policy"), start=1):
        mean_val = g[metric].mean()
        p95_val = np.percentile(g[metric], 95)

        plt.scatter(i, mean_val, color="red", zorder=3)
        plt.scatter(i, p95_val, color="green", zorder=3)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()




