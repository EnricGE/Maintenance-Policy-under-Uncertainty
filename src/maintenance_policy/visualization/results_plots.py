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
    """
    Risk vs reward plot:
      x = expected total cost
      y = probability of at least one failure
    """
    summary = (
        df.groupby("policy")
        .agg(
            mean_cost=("total_cost", "mean"),
            prob_ge_1_failure=("num_failures", lambda s: (s >= 1).mean()),
        )
        .reset_index()
    )

    plt.figure()

    plt.scatter(
        summary["mean_cost"],
        summary["prob_ge_1_failure"],
    )

    for _, row in summary.iterrows():
        plt.annotate(
            row["policy"],
            (row["mean_cost"], row["prob_ge_1_failure"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    plt.xlabel("Expected total cost")
    plt.ylabel("P(at least one failure)")
    plt.title("Risk–reward comparison of maintenance policies")
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "risk_reward.png")
    plt.close()
