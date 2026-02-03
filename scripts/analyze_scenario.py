from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario


def main() -> None:
    scenario_path = Path("data/generated/v0/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

    out_dir = Path("outputs/scenario_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SCENARIO OVERVIEW ===")
    print(f"Horizon (days): {scenario['horizon_days']}")
    print(f"Policies: {scenario.get('policies', [])}")

    rng = np.random.default_rng(42)
    n_samples = 10_000

    # --------------------------------------------------
    # Failure distribution
    # --------------------------------------------------
    failure = scenario["failure"]
    k = failure["shape_k"]
    lam = failure["scale_lambda_days"]

    failure_samples = lam * rng.weibull(k, size=n_samples)

    print("\nFailure model (Weibull):")
    print(f"  shape k = {k}")
    print(f"  scale λ = {lam} days")
    print(f"  mean TTF (sample) = {failure_samples.mean():.1f} days")

    plt.figure()
    plt.hist(failure_samples, bins=50, density=True)
    plt.xlabel("Time to failure (days)")
    plt.ylabel("Density")
    plt.title("Failure time distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_time_distribution.png")
    plt.close()

    # --------------------------------------------------
    # Repair duration
    # --------------------------------------------------
    repair = scenario["repair"]
    mean_h = repair["mean_hours"]
    sigma = repair["sigma"]

    mu = np.log(mean_h) - 0.5 * sigma**2
    repair_samples = rng.lognormal(mean=mu, sigma=sigma, size=n_samples)

    print("\nRepair model (Lognormal):")
    print(f"  mean repair time = {mean_h} h")
    print(f"  sigma = {sigma}")
    print(f"  mean (sample) = {repair_samples.mean():.2f} h")

    plt.figure()
    plt.hist(repair_samples, bins=50, density=True)
    plt.xlabel("Repair duration (hours)")
    plt.ylabel("Density")
    plt.title("Repair duration distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "repair_duration_distribution.png")
    plt.close()

    # --------------------------------------------------
    # Preventive maintenance schedule
    # --------------------------------------------------
    pm = scenario["pm"]
    interval = pm["interval_days"]
    horizon = scenario["horizon_days"]

    pm_times = np.arange(interval, horizon + 1, interval)

    print("\nPreventive maintenance:")
    print(f"  interval = {interval} days")
    print(f"  number of PM actions = {len(pm_times)}")

    plt.figure()
    plt.eventplot(pm_times, orientation="horizontal")
    plt.xlabel("Time (days)")
    plt.yticks([])
    plt.title("Planned PM events over horizon")
    plt.tight_layout()
    plt.savefig(out_dir / "pm_schedule.png")
    plt.close()

    print(f"\nSaved plots to: {out_dir.resolve()}")

    # --------------------------------------------------
    # Failure distribution + PM overlay
    # --------------------------------------------------
    failure = scenario["failure"]
    k = failure["shape_k"]
    lam = failure["scale_lambda_days"]

    pm = scenario["pm"]
    interval = pm["interval_days"]
    horizon = scenario["horizon_days"]
    pm_times = np.arange(interval, horizon + 1, interval)

    print("\nFailure model (Weibull):")
    print(f"  shape k = {k}")
    print(f"  scale λ = {lam} days")
    print(f"  mean TTF (sample) = {failure_samples.mean():.1f} days")

    plt.figure()
    plt.hist(failure_samples, bins=50, density=True, alpha=0.7)

    # Overlay PM times
    for t_pm in pm_times:
        plt.axvline(t_pm, linestyle="--", linewidth=1, alpha=0.5)

    plt.xlabel("Time to failure (days)")
    plt.ylabel("Density")
    plt.title("Failure time distribution with PM schedule overlay")
    plt.tight_layout()
    plt.savefig(out_dir / "failure_time_with_pm_overlay.png")
    plt.close()


if __name__ == "__main__":
    main()
