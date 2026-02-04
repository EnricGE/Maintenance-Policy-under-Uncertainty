from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario


def main() -> None:
    scenario_path = Path("data/generated/v1/scenario.json").resolve()
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
    # Repair duration distribution
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

    # --------------------------------------------------
    # CBM parameters + inspection schedule
    # --------------------------------------------------
    cbm = scenario.get("cbm")
    if cbm is None:
        print("\nCBM: not present in scenario.json (no 'cbm' key).")
    else:
        inspect_interval = int(cbm["inspect_interval_days"])
        inspect_cost = float(cbm.get("inspect_cost", 0.0))
        threshold_days = float(cbm["threshold_days"])
        action_cost = float(cbm.get("action_cost", 0.0))
        action_duration_h = float(cbm.get("action_duration_hours", 0.0))

        inspect_times = np.arange(inspect_interval, horizon + 1, inspect_interval)

        print("\nCondition-Based Maintenance (CBM):")
        print(f"  inspect interval = {inspect_interval} days")
        print(f"  inspect cost = {inspect_cost}")
        print(f"  threshold (age) = {threshold_days} days")
        print(f"  action cost = {action_cost}")
        print(f"  action duration = {action_duration_h} h")
        print(f"  number of inspections over horizon = {len(inspect_times)}")

        # Inspection schedule plot
        plt.figure()
        plt.eventplot(inspect_times, orientation="horizontal")
        plt.xlabel("Time (days)")
        plt.yticks([])
        plt.title("CBM inspection events over horizon")
        plt.tight_layout()
        plt.savefig(out_dir / "cbm_inspection_schedule.png")
        plt.close()

        # Simple “age threshold” visualization
        # Age increases linearly; threshold is a horizontal line.
        t = np.arange(0, horizon + 1)
        age = t  # assumes no reset; just to show what "threshold_days" means visually

        plt.figure()
        plt.plot(t, age, label="Age since last reset (illustration)")
        plt.axhline(threshold_days, linestyle="--", label="CBM threshold")
        plt.xlabel("Time (days)")
        plt.ylabel("Age (days)")
        plt.title("CBM threshold meaning (illustration)")
        plt.tight_layout()
        plt.savefig(out_dir / "cbm_threshold_illustration.png")
        plt.close()

    print(f"\nSaved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
