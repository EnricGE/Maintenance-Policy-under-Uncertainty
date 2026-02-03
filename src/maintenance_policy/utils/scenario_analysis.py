from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from maintenance_policy.preprocessing.loaders import load_scenario


def main() -> None:
    scenario_path = Path("data/generated/v0/scenario.json").resolve()
    scenario = load_scenario(scenario_path)

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

if __name__ == "__main__":
    main()
