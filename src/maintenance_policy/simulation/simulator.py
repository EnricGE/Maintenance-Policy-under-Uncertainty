from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class SimResult:
    total_cost: float
    downtime_hours: float
    num_failures: int
    num_pm: int


def _sample_weibull_time_to_failure_days(rng: np.random.Generator, k: float, lam_days: float) -> float:
    """
    Weibull time-to-failure:
      T = lam * W, where W ~ Weibull(k) in numpy (scale=1).
    """
    return float(lam_days * rng.weibull(k))


def _lognormal_mu_from_mean_sigma(mean: float, sigma: float) -> float:
    """
    If X ~ LogNormal(mu, sigma), then E[X] = exp(mu + sigma^2/2).
    => mu = ln(mean) - sigma^2/2
    """
    return float(np.log(mean) - 0.5 * sigma * sigma)


def _sample_lognormal_hours(rng: np.random.Generator, mean_hours: float, sigma: float) -> float:
    mu = _lognormal_mu_from_mean_sigma(mean_hours, sigma)
    x = rng.lognormal(mean=mu, sigma=sigma)
    # Avoid pathological near-zero values for toy sims
    return float(max(x, 0.01))


def simulate(
        policy: str,
        scenario: Dict[str, Any],
        rng: np.random.Generator,
)-> Dict[str, float]:
    """
    Simulate one horizon for a single asset using a simple renewal-process model.

    Policies:
      - "RTF": run-to-failure (only corrective maintenance)
      - "TBM": time-based maintenance every pm.interval_days

    Dynamics:
      - We sample a time-to-failure after each "reset" event.
      - Under TBM, next event is min(next_failure, next_pm).
      - After PM or repair, the system is reset.

    """
    policy = policy.upper().strip()
    horizon_days = int(scenario.get("horizon_days", 365))

    # Failure model
    fail_cfg = scenario.get("failure", {})
    if fail_cfg.get("dist", "weibull").lower() != "weibull":
        raise ValueError("v1 supports only failure.dist='weibull'")
    k = float(fail_cfg.get("shape_k", 2.0))
    lam_days = float(fail_cfg.get("scale_lambda_days", 120.0))

    # Repair model
    rep_cfg = scenario.get("repair", {})
    if rep_cfg.get("dist", "lognormal").lower() != "lognormal":
        raise ValueError("v1 supports only repair.dist='lognormal'")
    repair_mean_h = float(rep_cfg.get("mean_hours", 12.0))
    repair_sigma = float(rep_cfg.get("sigma", 0.4))

    # PM model
    pm_cfg = scenario.get("pm", {})
    pm_interval_days = int(pm_cfg.get("interval_days", 30))
    pm_duration_h = float(pm_cfg.get("duration_hours", 4.0))
    pm_cost = float(pm_cfg.get("cost", 2000.0))

    # Costs
    costs = scenario.get("costs", {})
    downtime_per_h = float(costs.get("downtime_per_hour", 500.0)) # €/h
    repair_cost = float(costs.get("repair_cost", 8000.0))
    failure_penalty = float(costs.get("failure_penalty", 20000.0))

    # Counters
    t = 0.0 # current time in days
    downtime_hours = 0.0
    total_cost = 0.0
    num_failures = 0
    num_pm = 0

    # TBM schedule: PM occurs at fixed calendar times
    # We'll track the next scheduled PM day.
    if policy == "TBM":
        next_pm_day = float(pm_interval_days)
    elif policy == "RTF":
        next_pm_day = float("inf")
    else:
        raise ValueError("Unknown policy. Use 'RTF' or 'TBM'.")
    
    # Start with a sampled failure time from "now"
    next_failure_day = t + _sample_weibull_time_to_failure_days(rng, k=k, lam_days=lam_days)

    while t < horizon_days:
        next_event_day = min(next_failure_day, next_pm_day)

        if next_event_day>= horizon_days:
            #Horizon breaks before next event
            break

        t = next_event_day

        if next_failure_day <= next_pm_day:
            # FAILURE event
            num_failures += 1

            repair_duration_h = _sample_lognormal_hours(rng, mean_hours=repair_mean_h, sigma=repair_sigma)
            downtime_hours += repair_duration_h

            total_cost += repair_cost + failure_penalty + downtime_per_h*repair_duration_h

            # Reset after repair: sample next failure from now
            next_failure_day = t + _sample_weibull_time_to_failure_days(rng, k=k, lam_days=lam_days)

            # PM schedule continues on calendar (unchanged)

        else:
            # PM event (TBM only)
            num_pm += 1

            downtime_hours += pm_duration_h
            total_cost += pm_cost + downtime_per_h*pm_duration_h

            # Reset after PM
            next_failure_day = t + _sample_weibull_time_to_failure_days(rng, k=k, lam_days=lam_days)

            # Schedule next PM
            next_pm_day += float(pm_interval_days)

    return {
        "policy": policy,
        "total_cost": float(total_cost),
        "downtime_hours": float(downtime_hours),
        "num_failures": int(num_failures),
        "num_pm": int(num_pm),
    }
