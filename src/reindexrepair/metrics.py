from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    return math.fsum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes + 1)
    )


def clopper_pearson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided exact upper confidence bound for a Bernoulli rate."""
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if successes == trials:
        return 1.0
    if successes == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / trials)
    # Invert P(X <= successes | p) = 1 - confidence.  The CDF is
    # monotonically decreasing in p for fixed successes < trials.
    target = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > target:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def clopper_pearson_lower(successes: int, trials: int, confidence: float = 0.95) -> float:
    """One-sided exact lower confidence bound for a Bernoulli rate."""
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    return 1.0 - clopper_pearson_upper(trials - successes, trials, confidence)


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Equal-tailed two-sided exact interval for a Bernoulli rate."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    tail_confidence = 1.0 - (1.0 - confidence) / 2.0
    return (
        clopper_pearson_lower(successes, trials, tail_confidence),
        clopper_pearson_upper(successes, trials, tail_confidence),
    )


def minimum_zero_failure_trials(
    threshold: float, confidence: float = 0.95, max_trials: int = 100000
) -> int:
    """Smallest n for which zero failures has an exact upper bound <= threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie in (0, 1)")
    trials = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - threshold))
    if trials > max_trials:
        raise ValueError("max_trials is too small")
    return trials


@dataclass(frozen=True)
class RepairOutcome:
    incident_id: str
    query_id: str
    f0_safe: bool
    f2_safe: bool
    f0_risk: float
    f2_risk: float


@dataclass(frozen=True)
class MirageSummary:
    joint_rmr: float
    conditional_false_safe_rate: float
    repair_gap: float
    f0_safe_count: int
    observation_count: int


def summarize_mirage(outcomes: Sequence[RepairOutcome]) -> MirageSummary:
    if not outcomes:
        raise ValueError("outcomes must not be empty")
    joint = [item.f0_safe and not item.f2_safe for item in outcomes]
    f0_safe = [item for item in outcomes if item.f0_safe]
    conditional = (
        sum(not item.f2_safe for item in f0_safe) / len(f0_safe) if f0_safe else float("nan")
    )
    return MirageSummary(
        joint_rmr=float(np.mean(joint)),
        conditional_false_safe_rate=float(conditional),
        repair_gap=float(np.mean([item.f2_risk - item.f0_risk for item in outcomes])),
        f0_safe_count=len(f0_safe),
        observation_count=len(outcomes),
    )


def incident_cluster_bootstrap(
    outcomes: Sequence[RepairOutcome],
    statistic,
    *,
    samples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI resampling incidents, preserving queries within incidents."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    groups: dict[str, list[RepairOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome.incident_id].append(outcome)
    incident_ids = sorted(groups)
    if not incident_ids:
        raise ValueError("outcomes must not be empty")
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(incident_ids, size=len(incident_ids), replace=True)
        sample: list[RepairOutcome] = []
        for draw, incident_id in enumerate(selected):
            # Relabel duplicated clusters so statistics cannot accidentally merge them.
            for outcome in groups[str(incident_id)]:
                sample.append(
                    RepairOutcome(
                        f"boot-{draw}", outcome.query_id, outcome.f0_safe, outcome.f2_safe,
                        outcome.f0_risk, outcome.f2_risk,
                    )
                )
        values[index] = statistic(sample)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
