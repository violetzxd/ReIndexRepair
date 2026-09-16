import math

from reindexrepair.metrics import (
    RepairOutcome,
    clopper_pearson_interval,
    clopper_pearson_lower,
    clopper_pearson_upper,
    minimum_zero_failure_trials,
    summarize_mirage,
)


def test_zero_of_ten_cannot_certify_five_percent_risk():
    assert 0.25 < clopper_pearson_upper(0, 10) < 0.27
    assert minimum_zero_failure_trials(0.05) == 59
    assert clopper_pearson_lower(10, 10) > 0.74
    assert clopper_pearson_lower(0, 10) == 0.0
    lower, upper = clopper_pearson_interval(3, 100)
    assert 0.006 < lower < 0.007
    assert 0.08 < upper < 0.09


def test_conditional_false_safe_rate_is_not_joint_rmr():
    outcomes = [
        RepairOutcome("i1", "q1", True, False, 0.0, 1.0),
        RepairOutcome("i2", "q2", False, False, 1.0, 1.0),
        RepairOutcome("i3", "q3", True, True, 0.0, 0.0),
        RepairOutcome("i4", "q4", False, True, 1.0, 0.0),
    ]
    summary = summarize_mirage(outcomes)
    assert summary.joint_rmr == 0.25
    assert summary.conditional_false_safe_rate == 0.5
    assert summary.repair_gap == 0.0
