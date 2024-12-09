"""Tests for the career model implementation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.career_model import calculate_career_paths as core_calculate_career_paths

GROWTH_RATE = 0.05
ANALYSIS_YEARS = 10
CURRENT_STARTING_SALARY = 1
NEW_CAREER_STARTING_SALARY = 1
COURSE_ANNUAL_COST = 1
COURSE_DURATION_YEARS = 3
PART_TIME_EARNINGS = 0


def exp_growth(init, r, yr):
    return init * (1 + r) ** yr


def test_career_model_matches_reference_implementation():
    """Test that the core career model matches the reference implementation."""
    # Test parameters
    params = {
        "current_starting_salary": CURRENT_STARTING_SALARY,
        "current_growth_rate": GROWTH_RATE,
        "retrain_start_date": "2025-09-01",
        "course_duration_years": COURSE_DURATION_YEARS,
        "course_annual_cost": COURSE_ANNUAL_COST,
        "part_time_earnings": PART_TIME_EARNINGS,
        "new_career_starting_salary": NEW_CAREER_STARTING_SALARY,
        "new_career_growth_rate": GROWTH_RATE,
        "analysis_years": ANALYSIS_YEARS,
    }

    # calculate the results manually to compare against

    current_career = np.array(
        [
            exp_growth(
                params["current_starting_salary"], params["current_growth_rate"], yr
            )
            for yr in range(ANALYSIS_YEARS)
        ]
    )
    new_career_ = np.array(
        [
            exp_growth(
                params["new_career_starting_salary"],
                params["new_career_growth_rate"],
                yr,
            )
            for yr in range(ANALYSIS_YEARS - params["course_duration_years"])
        ]
    )
    new_career = np.concatenate(
        [
            np.array([0] * params["course_duration_years"]),
            np.array(new_career_),
        ]
    )
    course_costs = np.concatenate(
        [
            np.array([params["course_annual_cost"]] * params["course_duration_years"]),
            np.array([0] * (ANALYSIS_YEARS - params["course_duration_years"])),
        ]
    )
    _df = pd.DataFrame(
        {
            "current_career": current_career,
            "new_career": new_career,
            "course_costs": course_costs,
        }
    )

    ref_summary = {
        "total_nominal_cost": current_career.sum()
        - new_career.sum()
        + course_costs.sum(),
        "break_even": False,
        "years_to_break_even": None,
    }

    # Calculate using both implementations
    core_results, core_summary = core_calculate_career_paths(**params)

    # Test that the nominal costs match
    np.testing.assert_almost_equal(
        ref_summary["total_nominal_cost"],
        core_summary["total_nominal_cost"],
        decimal=2,
    )

    # Test that the year-by-year calculations match
    pd.testing.assert_series_equal(
        _df["current_career"],
        core_results["current_career"],
        check_names=False,
        check_index=False,
    )

    pd.testing.assert_series_equal(
        _df["new_career"],
        core_results["new_career"],
        check_names=False,
        check_index=False,
    )

    # Test that course costs are correct
    expected_course_costs = pd.Series(
        [1] * 3 + [0.0] * (params["analysis_years"] - 3), name="course_costs"
    )
    pd.testing.assert_series_equal(
        core_results["course_costs"],
        expected_course_costs,
        check_names=False,
    )

    # Test that net earnings during training are negative
    assert all(core_results["course_costs"].iloc[:3] > 0)

    assert all(core_results["new_career_total"].iloc[:3] < 0), (
        "Net earnings during training should be negative when "
        "course_annual_cost > part_time_earnings"
    )


def test_career_model_edge_cases():
    """Test edge cases and boundary conditions."""
    # Test with zero course costs
    params = {
        "current_starting_salary": 50000,
        "current_growth_rate": 0.05,
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 0,  # Zero course costs
        "part_time_earnings": 20000,
        "new_career_starting_salary": 100000,
        "new_career_growth_rate": 0.05,
        "analysis_years": 30,
    }
    results, summary = core_calculate_career_paths(**params)
    assert all(results["course_costs"] == 0), "Course costs should be zero"

    # Test with equal salaries and growth rates
    params.update(
        {
            "current_starting_salary": 50000,
            "new_career_starting_salary": 50000,
            "current_growth_rate": 0.05,
            "new_career_growth_rate": 0.05,
        }
    )
    results, summary = core_calculate_career_paths(**params)
    # Should show a loss due to training period with lower earnings
    assert (
        summary["total_nominal_cost"] > 0
    ), "Should show a loss with equal salaries due to training period"

    # test with large part time earnings
    params.update(
        {
            "current_starting_salary": 10_000,
            "course_annual_cost": 0,
            "part_time_earnings": 100_000,
        }
    )
    results, summary = core_calculate_career_paths(**params)
    assert all(results["new_career_total"] > 0), "Should show positive earnings"
    assert (
        summary["total_nominal_cost"] < 0
    ), "Should show a total profit (negative nominal cost)"
    assert summary["break_even"], "Should show a break even point"
    assert (
        summary["years_to_break_even"] == 0
    ), "Should show a break even point immediately"


def test_positive_earnings_during_training():
    """Test that there are POSITIVE earnings when course costs are less than part-time earnings."""
    params = {
        "current_starting_salary": 50_000,
        "current_growth_rate": 0.05,
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 25_000,
        "part_time_earnings": 30_000,
        "new_career_starting_salary": 100_000,
        "new_career_growth_rate": 0.05,
        "analysis_years": 30,
        "inflation_rate": 0.02,
    }

    current_career = np.array(
        [
            exp_growth(
                params["current_starting_salary"], params["current_growth_rate"], yr
            )
            for yr in range(ANALYSIS_YEARS)
        ]
    )

    new_career = np.array(
        [
            exp_growth(
                params["new_career_starting_salary"],
                params["new_career_growth_rate"],
                yr,
            )
            for yr in range(ANALYSIS_YEARS - params["course_duration_years"])
        ]
    )

    course_costs = np.concatenate(
        [
            np.array(
                [params["course_annual_cost"] - params["part_time_earnings"]]
                * params["course_duration_years"]
            ),
            np.array([0] * (ANALYSIS_YEARS - params["course_duration_years"])),
        ]
    )

    # Calculate using both implementations
    core_results, core_summary = core_calculate_career_paths(**params)

    # Calculate net earnings during training (earnings - costs)
    ref_net_earnings = current_career - course_costs

    # First n years should be POSITIVE
    assert all(core_results["new_career_total"][:3] > 0), (
        "Net earnings during training should be positive when "
        "course_annual_cost < part_time_earnings"
    )


if __name__ == "__main__":
    pytest.main([__file__])
