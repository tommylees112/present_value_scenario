"""Tests for the career model implementation."""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.core.career_model import calculate_career_paths

# Test constants
DEFAULT_PARAMS = {
    "current_starting_salary": 50_000,
    "current_growth_rate": 0.05,
    "retrain_start_date": "2025-09-01",
    "course_duration_years": 3,
    "course_annual_cost": 25_000,
    "part_time_earnings": 30_000,
    "new_career_starting_salary": 50_000,
    "new_career_growth_rate": 0.05,
    "analysis_years": 30,
    "inflation_rate": 0.02,
}


@pytest.fixture
def base_params() -> Dict[str, Any]:
    """Provide base parameters for testing."""
    return DEFAULT_PARAMS.copy()


def calculate_expected_earnings(
    starting_salary: float, growth_rate: float, years: int, start_year: int = 0
) -> np.ndarray:
    """Calculate expected earnings with compound growth."""
    return np.array(
        [
            starting_salary * (1 + growth_rate) ** yr
            for yr in range(start_year, start_year + years)
        ]
    )


class TestCareerModel:
    """Test suite for career model calculations."""

    def test_basic_career_paths(self, base_params):
        """Test basic career path calculations."""
        results, summary = calculate_career_paths(**base_params)

        assert isinstance(results, pd.DataFrame)
        assert isinstance(summary, dict)
        assert "total_nominal_cost" in summary
        assert "break_even" in summary
        assert "years_to_break_even" in summary

    def test_earnings_during_training(self, base_params):
        """Test that earnings during training period are correct."""
        results, _ = calculate_career_paths(**base_params)

        # First 3 years should have part-time earnings
        training_earnings = results["new_career"].iloc[:3]
        expected_earnings = np.array([base_params["part_time_earnings"]] * 3)
        np.testing.assert_array_almost_equal(training_earnings, expected_earnings)

    def test_current_career_growth(self, base_params):
        """Test that current career earnings grow correctly."""
        results, _ = calculate_career_paths(**base_params)

        expected_earnings = calculate_expected_earnings(
            base_params["current_starting_salary"],
            base_params["current_growth_rate"],
            base_params["analysis_years"],
        )

        np.testing.assert_array_almost_equal(
            results["current_career"], expected_earnings
        )

    def test_new_career_growth(self, base_params):
        """Test that new career earnings grow correctly after training."""
        results, _ = calculate_career_paths(**base_params)

        # Calculate expected earnings after training
        training_years = base_params["course_duration_years"]
        working_years = base_params["analysis_years"] - training_years

        expected_earnings = calculate_expected_earnings(
            base_params["new_career_starting_salary"],
            base_params["new_career_growth_rate"],
            working_years,
        )

        # Compare post-training earnings
        np.testing.assert_array_almost_equal(
            results["new_career"].iloc[training_years:], expected_earnings
        )

    def test_break_even_calculation(self, base_params):
        """Test break-even calculations."""
        # Set parameters to ensure breaking even
        params = base_params.copy()
        params["new_career_starting_salary"] = (
            200_000  # High salary to ensure break-even
        )

        _, summary = calculate_career_paths(**params)

        assert summary["break_even"] == True
        assert summary["years_to_break_even"] is not None
        assert summary["total_nominal_cost"] < 0  # Should be profitable

    def test_no_break_even(self, base_params):
        """Test case where career change never breaks even."""
        # Set parameters to ensure never breaking even
        params = base_params.copy()
        params["new_career_starting_salary"] = 10_000  # Too low to break even

        _, summary = calculate_career_paths(**params)

        assert summary["break_even"] == False
        assert summary["years_to_break_even"] is None
        assert summary["total_nominal_cost"] > 0  # Should be unprofitable

    def test_edge_cases(self, base_params):
        """Test edge cases and boundary conditions."""
        test_cases = [
            {
                "name": "zero_course_cost",
                "params": {"course_annual_cost": 0, "current_starting_salary": 30_000},
                "expected": {"should_break_even": True},
            },
            {
                "name": "equal_salaries",
                "params": {
                    "current_starting_salary": 50_000,
                    "new_career_starting_salary": 50_000,
                },
                "expected": {"should_break_even": False},
            },
            {
                "name": "high_part_time_earnings",
                "params": {
                    "part_time_earnings": 100_000,
                    "new_career_starting_salary": 100_000,
                },
                "expected": {"should_break_even": True},
            },
        ]

        for case in test_cases:
            params = base_params.copy()
            params.update(case["params"])

            df, summary = calculate_career_paths(**params)

            assert (
                summary["break_even"] == case["expected"]["should_break_even"]
            ), f"Failed for case: {case['name']}"


if __name__ == "__main__":
    pytest.main([__file__])
