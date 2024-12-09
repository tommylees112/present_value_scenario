"""Core career change financial model."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CareerModelParams:
    """Parameters for career change financial model."""

    # Timing parameters
    retrain_start_date: str
    course_duration_years: float
    analysis_years: int

    # Salary parameters
    current_starting_salary: float
    new_career_starting_salary: float
    part_time_earnings: float = 0.0

    # Growth rates
    current_growth_rate: float = 0.05
    new_career_growth_rate: float = 0.05

    # Financial parameters
    inflation_rate: float = 0.02
    course_annual_cost: float = 0.0

    def __post_init__(self):
        """Convert and validate parameters after initialization."""
        # Convert date string to datetime
        if isinstance(self.retrain_start_date, str):
            try:
                self.retrain_start_date = datetime.strptime(
                    self.retrain_start_date, "%Y-%m-%d"
                )
            except ValueError as e:
                raise ValueError(
                    f"retrain_start_date must be in format YYYY-MM-DD. "
                    f"Provided value: {self.retrain_start_date}"
                ) from e

        # Define rate parameters and their descriptions
        rate_params = {
            "current_growth_rate": "Annual growth rate for current career salary",
            "new_career_growth_rate": "Annual growth rate for new career salary",
            "inflation_rate": "Annual inflation rate",
        }

        # Validate rate parameters
        for param, description in rate_params.items():
            try:
                self._validate_rate(param)
            except ValueError as e:
                raise ValueError(
                    f"{description} ({param}) must be between 0 and 1. "
                    f"Provided value: {getattr(self, param)}"
                ) from e

        # Define positive parameters and their descriptions
        positive_params = {
            "course_duration_years": "Duration of retraining course in years",
            "analysis_years": "Number of years to analyze",
            "current_starting_salary": "Starting salary in current career",
            "new_career_starting_salary": "Starting salary in new career",
        }

        # Validate positive parameters
        for param, description in positive_params.items():
            try:
                self._validate_positive(param)
            except ValueError as e:
                raise ValueError(
                    f"{description} ({param}) must be positive. "
                    f"Provided value: {getattr(self, param)}"
                ) from e

        # Define non-negative parameters and their descriptions
        non_negative_params = {
            "part_time_earnings": "Earnings while retraining",
            "course_annual_cost": "Annual cost of retraining course",
        }

        # Validate non-negative parameters
        for param, description in non_negative_params.items():
            try:
                self._validate_non_negative(param)
            except ValueError as e:
                raise ValueError(
                    f"{description} ({param}) must be non-negative. "
                    f"Provided value: {getattr(self, param)}"
                ) from e

    def _validate_rate(self, field_name: str):
        """Validate that a rate is between 0 and 1."""
        value = getattr(self, field_name)
        if not 0 <= value <= 1:
            raise ValueError(
                f"{field_name} must be between 0 and 1. " f"Provided value: {value}"
            )

    def _validate_positive(self, field_name: str):
        """Validate that a value is positive."""
        value = getattr(self, field_name)
        if value <= 0:
            raise ValueError(
                f"{field_name} must be positive (> 0). " f"Provided value: {value}"
            )

    def _validate_non_negative(self, field_name: str):
        """Validate that a value is non-negative."""
        value = getattr(self, field_name)
        if value < 0:
            raise ValueError(
                f"{field_name} must be non-negative (>= 0). " f"Provided value: {value}"
            )


class CareerModel:
    """Financial model for analyzing career change decisions."""

    def __init__(self, params: CareerModelParams):
        """Initialize the career model with parameters."""
        self.params = params
        self.results = None
        self.summary = None

    def _calculate_current_career(self) -> pd.Series:
        """Calculate earnings for current career path."""
        years = np.arange(self.params.analysis_years)
        return (
            self.params.current_starting_salary
            * (1 + self.params.current_growth_rate) ** years
        )

    def _calculate_new_career(self) -> pd.Series:
        """Calculate earnings for new career path."""
        # Study period
        study_years = np.arange(self.params.course_duration_years)
        study_earnings = pd.Series(self.params.part_time_earnings, index=study_years)

        # Working period
        work_years = np.arange(
            self.params.course_duration_years, self.params.analysis_years
        )
        work_earnings = self.params.new_career_starting_salary * (
            1 + self.params.new_career_growth_rate
        ) ** (work_years - self.params.course_duration_years)

        return pd.concat([study_earnings, pd.Series(work_earnings)])

    def _calculate_course_costs(self) -> pd.Series:
        """Calculate course costs over time."""
        costs = np.zeros(self.params.analysis_years)
        costs[: int(self.params.course_duration_years)] = self.params.course_annual_cost
        return pd.Series(costs)

    def _calculate_part_time_earnings(self) -> pd.Series:
        """Calculate part-time earnings over time."""

        pt_uni = pd.Series(
            self.params.part_time_earnings,
            index=np.arange(self.params.course_duration_years),
        )
        pt_zeros = pd.Series(
            0,
            index=np.arange(
                self.params.analysis_years - self.params.course_duration_years
            ),
        )
        # pad with zeros to same length as self.params.analysis_years
        part_time_earnings = pd.concat(
            [
                pt_uni,
                pt_zeros,
            ]
        ).reset_index(drop=True)
        return part_time_earnings

    def calculate(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculate career change financial outcomes.

        Returns:
        --------
        results: DataFrame with year-by-year cash flows
        summary: Dictionary of key metrics
        """
        # Calculate nominal cash flows
        current_earnings = self._calculate_current_career()
        new_earnings = self._calculate_new_career()
        course_costs = self._calculate_course_costs()
        part_time_earnings = self._calculate_part_time_earnings()

        # Create date range for analysis
        start_date = self.params.retrain_start_date
        dates = [
            start_date + pd.DateOffset(years=i)
            for i in range(self.params.analysis_years)
        ]

        # check that all series are the same length
        if not all(
            len(series) == self.params.analysis_years
            for series in [
                current_earnings,
                new_earnings,
                course_costs,
                part_time_earnings,
            ]
        ):
            raise ValueError("All series must be the same length")

        # Compile results
        self.results = pd.DataFrame(
            {
                "date": dates,
                "year": np.arange(self.params.analysis_years),
                "current_career": current_earnings,
                "new_career": new_earnings.to_numpy(),
                "course_costs": course_costs.to_numpy(),
                "part_time_earnings": part_time_earnings.to_numpy(),
                "new_career_total": new_earnings.to_numpy() - course_costs.to_numpy(),
            }
        )

        # Calculate total nominal cost
        nominal_cost = current_earnings.sum() - new_earnings.sum() + course_costs.sum()

        self.summary = {
            "total_nominal_cost": nominal_cost,
            "break_even": nominal_cost < 0,
            "years_to_break_even": self._calculate_breakeven_year(),
        }

        return self.results, self.summary

    def _calculate_breakeven_year(self) -> Optional[int]:
        """Calculate the year when cumulative cash flow becomes positive."""
        if self.results is None:
            return None

        cumulative = (
            self.results["new_career"]
            - self.results["current_career"]
            - self.results["course_costs"]
        ).cumsum()

        breakeven_years = np.where(cumulative > 0)[0]
        return breakeven_years[0] if len(breakeven_years) > 0 else None


def calculate_career_paths(**kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Unified interface for career path calculations.

    Parameters:
    -----------
    **kwargs: Keyword arguments matching CareerModelParams fields

    Returns:
    --------
    results: DataFrame with year-by-year cash flows
    summary: Dictionary of key metrics
    """
    params = CareerModelParams(**kwargs)
    model = CareerModel(params)
    return model.calculate()
