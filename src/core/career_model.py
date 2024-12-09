"""Core career change financial model."""

from typing import Dict, NamedTuple, Union

import numpy as np
import pandas as pd

from .career_model_params import CareerModelParams


class CareerModelResult(NamedTuple):
    """Results from career model calculations."""

    cash_flows: pd.DataFrame
    summary: Dict[str, any]


class CareerModel:
    """Financial model for analyzing career change decisions."""

    def __init__(self, params: Union[CareerModelParams, Dict]):
        """Initialize model with parameters.

        Args:
            params: Either a CareerModelParams object or a dictionary of parameters
        """
        if isinstance(params, dict):
            self.params = CareerModelParams.from_dict(params)
        else:
            self.params = params
        self.params.validate()

    def calculate(self) -> CareerModelResult:
        """Calculate career change financial outcomes."""
        # Calculate earnings streams
        current_earnings = self._calculate_current_career()
        new_earnings = self._calculate_new_career()
        course_costs = self._calculate_course_costs()

        # Create results dataframe
        results = self._create_results_dataframe(
            current_earnings, new_earnings, course_costs
        )

        # Calculate summary metrics
        summary = self._calculate_summary_metrics(results)

        return CareerModelResult(results, summary)

    def _calculate_current_career(self) -> np.ndarray:
        """Calculate earnings for current career path."""
        years = np.arange(self.params.analysis_years)
        return (
            self.params.current_starting_salary
            * (1 + self.params.current_growth_rate) ** years
        )

    def _calculate_new_career(self) -> np.ndarray:
        """Calculate earnings for new career path."""
        # Study period earnings
        study_period = np.full(
            int(self.params.course_duration_years), self.params.part_time_earnings
        )

        # Working period earnings
        working_years = np.arange(self.params.analysis_years - len(study_period))
        working_period = (
            self.params.new_career_starting_salary
            * (1 + self.params.new_career_growth_rate) ** working_years
        )

        return np.concatenate([study_period, working_period])

    def _calculate_course_costs(self) -> np.ndarray:
        """Calculate course costs over time."""
        costs = np.zeros(self.params.analysis_years)
        costs[: int(self.params.course_duration_years)] = self.params.course_annual_cost
        return costs

    def _create_results_dataframe(
        self,
        current_earnings: np.ndarray,
        new_earnings: np.ndarray,
        course_costs: np.ndarray,
    ) -> pd.DataFrame:
        """Create DataFrame with all cash flows."""
        dates = [
            self.params.retrain_start_date + pd.DateOffset(years=i)
            for i in range(self.params.analysis_years)
        ]

        return pd.DataFrame(
            {
                "date": dates,
                "year": np.arange(self.params.analysis_years),
                "current_career": current_earnings,
                "new_career": new_earnings,
                "course_costs": course_costs,
                "new_career_total": new_earnings - course_costs,
            }
        )

    def _calculate_summary_metrics(self, results: pd.DataFrame) -> Dict:
        """Calculate summary metrics from results."""
        # Calculate total nominal cost
        nominal_cost = (
            results["current_career"].sum()
            - results["new_career"].sum()
            + results["course_costs"].sum()
        )

        # Calculate break-even metrics
        cumulative_difference = (
            results["new_career_total"] - results["current_career"]
        ).cumsum()
        breaks_even = cumulative_difference.iloc[-1] > 0

        years_to_break_even = None
        if breaks_even:
            break_even_points = np.where(cumulative_difference > 0)[0]
            if len(break_even_points) > 0:
                years_to_break_even = break_even_points[0]

        return {
            "total_nominal_cost": nominal_cost,
            "break_even": breaks_even,
            "years_to_break_even": years_to_break_even,
        }


def calculate_career_paths(
    params: Union[CareerModelParams, Dict, None] = None, **kwargs
) -> CareerModelResult:
    """
    Unified interface for career path calculations.

    Parameters:
    -----------
    params: Either a CareerModelParams object, dictionary of parameters, or None
    **kwargs: Additional keyword arguments to override params if it's a dictionary,
             or used as parameters if params is None

    Returns:
    --------
    CareerModelResult containing:
    - cash_flows: DataFrame with year-by-year cash flows
    - summary: Dictionary of key metrics
    """
    if params is None:
        params = kwargs
    elif isinstance(params, dict):
        params = {**params, **kwargs}  # Override with any provided kwargs

    if isinstance(params, dict):
        params = CareerModelParams.from_dict(params)

    model = CareerModel(params)
    return model.calculate()
