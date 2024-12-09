from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class CareerModelParams:
    """Parameters for career change financial model."""

    # Timing parameters
    retrain_start_date: datetime
    course_duration_years: float
    analysis_years: int

    # Salary parameters
    current_starting_salary: float
    new_career_starting_salary: float
    part_time_earnings: float = 0.0

    # Growth rates
    current_growth_rate: float = 0.05
    new_career_growth_rate: float = 0.05
    inflation_rate: float = 0.02
    course_annual_cost: float = 0.0

    @classmethod
    def from_dict(cls, params_dict: Dict) -> "CareerModelParams":
        """Create parameters from dictionary."""
        if isinstance(params_dict.get("retrain_start_date"), str):
            params_dict["retrain_start_date"] = datetime.strptime(
                params_dict["retrain_start_date"], "%Y-%m-%d"
            )
        return cls(**params_dict)

    def validate(self) -> None:
        """Validate parameter values."""
        self._validate_rates()
        self._validate_positive_values()
        self._validate_non_negative_values()

    def _validate_rates(self) -> None:
        """Validate rate parameters are between 0 and 1."""
        rate_params = {
            "current_growth_rate": "Annual growth rate for current career salary",
            "new_career_growth_rate": "Annual growth rate for new career salary",
            "inflation_rate": "Annual inflation rate",
        }
        for param, description in rate_params.items():
            value = getattr(self, param)
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{description} ({param}) must be between 0 and 1. "
                    f"Provided value: {value}"
                )

    def _validate_positive_values(self) -> None:
        """Validate parameters that must be positive."""
        positive_params = {
            "course_duration_years": "Duration of retraining course in years",
            "analysis_years": "Number of years to analyze",
            "current_starting_salary": "Starting salary in current career",
            "new_career_starting_salary": "Starting salary in new career",
        }
        for param, description in positive_params.items():
            value = getattr(self, param)
            if value <= 0:
                raise ValueError(
                    f"{description} ({param}) must be positive. "
                    f"Provided value: {value}"
                )

    def _validate_non_negative_values(self) -> None:
        """Validate parameters that must be non-negative."""
        non_negative_params = {
            "part_time_earnings": "Earnings while retraining",
            "course_annual_cost": "Annual cost of retraining course",
        }
        for param, description in non_negative_params.items():
            value = getattr(self, param)
            if value < 0:
                raise ValueError(
                    f"{description} ({param}) must be non-negative. "
                    f"Provided value: {value}"
                )
