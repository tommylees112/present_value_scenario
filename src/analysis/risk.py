"""Risk analysis module for career change decisions."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from ..core.career_model import CareerModelResult


class RiskLevel(Enum):
    """Risk levels for career change."""

    LOW = "Low Risk"
    MEDIUM_LOW = "Medium-Low Risk"
    MEDIUM = "Medium Risk"
    MEDIUM_HIGH = "Medium-High Risk"
    HIGH = "High Risk"


@dataclass
class RiskFactor:
    """Represents a single risk factor in the analysis."""

    name: str
    description: str
    weight: float
    threshold: float
    comparison: str  # '>', '<', '>=', '<='


class RiskAnalyzer:
    """Analyzes risk factors in career change decisions."""

    def __init__(self, model_result: CareerModelResult):
        """Initialize with career model results.

        Args:
            model_result (CareerModelResult): The result from the career model containing cash flows and summary data.

        Attributes:
            cash_flows (Dict): A dictionary containing cash flow data for the current and new career paths.
            summary (Dict): A summary of key metrics from the career model.
            risk_factors (List[RiskFactor]): A list of risk factors to be analyzed.
        """
        self.cash_flows = model_result.cash_flows
        self.summary = model_result.summary
        self.risk_factors = self._initialize_risk_factors()

    def _initialize_risk_factors(self) -> List[RiskFactor]:
        """Initialize standard risk factors for analysis."""
        return [
            RiskFactor(
                name="course_duration",
                description="Length of retraining period",
                weight=1.0,
                threshold=2.0,
                comparison=">",
            ),
            RiskFactor(
                name="negative_earnings",
                description="Periods of negative earnings",
                weight=2.0,
                threshold=0,
                comparison="<",
            ),
            RiskFactor(
                name="break_even_time",
                description="Time to break even",
                weight=1.5,
                threshold=5,
                comparison=">",
            ),
            # Add more risk factors as needed
        ]

    def calculate_risk_metrics(self) -> Dict:
        """Calculate comprehensive risk metrics.

        Returns:
            Dict: A dictionary containing various risk metrics including:
                - max_earnings_drop: Maximum drop in earnings during the transition.
                - year_of_max_earnings_drop: Year when the maximum earnings drop occurs.
                - recovery_period: Number of years to recover to original earnings level.
                - opportunity_cost: Total opportunity cost during the transition period.
                - risk_rating: Overall risk rating based on all factors.
                - risk_factors: Individual risk scores for each risk factor.
        """
        return {
            "max_earnings_drop": self._calculate_max_earnings_drop(),
            "year_of_max_earnings_drop": self._calculate_year_of_max_earnings_drop(),
            "recovery_period": self._calculate_recovery_period(),
            "opportunity_cost": self._calculate_opportunity_cost(),
            "risk_rating": self._calculate_risk_rating(),
            "risk_factors": self._analyze_individual_risk_factors(),
        }

    def _calculate_max_earnings_drop(self) -> float:
        """Calculate maximum drop in earnings during transition.

        Returns:
            float: The maximum earnings drop, calculated as the minimum difference between new and current earnings.
        """
        current_earnings = self.cash_flows["current_career"]
        new_total = self.cash_flows["new_career_total"]
        earnings_difference = new_total - current_earnings
        return abs(min(earnings_difference.min(), 0))

    def _calculate_year_of_max_earnings_drop(self) -> Optional[int]:
        """Calculate the year when the maximum earnings drop occurs.

        Returns:
            Optional[int]: The year index of the maximum earnings drop, or None if no drop occurs.
        """
        current_earnings = self.cash_flows["current_career"]
        new_total = self.cash_flows["new_career_total"]
        earnings_difference = new_total - current_earnings

        # Find the index of the minimum earnings difference
        min_index = np.argmin(earnings_difference)
        return min_index if earnings_difference[min_index] < 0 else None

    def _calculate_recovery_period(self) -> Optional[int]:
        """Calculate number of years to recover to original earnings level.

        Returns:
            Optional[int]: The first year where new earnings exceed current earnings, or None if never.
        """
        current_earnings = self.cash_flows["current_career"]
        new_total = self.cash_flows["new_career_total"]

        # Find first point where new earnings exceed current earnings
        crossover_points = np.where(new_total >= current_earnings)[0]
        return crossover_points[0] if len(crossover_points) > 0 else None

    def _calculate_opportunity_cost(self) -> float:
        """Calculate total opportunity cost during transition period.

        Returns:
            float: The total opportunity cost, calculated as the sum of
            differences between current and new earnings.
        """
        current_earnings = self.cash_flows["current_career"]
        new_total = self.cash_flows["new_career_total"]
        earnings_difference = current_earnings - new_total
        return max(earnings_difference.sum(), 0)

    def _analyze_individual_risk_factors(self) -> Dict[str, float]:
        """Analyze each risk factor individually."""
        risk_scores = {}

        for factor in self.risk_factors:
            if factor.name == "course_duration":
                value = len(self.cash_flows[self.cash_flows["course_costs"] > 0])
            elif factor.name == "negative_earnings":
                value = self.cash_flows["new_career_total"].min()
            elif factor.name == "break_even_time":
                value = self.summary.get("years_to_break_even", float("inf"))

            risk_scores[factor.name] = self._evaluate_risk_factor(factor, value)

        return risk_scores

    def _evaluate_risk_factor(self, factor, value):
        """Evaluate a single risk factor."""
        # Ensure value is not None before comparison
        if value is None:
            return 0  # or some default risk score

        comparison_ops = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y,
            "!=": lambda x, y: x != y,
        }

        if comparison_ops[factor.comparison](value, factor.threshold):
            return factor.weight
        return 0.0

    def _calculate_risk_rating(self) -> RiskLevel:
        """Calculate overall risk rating based on all factors."""
        risk_scores = self._analyze_individual_risk_factors()
        total_risk_score = sum(risk_scores.values())

        # Convert score to risk level
        if total_risk_score <= 1.0:
            return RiskLevel.LOW
        elif total_risk_score <= 2.0:
            return RiskLevel.MEDIUM_LOW
        elif total_risk_score <= 3.0:
            return RiskLevel.MEDIUM
        elif total_risk_score <= 4.0:
            return RiskLevel.MEDIUM_HIGH
        else:
            return RiskLevel.HIGH

    def get_risk_summary(self) -> Dict:
        """Get a human-readable summary of risk analysis."""
        metrics = self.calculate_risk_metrics()

        return {
            "risk_level": metrics["risk_rating"].value,
            "max_earnings_drop": f"£{metrics['max_earnings_drop']:,.2f}",
            "year_of_max_earnings_drop": (
                f"{metrics['year_of_max_earnings_drop']} years"
                if metrics["year_of_max_earnings_drop"]
                else "Never"
            ),
            "recovery_period": (
                f"{metrics['recovery_period']} years"
                if metrics["recovery_period"]
                else "Never"
            ),
            "opportunity_cost": f"£{metrics['opportunity_cost']:,.2f}",
            "risk_factors": {
                factor.description: metrics["risk_factors"][factor.name]
                for factor in self.risk_factors
            },
        }
