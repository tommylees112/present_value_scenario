"""Base class for career analysis implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class CareerAnalysis(ABC):
    def __init__(
        self,
        fixed_parameters: Dict[str, Any],
        time_horizons: list[int],
        course_costs: list[float],
    ):
        """
        Initialize base career analysis.

        Parameters:
        -----------
        fixed_parameters: Parameters that remain constant during analysis
        time_horizons: List of time periods (years) to analyze
        course_costs: List of annual course costs to analyze
        """
        self.fixed_parameters = fixed_parameters
        self.time_horizons = time_horizons
        self.course_costs = course_costs
        self.results = {}

    @abstractmethod
    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run the specific analysis implementation."""
        pass

    @abstractmethod
    def visualize_results(self, **kwargs):
        """Create visualizations of the analysis results."""
        pass

    def get_results_summary(self) -> pd.DataFrame:
        """Return a summary of the analysis results."""
        pass
