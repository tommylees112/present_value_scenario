"""Probabilistic analysis for career change decisions."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.career_model import calculate_career_paths
from .base import CareerAnalysis


class ProbabilisticAnalysis(CareerAnalysis):
    """
    Analyzes career change outcomes using probability distributions and sensitivity analysis.
    """

    def __init__(
        self,
        fixed_parameters: Dict[str, Any],
        time_horizons: List[int],
        course_costs: List[float],
        parameter_distributions: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize probabilistic analysis.

        Parameters:
        -----------
        fixed_parameters: Base parameters for the analysis
        time_horizons: List of time periods to analyze
        course_costs: List of course costs to analyze
        parameter_distributions: Dictionary mapping parameter names to their distribution specs
            Example: {
                "current_growth_rate": {
                    "distribution": "normal",
                    "mean": 0.05,
                    "std": 0.01
                }
            }
        """
        super().__init__(fixed_parameters, time_horizons, course_costs)
        self.parameter_distributions = parameter_distributions or {}
        self.sensitivity_results = {}

    def _get_distribution(self, param_name: str) -> Any:
        """Get scipy.stats distribution object for a parameter."""
        if param_name not in self.parameter_distributions:
            return None

        dist_spec = self.parameter_distributions[param_name]
        dist_name = dist_spec.pop("distribution")
        distribution = getattr(stats, dist_name)

        return distribution(**dist_spec)

    def run_sensitivity_analysis(
        self, param_name: str, param_range: Tuple[float, float], n_points: int = 50
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis for a specific parameter.

        Parameters:
        -----------
        param_name: Name of parameter to analyze
        param_range: (min, max) values for parameter
        n_points: Number of points to evaluate
        """
        param_values = np.linspace(*param_range, n_points)
        results = []

        for value in param_values:
            params = self.fixed_parameters.copy()
            params[param_name] = value

            for years in self.time_horizons:
                for cost in self.course_costs:
                    params.update(
                        {
                            "analysis_years": years,
                            "course_annual_cost": cost,
                        }
                    )

                    _, summary = calculate_career_paths(**params)
                    results.append(
                        {
                            "parameter": param_name,
                            "value": value,
                            "time_horizon": years,
                            "course_cost": cost,
                            "total_cost": summary["total_nominal_cost"],
                        }
                    )

        return pd.DataFrame(results)

    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run probabilistic analysis including sensitivity studies."""
        # Run sensitivity analysis for each uncertain parameter
        for param_name in self.parameter_distributions:
            dist = self._get_distribution(param_name)
            if dist is None:
                continue

            # Get reasonable range for parameter (±3 standard deviations)
            param_range = (
                dist.ppf(0.001),  # 0.1th percentile
                dist.ppf(0.999),  # 99.9th percentile
            )

            self.sensitivity_results[param_name] = self.run_sensitivity_analysis(
                param_name=param_name,
                param_range=param_range,
            )

        return self.sensitivity_results

    def get_results_summary(self) -> pd.DataFrame:
        """Generate summary of sensitivity analysis results."""
        summaries = []

        for param_name, results_df in self.sensitivity_results.items():
            # Calculate elasticity (% change in output / % change in input)
            baseline_value = self.fixed_parameters.get(param_name, 0)
            baseline_cost = results_df[
                results_df["value"].abs() == abs(baseline_value)
            ]["total_cost"].iloc[0]

            for years in self.time_horizons:
                for cost in self.course_costs:
                    subset = results_df[
                        (results_df["time_horizon"] == years)
                        & (results_df["course_cost"] == cost)
                    ]

                    # Calculate elasticity using linear regression
                    x = subset["value"] / baseline_value
                    y = subset["total_cost"] / baseline_cost
                    slope, _ = np.polyfit(x, y, 1)

                    summaries.append(
                        {
                            "parameter": param_name,
                            "time_horizon": years,
                            "course_cost": cost,
                            "elasticity": slope,
                            "min_impact": subset["total_cost"].min(),
                            "max_impact": subset["total_cost"].max(),
                            "impact_range": subset["total_cost"].max()
                            - subset["total_cost"].min(),
                        }
                    )

        return pd.DataFrame(summaries)

    def visualize_results(self, **kwargs):
        """Create visualizations of sensitivity analysis results."""
        # Implementation will include:
        # - Tornado diagrams
        # - Sensitivity curves
        # - Parameter distribution plots
        # - Impact distribution plots
        pass
