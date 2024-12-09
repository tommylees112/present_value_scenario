"""Monte Carlo simulation for career change analysis."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..core.career_model import calculate_career_paths
from ..utils.plot_utils import thousands_formatter
from .base import CareerAnalysis
from .distributions import ParameterDistribution
from .plots import plot_parameter_distributions


class MonteCarloAnalysis(CareerAnalysis):
    """Performs Monte Carlo simulation to analyze career change outcomes under uncertainty."""

    def __init__(
        self,
        parameter_distributions: Dict[str, ParameterDistribution],
        fixed_parameters: Dict[str, Any],
        time_horizons: List[int],
        course_costs: List[float],
        n_simulations: int = 1000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo analysis.

        Parameters:
        -----------
        parameter_distributions: Dictionary mapping parameter names to their distributions
        fixed_parameters: Base parameters for the analysis
        time_horizons: List of time periods to analyze
        course_costs: List of course costs to analyze
        n_simulations: Number of Monte Carlo simulations to run
        random_seed: Random seed for reproducibility
        """
        super().__init__(fixed_parameters, time_horizons, course_costs)

        # Create random state if seed provided
        if random_seed is not None:
            random_state = np.random.RandomState(random_seed)
            # Set random state for all parameter distributions
            for dist in parameter_distributions.values():
                dist.random_state = random_state

        self.parameter_distributions = parameter_distributions
        self.n_simulations = n_simulations
        self.simulation_results = {}
        self.summary_statistics = None

    def run_simulation(self):
        """Run Monte Carlo simulations for each scenario."""
        for years in self.time_horizons:
            for cost in self.course_costs:
                key = (years, cost)
                simulation_results = []

                # Generate parameter samples for each simulation
                parameter_samples = {
                    name: dist.sample(self.n_simulations)
                    for name, dist in self.parameter_distributions.items()
                }

                for i in range(self.n_simulations):
                    # Get parameter values for this simulation
                    param_values = {
                        name: parameter_samples[name][i] for name in parameter_samples
                    }

                    # Combine sampled and fixed parameters for this simulation
                    params = {
                        **self.fixed_parameters,
                        **param_values,
                        "analysis_years": years,
                        "course_annual_cost": cost,
                    }

                    # Calculate results for this simulation
                    _, summary = calculate_career_paths(**params)

                    # Store both results and parameter values
                    result_dict = {
                        **summary,
                        **param_values,  # Include parameter values in results
                    }
                    simulation_results.append(result_dict)

                # Create DataFrame
                df = pd.DataFrame(simulation_results)
                self.simulation_results[key] = df

            self._calculate_summary_statistics()

    def run_analysis(self, **kwargs) -> Dict[str, Any]:
        """Run the Monte Carlo analysis."""
        self.run_simulation()
        return self.simulation_results

    def _calculate_summary_statistics(self):
        """Calculate summary statistics across all simulations."""
        summaries = []

        for (years, cost), df in self.simulation_results.items():
            summary = {
                "time_horizon": years,
                "course_cost": cost,
                "mean_cost": df["total_nominal_cost"].mean(),
                "std_dev": df["total_nominal_cost"].std(),
                "percentile_5": df["total_nominal_cost"].quantile(0.05),
                "median": df["total_nominal_cost"].median(),
                "percentile_95": df["total_nominal_cost"].quantile(0.95),
                "probability_profitable": (df["total_nominal_cost"] < 0).mean(),
            }
            summaries.append(summary)

        self.summary_statistics = pd.DataFrame(summaries)

    def get_results_summary(self) -> pd.DataFrame:
        """Return a summary of the Monte Carlo analysis results."""
        if self.summary_statistics is None:
            raise ValueError("Must run simulation before getting summary")
        return self.summary_statistics

    def plot_distribution(self, metric: str):
        """Plot the distribution of outcomes for a specific metric."""
        if not self.simulation_results:
            raise ValueError("Must run simulation before plotting")

        # Extract values for the given metric across all time horizons and costs
        all_values = []
        for df in self.simulation_results.values():
            all_values.extend(df[metric.lower()].values)

        plt.figure(figsize=(10, 6))
        sns.histplot(all_values, kde=True)
        if metric == "total_nominal_cost":
            plt.title(
                "Distribution of Total Nominal Cost\n"
                "nominal_cost = (current_career_earnings - new_career_earnings + course_costs)\n"
                "+ve = LOSING money, -ve = MAKING money"
            )
        else:
            plt.title(f"Distribution of {metric}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")

        # Add percentile lines
        percentiles = [5, 50, 95]
        colors = ["r", "g", "r"]
        styles = ["--", "-", "--"]

        for p, c, s in zip(percentiles, colors, styles):
            percentile_value = np.percentile(all_values, p)
            plt.axvline(
                percentile_value,
                color=c,
                linestyle=s,
                label=f"{p}th percentile: {percentile_value:,.0f}",
            )

        plt.gca().axvline(x=0, color="black", linestyle="--", lw=2)

        # Add annotations for making/losing money regions
        plt.text(
            -plt.gca().get_xlim()[1] * 0.75,
            plt.gca().get_ylim()[1] * 0.5,
            "Making money\nafter studying",
            color="green",
            rotation=90,
            verticalalignment="center",
        )
        plt.text(
            plt.gca().get_xlim()[1] * 0.95,
            plt.gca().get_ylim()[1] * 0.5,
            "Losing money\nafter studying",
            color="red",
            rotation=90,
            verticalalignment="center",
        )
        plt.legend()
        plt.grid(True)
        sns.despine()
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        return plt.gcf()

    def calculate_parameter_sensitivities(
        self, metric: str
    ) -> Dict[str, Dict[str, float]]:
        """Calculate parameter sensitivities using multiple metrics.

        Returns a dictionary with different sensitivity measures:
        - correlation: Pearson correlation coefficient
        - elasticity: % change in output per % change in input
        - impact: Absolute change in output for a 10% change in input
        """
        sensitivities = {}

        # Combine all results for analysis
        all_values = []
        all_parameters = {name: [] for name in self.parameter_distributions.keys()}

        # Collect values from all simulations
        for df in self.simulation_results.values():
            metric_values = df[metric.lower()].values
            all_values.extend(metric_values)

            for param_name in self.parameter_distributions.keys():
                param_values = df[param_name].values
                all_parameters[param_name].extend(param_values)

        all_values = np.array(all_values)

        for param_name, param_values in all_parameters.items():
            param_values = np.array(param_values)

            # Calculate correlation
            correlation = np.corrcoef(param_values, all_values)[0, 1]

            # Calculate elasticity (% change in output per % change in input)
            # Using linear regression on log-transformed values
            valid_mask = (param_values > 0) & (all_values != 0)  # Avoid log(0)
            if valid_mask.any():
                log_params = np.log(param_values[valid_mask])
                log_values = np.log(np.abs(all_values[valid_mask]))
                elasticity = np.polyfit(log_params, log_values, 1)[0]
            else:
                elasticity = np.nan

            # Calculate impact of 10% change
            param_mean = np.mean(param_values)
            param_shift = param_mean * 0.1  # 10% shift

            # Estimate impact using local gradient
            gradient = np.polyfit(param_values, all_values, 1)[0]
            impact = gradient * param_shift

            sensitivities[param_name] = {
                "correlation": correlation,
                "elasticity": elasticity,
                "impact_10pct": impact,
                "mean_value": param_mean,
            }

        return sensitivities

    def plot_parameter_sensitivity(self, metric: str):
        """Create enhanced tornado plot showing parameter sensitivities."""
        if not self.simulation_results:
            raise ValueError("Must run simulation before plotting")

        sensitivities = self.calculate_parameter_sensitivities(metric)

        # Create figure with multiple subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

        # Sort parameters by absolute correlation
        sorted_params = sorted(
            sensitivities.keys(),
            key=lambda x: abs(sensitivities[x]["correlation"]),
            reverse=True,
        )

        # Plot correlations
        correlations = [sensitivities[p]["correlation"] for p in sorted_params]
        colors = ["red" if c < 0 else "green" for c in correlations]
        ax1.barh(sorted_params, correlations, color=colors)
        ax1.set_title("Parameter Correlations")
        ax1.set_xlabel("Correlation Coefficient")

        # Plot elasticities
        elasticities = [sensitivities[p]["elasticity"] for p in sorted_params]
        colors = ["red" if e < 0 else "green" for e in elasticities]
        ax2.barh(sorted_params, elasticities, color=colors)
        ax2.set_title(
            "Parameter Elasticities\n(% change in output per % change in input)"
        )
        ax2.set_xlabel("Elasticity")

        # Plot 10% impact
        impacts = [sensitivities[p]["impact_10pct"] for p in sorted_params]
        colors = ["red" if i < 0 else "green" for i in impacts]
        ax3.barh(sorted_params, impacts, color=colors)
        ax3.set_title("Impact of 10% Parameter Increase")
        ax3.set_xlabel(f"Change in {metric}")
        ax3.ticklabel_format(style="plain", axis="x")

        plt.tight_layout()
        return fig

    def visualize_results(self, **kwargs):
        """Create visualizations of Monte Carlo simulation results."""
        self.plot_distribution("total_nominal_cost")
        self.plot_parameter_sensitivity("total_nominal_cost")
        plot_parameter_distributions(self.parameter_distributions)
        plt.show()
