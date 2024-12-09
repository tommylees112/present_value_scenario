"""Monte Carlo simulation for career change analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

from ..core.career_model import calculate_career_paths
from ..utils.plot_utils import thousands_formatter
from .base import CareerAnalysis

ALLOWED_DISTRIBUTIONS = {
    "normal": {"required_params": ["mean", "std"]},
    "truncated_normal": {"required_params": ["mean", "std", "min", "max"]},
    "lognormal": {"required_params": ["mu", "sigma"]},
    "uniform": {"required_params": ["min", "max"]},
}


@dataclass
class ParameterDistribution:
    """
    Represents a probability distribution for a model parameter.

    Parameters:
    -----------
    name: Name of the parameter
    distribution_type: The statistical distribution to use
    params: Dictionary of parameters specific to the distribution type

    Allowed Distributions:
    ---------------------
    normal: requires 'mean' and 'std' parameters
    truncated_normal: requires 'mean', 'std', 'min', and 'max' parameters
    lognormal: requires 'mu' and 'sigma' parameters
    uniform: requires 'min' and 'max' parameters
    """

    name: str
    distribution_type: str
    params: Dict[str, float]

    def __post_init__(self):
        """Validate the distribution type and parameters after initialization."""
        if self.distribution_type not in ALLOWED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution type: {self.distribution_type}. "
                f"Allowed types are: {list(ALLOWED_DISTRIBUTIONS.keys())}"
            )

        required_params = ALLOWED_DISTRIBUTIONS[self.distribution_type][
            "required_params"
        ]
        missing_params = [
            param for param in required_params if param not in self.params
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {self.distribution_type} distribution: "
                f"{missing_params}. Required parameters are: {required_params}. "
                f"Provided parameters: {list(self.params.keys())}"
            )

    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution."""
        if self.distribution_type == "normal":
            return stats.norm.rvs(
                loc=self.params["mean"], scale=self.params["std"], size=size
            )
        elif self.distribution_type == "truncated_normal":
            # Calculate the standardized bounds
            a = (self.params["min"] - self.params["mean"]) / self.params["std"]
            b = (self.params["max"] - self.params["mean"]) / self.params["std"]
            return stats.truncnorm.rvs(
                a=a,
                b=b,
                loc=self.params["mean"],
                scale=self.params["std"],
                size=size,
            )
        elif self.distribution_type == "lognormal":
            return stats.lognorm.rvs(
                s=self.params["sigma"], scale=np.exp(self.params["mu"]), size=size
            )
        elif self.distribution_type == "uniform":
            return stats.uniform.rvs(
                loc=self.params["min"],
                scale=self.params["max"] - self.params["min"],
                size=size,
            )
        else:
            # This should never happen due to __post_init__ validation
            raise ValueError(
                f"Unsupported distribution type: {self.distribution_type}. "
                f"Allowed types are: {list(ALLOWED_DISTRIBUTIONS.keys())}"
            )


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
        self.parameter_distributions = parameter_distributions
        self.n_simulations = n_simulations
        self.simulation_results = {}
        self.summary_statistics = None

        if random_seed is not None:
            np.random.seed(random_seed)

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

                self.simulation_results[key] = pd.DataFrame(simulation_results)

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

    def plot_parameter_sensitivity(self, metric: str):
        """Create a tornado plot showing parameter sensitivity."""
        if not self.simulation_results:
            raise ValueError("Must run simulation before plotting")

        parameter_impacts = {}

        # Combine all results
        all_values = []
        all_parameters = {name: [] for name in self.parameter_distributions.keys()}

        # Collect values from all simulations
        for df in self.simulation_results.values():
            metric_values = df[metric.lower()].values
            all_values.extend(metric_values)

            # For each simulation result, store the parameter values
            for param_name in self.parameter_distributions.keys():
                param_values = df[param_name].values  # Now we know these columns exist
                all_parameters[param_name].extend(param_values)

        # Calculate correlation for each parameter
        for param_name, param_values in all_parameters.items():
            param_values = np.array(param_values)
            all_values_array = np.array(all_values)

            # Ensure arrays are the same length
            min_len = min(len(param_values), len(all_values_array))
            param_values = param_values[:min_len]
            all_values_array = all_values_array[:min_len]

            # Calculate correlation
            correlation = np.corrcoef(param_values, all_values_array)[0, 1]
            parameter_impacts[param_name] = correlation

        if not parameter_impacts:
            raise ValueError("No parameter impacts could be calculated")

        # Create tornado plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(parameter_impacts))

        # Sort by absolute impact
        sorted_impacts = sorted(
            parameter_impacts.items(), key=lambda x: abs(x[1]), reverse=True
        )
        names, impacts = zip(*sorted_impacts)

        # Create horizontal bar plot
        plt.barh(y_pos, impacts)
        plt.yticks(y_pos, [self.parameter_distributions[name].name for name in names])
        plt.xlabel(f"Correlation with {metric}")
        plt.title("Parameter Sensitivity Analysis")
        plt.grid(True)
        sns.despine()

        return plt.gcf()

    def visualize_results(self, **kwargs):
        """Create visualizations of Monte Carlo simulation results."""
        self.plot_distribution("total_nominal_cost")
        self.plot_parameter_sensitivity("total_nominal_cost")
        plt.show()
