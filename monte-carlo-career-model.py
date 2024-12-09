from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

from fam1 import calculate_career_paths


# Formatter function to display y-axis in thousands
def thousands_formatter(x, pos):
    return f"£{x*1e-3:,.0f}k"


@dataclass
class ParameterDistribution:
    """
    Represents a probability distribution for a model parameter

    distribution_type: The statistical distribution to use (e.g., 'normal', 'lognormal', 'uniform')
    params: Dictionary of parameters specific to the distribution type
    """

    name: str
    distribution_type: str
    params: Dict[str, float]

    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution"""
        if self.distribution_type == "normal":
            return stats.norm.rvs(
                loc=self.params["mean"], scale=self.params["std"], size=size
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
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")


class MonteCarloCareerAnalysis:
    """
    Analyzes career paths using Monte Carlo simulation with probabilistic inputs
    """

    def __init__(
        self,
        parameter_distributions: Dict[str, ParameterDistribution],
        fixed_parameters: Dict[str, float],
        n_simulations: int = 1000,
        analysis_years: int = 30,
    ):
        """
        Initialize the Monte Carlo analysis

        parameter_distributions: Dictionary mapping parameter names to their distributions
        fixed_parameters: Dictionary of parameters that don't vary
        n_simulations: Number of Monte Carlo simulations to run. Defaults to 1000
        analysis_years: Number of years to project. Defaults to 30
        """
        self.parameter_distributions = parameter_distributions
        self.fixed_parameters = fixed_parameters
        self.n_simulations = n_simulations
        self.analysis_years = analysis_years

        # Storage for simulation results
        self.simulation_results = []
        self.summary_statistics = None

    def run_simulation(self):
        """Execute the Monte Carlo simulation"""
        # Generate parameter samples for each simulation
        parameter_samples = {
            name: dist.sample(self.n_simulations)
            for name, dist in self.parameter_distributions.items()
        }

        # Run simulations
        for i in range(self.n_simulations):
            # Combine sampled and fixed parameters for this simulation
            params = {
                **self.fixed_parameters,
                **{name: parameter_samples[name][i] for name in parameter_samples},
            }

            # Calculate results for this simulation
            results, summary = calculate_career_paths(**params)

            # Store results
            self.simulation_results.append(
                {
                    "simulation_id": i,
                    "parameters": params,
                    "results": results,
                    "summary": summary,
                }
            )

        self._calculate_summary_statistics()

    def _calculate_summary_statistics(self):
        """Calculate summary statistics across all simulations"""
        # Extract key metrics from all simulations
        metrics = {}
        for metric in self.simulation_results[0]["summary"].keys():
            values = [sim["summary"][metric] for sim in self.simulation_results]
            metrics[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "percentile_5": np.percentile(values, 5),
                "percentile_95": np.percentile(values, 95),
            }

        self.summary_statistics = metrics

    def plot_distribution(self, metric: str):
        """Plot the distribution of outcomes for a specific metric"""
        values = [sim["summary"][metric] for sim in self.simulation_results]

        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        if metric == "Total Nominal Cost":
            plt.title(
                f"Distribution of {metric}\n nominal_cost = (immediate_career_earnings - retraining_path_earnings)\n+ve=LOSING money -ve=MAKING money"
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
            percentile_value = np.percentile(values, p)
            plt.axvline(
                percentile_value,
                color=c,
                linestyle=s,
                label=f"{p}th percentile: {percentile_value:,.0f}",
            )

        plt.legend()
        plt.grid(True)
        sns.despine(ax=plt.gca())
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        return plt.gcf()

    def plot_parameter_sensitivity(self, metric: str):
        """Create a tornado plot showing parameter sensitivity"""
        parameter_impacts = {}

        # Calculate impact of each parameter
        for param_name in self.parameter_distributions.keys():
            values = [sim["summary"][metric] for sim in self.simulation_results]
            parameters = [
                sim["parameters"][param_name] for sim in self.simulation_results
            ]

            # Calculate correlation
            correlation = np.corrcoef(parameters, values)[0, 1]
            parameter_impacts[param_name] = correlation

        # Create tornado plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(parameter_impacts))

        # Sort by absolute impact
        sorted_impacts = sorted(
            parameter_impacts.items(), key=lambda x: abs(x[1]), reverse=True
        )

        names, impacts = zip(*sorted_impacts)

        plt.barh(y_pos, impacts)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        plt.yticks(y_pos, names)
        plt.xlabel("Correlation with " + metric)
        plt.title("Parameter Sensitivity Analysis")
        plt.grid(True)
        sns.despine(plt.gcf())

        return plt.gcf()


# Example usage
def create_example_analysis():
    """Create example analysis with probability distributions"""
    # Define parameter distributions
    parameter_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Starting Salary",
            distribution_type="normal",
            params={"mean": 100000, "std": 5000},
        ),
        "current_growth_rate": ParameterDistribution(
            name="Current Growth Rate",
            distribution_type="normal",
            params={"mean": 0.05, "std": 0.01},
        ),
        "new_career_starting_salary": ParameterDistribution(
            name="New Career Starting Salary",
            distribution_type="lognormal",
            params={"mu": np.log(100000), "sigma": 0.2},
        ),
        "new_career_growth_rate": ParameterDistribution(
            name="New Career Growth Rate",
            distribution_type="normal",
            params={"mean": 0.05, "std": 0.015},
        ),
        "part_time_earnings": ParameterDistribution(
            name="Part Time Earnings",
            distribution_type="uniform",
            params={"min": 0, "max": 20000},
        ),
    }

    # Define fixed parameters
    fixed_parameters = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 32500,
        "analysis_years": 30,
        "discount_rate": 0.05,
        "inflation_rate": 0.02,
    }

    return MonteCarloCareerAnalysis(
        parameter_distributions=parameter_distributions,
        fixed_parameters=fixed_parameters,
        n_simulations=1000,
    )


def plot_parameter_distributions(parameter_distributions):
    """
    Create a side-by-side visualization of parameter distributions with annotations
    showing key statistics and properties of each distribution.
    """
    n_params = len(parameter_distributions)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))
    fig.suptitle("Probability Distributions of Model Parameters", fontsize=12, y=1.05)

    # Generate x-ranges for each distribution
    for ax, (param_name, dist) in zip(axes, parameter_distributions.items()):
        if dist.distribution_type == "normal":
            # Generate points for normal distribution
            x = np.linspace(
                dist.params["mean"] - 4 * dist.params["std"],
                dist.params["mean"] + 4 * dist.params["std"],
                1000,
            )
            y = stats.norm.pdf(x, dist.params["mean"], dist.params["std"])

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Add vertical lines for mean and standard deviations
            ax.axvline(dist.params["mean"], color="r", linestyle="--", label="Mean")
            ax.axvline(
                dist.params["mean"] + dist.params["std"],
                color="g",
                linestyle=":",
                label="±1 SD",
            )
            ax.axvline(
                dist.params["mean"] - dist.params["std"], color="g", linestyle=":"
            )

            # Add annotations
            ax.text(
                0.05,
                0.95,
                f"Mean: {dist.params['mean']:,.0f}\nSD: {dist.params['std']:,.0f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        elif dist.distribution_type == "lognormal":
            # Generate points for lognormal distribution
            x = np.linspace(
                0, np.exp(dist.params["mu"] + 4 * dist.params["sigma"]), 1000
            )
            y = stats.lognorm.pdf(
                x, s=dist.params["sigma"], scale=np.exp(dist.params["mu"])
            )

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Calculate and plot key statistics
            median = np.exp(dist.params["mu"])
            mode = np.exp(dist.params["mu"] - dist.params["sigma"] ** 2)
            mean = np.exp(dist.params["mu"] + dist.params["sigma"] ** 2 / 2)

            ax.axvline(median, color="r", linestyle="--", label="Median")
            ax.axvline(mode, color="g", linestyle=":", label="Mode")

            # Add annotations
            ax.text(
                0.05,
                0.95,
                f"Median: {median:,.0f}\nMode: {mode:,.0f}\nMean: {mean:,.0f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        elif dist.distribution_type == "uniform":
            # Generate points for uniform distribution
            x = np.linspace(
                dist.params["min"] - 0.1 * (dist.params["max"] - dist.params["min"]),
                dist.params["max"] + 0.1 * (dist.params["max"] - dist.params["min"]),
                1000,
            )
            y = stats.uniform.pdf(
                x, dist.params["min"], dist.params["max"] - dist.params["min"]
            )

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Add vertical lines for bounds
            ax.axvline(dist.params["min"], color="r", linestyle="--", label="Bounds")
            ax.axvline(dist.params["max"], color="r", linestyle="--")

            # Add annotations
            mean = (dist.params["min"] + dist.params["max"]) / 2
            ax.text(
                0.05,
                0.95,
                f"Min: {dist.params['min']:,.0f}\nMax: {dist.params['max']:,.0f}\nMean: {mean:,.0f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Customize each subplot
        ax.set_title(param_name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("MacOSX")

    analysis = create_example_analysis()
    analysis.run_simulation()

    # Get summary statistics
    print(analysis.summary_statistics["Total Nominal Cost"])
    # Shows mean, median, standard deviation, and percentiles

    plot_parameter_distributions(analysis.parameter_distributions)

    # Visualize the distribution of outcomes
    analysis.plot_distribution("Total Nominal Cost")
    plt.show()

    # Understand which parameters matter most
    analysis.plot_parameter_sensitivity("Total Nominal Cost")
    plt.show()
