"""
Parameter Space Exploration:

- Creates a grid of combinations between current and new career starting salaries
- Calculates the nominal cost for each combination
- Does this analysis for different time horizons (e.g., 10, 20, and 30 years)

Visualization:
- Creates a heatmap for each time horizon where:
    - The x-axis shows current starting salary
    - The y-axis shows new career starting salary
    - Colors indicate profitability (green for profitable, red for unprofitable)
    - A dashed black line shows the "break-even" boundary

Break-even Analysis:

- Finds the minimum new career salary needed to break even for each current salary
- Calculates the ratio of new salary to current salary needed for profitability
- Shows how these requirements change with different time horizons

"""

import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from fam1 import calculate_career_paths, thousands_formatter


class CareerParameterSpaceAnalysis:
    """
    Analyzes how varying combinations of parameters affect career change outcomes
    """

    def __init__(
        self,
        fixed_parameters: Dict[str, float],
        time_horizons: List[int],
        course_costs: List[float],
    ):
        """
        Initialize the parameter space analysis

        Parameters:
        -----------
        fixed_parameters: Dictionary of parameters to hold constant
        time_horizons: List of time periods (in years) to analyze
        course_costs: List of annual course costs to analyze
        """
        self.fixed_parameters = fixed_parameters
        self.time_horizons = time_horizons
        self.course_costs = course_costs
        self.results = {}

    def create_parameter_grid(
        self,
        current_salary_range: Tuple[float, float, int],
        new_salary_range: Tuple[float, float, int],
    ) -> np.ndarray:
        """
        Create a grid of parameter combinations to evaluate

        Parameters:
        -----------
        current_salary_range: Tuple of (min, max, num_points) for current salary
        new_salary_range: Tuple of (min, max, num_points) for new career salary

        Returns:
        --------
        Tuple of parameter grids and their values
        """
        # Create parameter ranges
        current_salaries = np.linspace(*current_salary_range)
        new_salaries = np.linspace(*new_salary_range)

        # Create 2D grids
        self.current_grid, self.new_grid = np.meshgrid(current_salaries, new_salaries)

        # Store values for plotting
        self.current_values = current_salaries
        self.new_values = new_salaries

        return self.current_grid, self.new_grid

    def calculate_nominal_costs(self) -> Dict[Tuple[int, float], np.ndarray]:
        """
        Calculate nominal costs for all parameter combinations, time horizons, and course costs
        """
        for years in self.time_horizons:
            for cost in self.course_costs:
                # Create unique key for this combination
                key = (years, cost)

                # Initialize results array
                costs = np.zeros_like(self.current_grid)

                # Calculate costs for each parameter combination
                param_combinations = list(
                    itertools.product(
                        range(len(self.new_values)), range(len(self.current_values))
                    )
                )

                # Calculate costs with progress bar
                for i, j in tqdm(
                    param_combinations,
                    desc=f"Calculating {years} year horizon, £{cost:,} annual cost",
                ):
                    params = {
                        **self.fixed_parameters,
                        "current_starting_salary": self.current_grid[i, j],
                        "new_career_starting_salary": self.new_grid[i, j],
                        "analysis_years": years,
                        "course_annual_cost": cost,
                    }

                    results, summary = calculate_career_paths(**params)
                    costs[i, j] = summary["Total Nominal Cost"]

                self.results[key] = costs

        return self.results

    def calculate_breakeven(self, costs: np.ndarray) -> List[Dict[str, float]]:
        """
        Calculate break-even points for a given cost matrix.

        Parameters:
        -----------
        costs: 2D numpy array of costs for each parameter combination

        Returns:
        --------
        List of dictionaries containing break-even information
        """
        break_even_points = []

        for j, current_salary in enumerate(self.current_values):
            cost_column = costs[:, j]
            if np.any(cost_column < 0):
                break_even_idx = np.where(cost_column < 0)[0][0]
                break_even_salary = self.new_values[break_even_idx]
                break_even_points.append(
                    {
                        "Current Salary": current_salary,
                        "Break-even New Salary": break_even_salary,
                        "Break-even Ratio": break_even_salary / current_salary,
                    }
                )

        return break_even_points

    def find_break_even_points(self) -> pd.DataFrame:
        """
        Find the new career salary needed to break even for each current salary
        and time horizon
        """
        break_even_points = []

        for params, costs in self.results.items():
            points = self.calculate_breakeven(costs)
            for point in points:
                point["Time Horizon"] = params[0]
                point["Course Cost (£)"] = params[1]
                break_even_points.append(point)

        return pd.DataFrame(break_even_points)

    def plot_parameter_space(self, ind_plot_scale: float = 1):
        """
        Create visualizations of the parameter space analysis in a grid
        """
        n_horizons = len(self.time_horizons)
        n_costs = len(self.course_costs)

        fig, axes = plt.subplots(
            n_costs,
            n_horizons,
            figsize=(6 * n_horizons * ind_plot_scale, 5 * n_costs * ind_plot_scale),
        )

        # Ensure axes is 2D even with single row/column
        if n_horizons == 1 and n_costs == 1:
            axes = np.array([[axes]])
        elif n_horizons == 1:
            axes = axes.reshape(-1, 1)
        elif n_costs == 1:
            axes = axes.reshape(1, -1)

        # Create a plot for each combination
        for i, cost in enumerate(self.course_costs):
            for j, years in enumerate(self.time_horizons):
                ax = axes[i, j]
                key = (years, cost)
                costs = self.results[key]

                # Create heatmap
                im = ax.pcolormesh(
                    self.current_grid,
                    self.new_grid,
                    costs,
                    cmap="RdYlGn_r",
                    norm=TwoSlopeNorm(vcenter=0),
                    shading="auto",
                )

                # Calculate and plot break-even line
                break_even_points = self.calculate_breakeven(costs)
                avg_ratio = np.mean(
                    [point["Break-even Ratio"] for point in break_even_points]
                )

                contour = ax.contour(
                    self.current_grid,
                    self.new_grid,
                    costs,
                    levels=[0],
                    colors="black",
                    linestyles="dashed",
                    linewidths=2,
                )

                # Add labels and formatting
                ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
                ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

                if i == n_costs - 1:  # Bottom row
                    ax.set_xlabel("Current Starting Salary (£)")
                if j == 0:  # Leftmost column
                    ax.set_ylabel("New Career Starting Salary (£)")

                ax.set_title(f"{years} Year Horizon\nAnnual Cost: £{cost:,}")

                # Add annotations for regions
                ax.text(
                    0.05,
                    0.95,
                    "Profitable\nCareer Change",
                    transform=ax.transAxes,
                    color="green",
                    fontweight="bold",
                    verticalalignment="top",
                )
                ax.text(
                    0.05,
                    0.05,
                    "Unprofitable\nCareer Change",
                    transform=ax.transAxes,
                    color="red",
                    fontweight="bold",
                    verticalalignment="bottom",
                )

                # Add break-even line label
                if break_even_points:
                    contour.collections[0].set_label(
                        f"Break-even line (avg ratio: {avg_ratio:.2f}x)"
                    )
                    ax.legend(loc="upper right")

                # Add colorbar (only for rightmost column)
                if j == n_horizons - 1:
                    cbar = plt.colorbar(im, ax=ax, label="Nominal Cost (£)")
                    cbar.ax.yaxis.set_major_formatter(
                        FuncFormatter(thousands_formatter)
                    )

        plt.tight_layout()
        return fig


# Example usage
def run_parameter_space_analysis(n_points: int = 40):
    fixed_params = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "part_time_earnings": 0,
        "current_growth_rate": 0.05,
        "new_career_growth_rate": 0.05,
        "discount_rate": 0.05,
        "inflation_rate": 0.02,
    }

    course_costs = [0, 12_500, 20_000, 32_500]

    analysis = CareerParameterSpaceAnalysis(
        fixed_parameters=fixed_params,
        time_horizons=[5, 10, 30],
        course_costs=course_costs,
    )

    # Create parameter grid
    analysis.create_parameter_grid(
        current_salary_range=(
            40_000,
            120_000,
            n_points,
        ),  # Current salary from 40k to 120k
        new_salary_range=(
            40_000,
            120_000,
            n_points,
        ),  # New career salary from 40k to 120k
    )

    # Calculate results
    analysis.calculate_nominal_costs()

    # Create visualizations
    # analysis.plot_parameter_space()

    # Find break-even points
    break_even_df = analysis.find_break_even_points()

    return analysis, break_even_df


if __name__ == "__main__":
    # Run the analysis
    analysis, break_even_df = run_parameter_space_analysis(n_points=10)

    # Print break-even points
    print("\nBreak-even Analysis:")
    print(
        break_even_df.groupby("Time Horizon").agg(
            {"Break-even Ratio": ["mean", "min", "max"]}
        )
    )

    analysis.plot_parameter_space()
    plt.show()
