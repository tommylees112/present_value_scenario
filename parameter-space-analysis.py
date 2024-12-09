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

    def __init__(self, fixed_parameters: Dict[str, float], time_horizons: List[int]):
        """
        Initialize the parameter space analysis

        Parameters:
        -----------
        fixed_parameters: Dictionary of parameters to hold constant
        time_horizons: List of time periods (in years) to analyze
        """
        self.fixed_parameters = fixed_parameters
        self.time_horizons = time_horizons

        # Store results
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

    def calculate_nominal_costs(self) -> Dict[int, np.ndarray]:
        """
        Calculate nominal costs for all parameter combinations and time horizons
        """
        for years in self.time_horizons:
            # Initialize results array
            costs = np.zeros_like(self.current_grid)

            # Calculate costs for each parameter combination
            # Pre-calculate all parameter combinations
            param_combinations = list(
                itertools.product(
                    range(len(self.new_values)), range(len(self.current_values))
                )
            )

            # Initialize costs array
            costs = np.zeros_like(self.current_grid)

            # Calculate costs with progress bar
            for i, j in tqdm(
                param_combinations, desc=f"Calculating {years} year horizon"
            ):
                # Set parameters for this calculation
                params = {
                    **self.fixed_parameters,
                    "current_starting_salary": self.current_grid[i, j],
                    "new_career_starting_salary": self.new_grid[i, j],
                    "analysis_years": years,
                }

                # Calculate results for this combination
                results, summary = calculate_career_paths(**params)
                costs[i, j] = summary["Total Nominal Cost"]
            # for i in range(len(self.new_values)):
            #     for j in range(len(self.current_values)):
            #         # Set parameters for this calculation
            #         params = {
            #             **self.fixed_parameters,
            #             "current_starting_salary": self.current_grid[i, j],
            #             "new_career_starting_salary": self.new_grid[i, j],
            #             "analysis_years": years,
            #         }

            #         # Calculate results for this combination
            #         results, summary = calculate_career_paths(**params)
            #         costs[i, j] = summary["Total Nominal Cost"]

            self.results[years] = costs

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

        for years, costs in self.results.items():
            points = self.calculate_breakeven(costs)
            for point in points:
                point["Time Horizon"] = years
                break_even_points.append(point)

        return pd.DataFrame(break_even_points)

    def plot_parameter_space(self):
        """
        Create visualizations of the parameter space analysis in 2D
        """
        n_horizons = len(self.time_horizons)
        fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 5))
        if n_horizons == 1:
            axes = [axes]

        # Create a plot for each time horizon
        for ax, (years, costs) in zip(axes, self.results.items()):
            # Create heatmap with diverging colormap centered at 0
            im = ax.pcolormesh(
                self.current_grid,
                self.new_grid,
                costs,
                cmap="RdYlGn_r",
                norm=TwoSlopeNorm(vcenter=0),
                shading="auto",
            )

            # Calculate average break-even ratio for this time horizon
            break_even_points = self.calculate_breakeven(costs)
            avg_ratio = np.mean(
                [point["Break-even Ratio"] for point in break_even_points]
            )

            # Add break-even line with ratio in label
            contour = ax.contour(
                self.current_grid,
                self.new_grid,
                costs,
                levels=[0],
                colors="black",
                linestyles="dashed",
                linewidths=2,
            )
            contour.collections[0].set_label(
                f"Break-even line (avg ratio: {avg_ratio:.2f}x)"
            )

            # Format axis labels with thousands formatter
            ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

            # Customize plot
            ax.set_title(f"{years} Year Horizon")
            ax.set_xlabel("Current Starting Salary (£)")
            ax.set_ylabel("New Career Starting Salary (£)")

            # Add colorbar with thousands formatter
            cbar = plt.colorbar(im, ax=ax, label="Nominal Cost (£)")
            cbar.ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

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

            # Add legend for break-even line
            ax.legend(loc="upper right")

        plt.tight_layout()
        return fig


# Example usage
def run_parameter_space_analysis(n_points: int = 40):
    # Define fixed parameters
    fixed_params = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 32_500,
        "part_time_earnings": 0,
        "current_growth_rate": 0.05,
        "new_career_growth_rate": 0.05,
        "discount_rate": 0.05,
        "inflation_rate": 0.02,
    }

    # Create analysis object
    analysis = CareerParameterSpaceAnalysis(
        fixed_parameters=fixed_params,
        time_horizons=[5, 10, 20, 30],  # Analyze 5, 10, 20, and 30 year horizons
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
    analysis, break_even_df = run_parameter_space_analysis(n_points=20)

    # Print break-even points
    print("\nBreak-even Analysis:")
    print(
        break_even_df.groupby("Time Horizon").agg(
            {"Break-even Ratio": ["mean", "min", "max"]}
        )
    )

    analysis.plot_parameter_space()
    plt.show()
