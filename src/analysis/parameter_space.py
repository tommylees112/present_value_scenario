"""Parameter space analysis implementation."""

import itertools
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from ..core.career_model import calculate_career_paths
from .base import CareerAnalysis


class ParameterSpaceAnalysis(CareerAnalysis):
    """Analyzes how varying combinations of parameters affect career change outcomes."""

    def __init__(
        self,
        fixed_parameters: Dict[str, Any],
        time_horizons: List[int],
        course_costs: List[float],
    ):
        super().__init__(fixed_parameters, time_horizons, course_costs)
        self.current_grid = None
        self.new_grid = None
        self.current_values = None
        self.new_values = None

    def create_parameter_grid(
        self,
        current_salary_range: Tuple[float, float, int],
        new_salary_range: Tuple[float, float, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create a grid of parameter combinations to evaluate."""
        # Create parameter ranges
        self.current_salaries = np.linspace(*current_salary_range)
        self.new_salaries = np.linspace(*new_salary_range)

        # Create 2D grids
        self.current_grid, self.new_grid = np.meshgrid(
            self.current_salaries, self.new_salaries
        )

        return self.current_grid, self.new_grid

    def run_analysis(self):
        """Run the analysis for all parameter combinations."""
        if self.current_grid is None or self.new_grid is None:
            raise ValueError("Must call create_parameter_grid before running analysis")

        # Store results for each time horizon and course cost combination
        all_results = {}

        for years in self.time_horizons:
            for cost in self.course_costs:
                n_current = len(self.current_salaries)
                n_new = len(self.new_salaries)

                # Initialize arrays to store results
                costs = np.zeros((n_current, n_new))
                break_even = np.zeros((n_current, n_new), dtype=bool)
                years_to_break_even = np.full((n_current, n_new), np.nan)

                # Create all parameter combinations
                param_combinations = list(
                    itertools.product(range(n_current), range(n_new))
                )

                # Run analysis for each parameter combination with progress bar
                for i, j in tqdm(
                    param_combinations,
                    desc=f"Calculating {years} year horizon, £{cost:,} annual cost",
                    leave=True,
                ):
                    params = {
                        **self.fixed_parameters,
                        "current_starting_salary": self.current_salaries[i],
                        "new_career_starting_salary": self.new_salaries[j],
                        "analysis_years": years,
                        "course_annual_cost": cost,
                    }

                    _, summary = calculate_career_paths(**params)

                    # Store results
                    costs[i, j] = summary["total_nominal_cost"]
                    break_even[i, j] = summary["break_even"]
                    years_to_break_even[i, j] = (
                        summary["years_to_break_even"]
                        if summary["years_to_break_even"] is not None
                        else np.nan
                    )

                all_results[(years, cost)] = {
                    "costs": costs,
                    "break_even": break_even,
                    "years_to_break_even": years_to_break_even,
                }

        self.results = all_results

    def calculate_breakeven(self, costs: np.ndarray) -> List[Dict[str, float]]:
        """Calculate break-even points for a given cost matrix."""
        break_even_points = []

        for j, current_salary in enumerate(self.current_values):
            cost_column = costs[:, j]
            if np.any(cost_column < 0):
                break_even_idx = np.where(cost_column < 0)[0][0]
                break_even_salary = self.new_values[break_even_idx]
                break_even_points.append(
                    {
                        "current_salary": current_salary,
                        "break_even_new_salary": break_even_salary,
                        "break_even_ratio": break_even_salary / current_salary,
                    }
                )

        return break_even_points

    def get_results_summary(self) -> pd.DataFrame:
        """Find break-even points across all scenarios."""
        if self.results is None:
            raise ValueError("Must run analysis before getting summary")

        summary_data = []
        for (years, cost), result_dict in self.results.items():
            for i, current_salary in enumerate(self.current_salaries):
                for j, new_salary in enumerate(self.new_salaries):
                    summary_data.append(
                        {
                            "time_horizon": years,
                            "course_cost": cost,
                            "current_salary": current_salary,
                            "new_salary": new_salary,
                            "total_cost": result_dict["costs"][i, j],
                            "break_even": result_dict["break_even"][i, j],
                            "years_to_break_even": result_dict["years_to_break_even"][
                                i, j
                            ],
                        }
                    )

        return pd.DataFrame(summary_data)

    def calculate_breakeven_ratio(self, costs: np.ndarray) -> float:
        """Calculate average break-even ratio for a given cost matrix."""
        break_even_points = []
        for i in range(len(self.current_salaries)):
            cost_column = costs[:, i]
            if np.any(cost_column < 0):
                break_even_idx = np.where(cost_column < 0)[0][0]
                break_even_ratio = (
                    self.new_salaries[break_even_idx] / self.current_salaries[i]
                )
                break_even_points.append(break_even_ratio)

        return np.mean(break_even_points) if break_even_points else np.nan

    def visualize_results(self, ind_plot_scale: float = 1):
        """Create visualizations of the parameter space analysis in a grid."""
        if self.results is None:
            raise ValueError("Must run analysis before visualizing")

        n_costs = len(self.course_costs)
        n_horizons = len(self.time_horizons)

        # Create figure with subplots grid
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

        # Create meshgrid for plotting
        X, Y = np.meshgrid(self.current_salaries, self.new_salaries)

        # Create a plot for each combination
        for i, cost in enumerate(self.course_costs):
            for j, years in enumerate(self.time_horizons):
                ax = axes[i, j]
                result_dict = self.results[(years, cost)]
                costs = result_dict["costs"]

                # Create pcolormesh plot
                im = ax.pcolormesh(
                    X,
                    Y,
                    costs,
                    cmap="RdYlBu_r",
                    norm=TwoSlopeNorm(vcenter=0),
                    shading="auto",
                )

                # Format tick labels
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"£{x:,.0f}"))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"£{x:,.0f}"))

                # Add labels only for bottom row and leftmost column
                if i == n_costs - 1:  # Bottom row
                    ax.set_xlabel("Current Starting Salary")
                if j == 0:  # Leftmost column
                    ax.set_ylabel("New Career Starting Salary")

                ax.set_title(f"{years} Year Horizon\nAnnual Cost: £{cost:,}")

                # Add annotations for regions
                if np.any(costs < 0):
                    ax.text(
                        0.75,
                        0.05,
                        "Profitable\nCareer Change",
                        transform=ax.transAxes,
                        color="green",
                        fontweight="bold",
                        verticalalignment="bottom",
                    )
                if np.any(costs > 0):
                    ax.text(
                        0.05,
                        0.95,
                        "Unprofitable\nCareer Change",
                        transform=ax.transAxes,
                        color="red",
                        fontweight="bold",
                        verticalalignment="top",
                    )

                # Calculate and add break-even ratio
                avg_ratio = self.calculate_breakeven_ratio(costs)

                # Add contour for break-even line with label
                contour = ax.contour(
                    X,
                    Y,
                    costs,
                    levels=[0],
                    colors="black",
                    linestyles="dashed",
                    linewidths=2,
                )

                if not np.isnan(avg_ratio):
                    # Format the label with the ratio
                    label = f"Break-even\n({avg_ratio:.2f}x)"
                    # Add manual legend entry
                    ax.plot(
                        [],
                        [],
                        color="black",
                        linestyle="dashed",
                        linewidth=2,
                        label=label,
                    )
                    ax.legend(loc="upper right")

                # Add colorbar only for rightmost column
                if j == n_horizons - 1:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("Nominal Cost (£)")
                    cbar.ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda x, p: f"£{x:,.0f}")
                    )

        plt.tight_layout()
        return fig
