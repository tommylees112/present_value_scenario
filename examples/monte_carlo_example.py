import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.monte_carlo import MonteCarloAnalysis, ParameterDistribution


def get_parameters():
    # Define parameter distributions
    parameter_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Starting Salary",
            distribution_type="truncated_normal",
            params={
                "mean": 100000,
                "std": 5000,
                "min": 0,  # Salary can't be negative
                "max": float("inf"),  # No upper limit
            },
        ),
        "current_growth_rate": ParameterDistribution(
            name="Current Growth Rate",
            distribution_type="truncated_normal",
            params={
                "mean": 0.05,
                "std": 0.01,
                "min": 0,  # Growth rate can't be negative
                "max": 1,  # Growth rate can't exceed 100%
            },
        ),
        "new_career_starting_salary": ParameterDistribution(
            name="New Career Starting Salary",
            distribution_type="truncated_normal",
            params={
                "mean": 100000,
                "std": 20000,
                "min": 0,  # Salary can't be negative
                "max": float("inf"),  # No upper limit
            },
        ),
        "new_career_growth_rate": ParameterDistribution(
            name="New Career Growth Rate",
            distribution_type="truncated_normal",
            params={
                "mean": 0.05,
                "std": 0.015,
                "min": 0,  # Growth rate can't be negative
                "max": 1,  # Growth rate can't exceed 100%
            },
        ),
        "part_time_earnings": ParameterDistribution(
            name="Part Time Earnings",
            distribution_type="truncated_normal",
            params={
                "mean": 10000,
                "std": 5000,
                "min": 0,  # Earnings can't be negative
                "max": float("inf"),  # No upper limit
            },
        ),
    }

    # Define fixed parameters
    fixed_parameters = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "inflation_rate": 0.02,
    }

    # Create and run analysis
    analysis = MonteCarloAnalysis(
        parameter_distributions=parameter_distributions,
        fixed_parameters=fixed_parameters,
        time_horizons=[10, 20, 30],
        course_costs=[12_500, 20_000],
        n_simulations=1000,
        random_seed=42,
    )

    return analysis


if __name__ == "__main__":
    analysis = get_parameters()
    analysis.run_analysis()

    # Print summary statistics
    results_summary = analysis.get_results_summary()
    print("\nMonte Carlo Analysis Results:")
    print(results_summary)

    # Create visualizations
    analysis.visualize_results()
    plt.show()
