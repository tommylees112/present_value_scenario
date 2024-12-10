import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.analysis.probabilistic import ProbabilisticAnalysis


def initialise_example():
    fixed_params = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "part_time_earnings": 0,
        "current_growth_rate": 0.05,
        "new_career_growth_rate": 0.05,
        "inflation_rate": 0.02,
        "current_starting_salary": 60_000,
        "new_career_starting_salary": 75_000,
    }

    # Define probability distributions for uncertain parameters
    parameter_distributions = {
        "current_growth_rate": {
            "distribution": "norm",
            "loc": 0.05,  # mean
            "scale": 0.01,  # standard deviation
        },
        "new_career_growth_rate": {
            "distribution": "norm",
            "loc": 0.05,
            "scale": 0.015,
        },
        # "discount_rate": {
        #     "distribution": "uniform",
        #     "loc": 0.03,  # minimum
        #     "scale": 0.04,  # range (max - min)
        # },
    }

    analysis = ProbabilisticAnalysis(
        fixed_parameters=fixed_params,
        time_horizons=[10, 20, 30],
        course_costs=[12_500, 20_000],
        parameter_distributions=parameter_distributions,
    )

    return analysis


if __name__ == "__main__":
    analysis = initialise_example()

    analysis.run_analysis()
    results_summary = analysis.get_results_summary()

    print("\nProbabilistic Analysis Results:")

    if not results_summary.empty:
        print("\nParameter Sensitivities (Elasticities):")
        """
        Elasticity: This value measures the sensitivity of the total cost to changes in the parameter. It is calculated as the percentage change in the output (total cost) divided by the percentage change in the input (parameter value).
        A negative elasticity (e.g., -16.961504 for current_growth_rate over 10 years) indicates that an increase in the parameter leads to a decrease in the total cost.
        A positive elasticity (e.g., 0.823912 for new_career_growth_rate over 10 years) indicates that an increase in the parameter leads to an increase in the total cost.
        """
        print(
            results_summary[["parameter", "time_horizon", "course_cost", "elasticity"]]
        )

        print("\nParameter Impact Ranges:")
        """ 
        Impact Range: This value represents the range of total costs observed when varying the parameter within its specified range. It is the difference between the maximum and minimum total costs observed.
        Larger impact ranges indicate that the parameter has a significant effect on the total cost, suggesting high uncertainty or variability in outcomes due to changes in that parameter.
        """
        print(
            results_summary[
                ["parameter", "time_horizon", "course_cost", "impact_range"]
            ]
        )
    else:
        print("No results to display.")

    analysis.visualize_results()
    plt.show()
