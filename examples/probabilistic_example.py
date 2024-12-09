import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.analysis.probabilistic import ProbabilisticAnalysis


def run_example():
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
        "discount_rate": {
            "distribution": "uniform",
            "loc": 0.03,  # minimum
            "scale": 0.04,  # range (max - min)
        },
    }

    analysis = ProbabilisticAnalysis(
        fixed_parameters=fixed_params,
        time_horizons=[10, 20, 30],
        course_costs=[12_500, 20_000],
        parameter_distributions=parameter_distributions,
    )

    analysis.run_analysis()
    results_summary = analysis.get_results_summary()

    print("\nProbabilistic Analysis Results:")
    print("\nParameter Sensitivities (Elasticities):")
    print(results_summary[["Parameter", "Time Horizon", "Course Cost", "Elasticity"]])

    print("\nParameter Impact Ranges:")
    print(results_summary[["Parameter", "Time Horizon", "Course Cost", "Impact Range"]])

    analysis.visualize_results()
    plt.show()


if __name__ == "__main__":
    run_example()
