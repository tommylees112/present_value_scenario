import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.parameter_space import ParameterSpaceAnalysis


def run_example():
    fixed_params = {
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "part_time_earnings": 0,
        "current_growth_rate": 0.05,
        "new_career_growth_rate": 0.05,
        "inflation_rate": 0.02,
    }

    analysis = ParameterSpaceAnalysis(
        fixed_parameters=fixed_params,
        time_horizons=[5, 10, 30],
        course_costs=[0, 12_500, 20_000, 32_500],
    )

    N_SIMS = 10
    analysis.create_parameter_grid(
        current_salary_range=(40_000, 120_000, N_SIMS),
        new_salary_range=(40_000, 120_000, N_SIMS),
    )

    analysis.run_analysis()
    results_summary = analysis.get_results_summary()
    print("\nResults Summary:")
    print(results_summary)

    analysis.visualize_results(ind_plot_scale=0.6)
    plt.show()


if __name__ == "__main__":
    run_example()
