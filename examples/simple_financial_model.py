import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt

from src.analysis.plots import plot_earnings_comparison
from src.analysis.risk import RiskAnalyzer
from src.core.career_model import CareerModel
from src.core.career_model_params import CareerModelParams

if __name__ == "__main__":
    params_dict = {
        "current_starting_salary": 75_000,
        "current_growth_rate": 0.05,
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 25_000,
        "part_time_earnings": 10_000,
        "new_career_starting_salary": 150_000,
        "new_career_growth_rate": 0.05,
        "analysis_years": 30,
        "inflation_rate": 0.02,
    }
    params = CareerModelParams.from_dict(params_dict)

    # After running career model
    career_model_result = CareerModel(params).calculate()
    df, summary = career_model_result

    plot_earnings_comparison(df, params_dict)

    risk_analyzer = RiskAnalyzer(career_model_result)

    # Get risk metrics
    risk_metrics = risk_analyzer.calculate_risk_metrics()

    # Get human-readable summary
    risk_summary = risk_analyzer.get_risk_summary()

    print(risk_metrics)
    print(risk_summary)
    plt.show()
