"""financial analysis model 1"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


# Formatter function to display y-axis in thousands
def thousands_formatter(x, pos):
    return f"£{x*1e-3:,.0f}k"


def calculate_career_paths(
    # Current career path parameters
    current_starting_salary: float,
    current_growth_rate: float,
    # Retraining path parameters
    retrain_start_date: str,
    course_duration_years: float,
    course_annual_cost: float,
    part_time_earnings: float,
    new_career_starting_salary: float,
    new_career_growth_rate: float,
    # Analysis parameters
    analysis_years: int = 30,
    discount_rate: float = 0.05,
    inflation_rate: float = 0.02,
):
    """
    Calculate and compare financial outcomes of immediate career vs retraining path

    Parameters:
    -----------
    current_starting_salary: Annual salary if starting work immediately
    current_growth_rate: Annual salary growth rate in current path
    retrain_start_date: When retraining begins (YYYY-MM-DD)
    course_duration_years: Length of retraining period
    course_annual_cost: Annual cost of retraining (tuition, materials, etc.)
    part_time_earnings: Annual earnings while retraining
    new_career_starting_salary: Starting salary after retraining
    new_career_growth_rate: Annual salary growth rate in new career
    analysis_years: How many years to project forward
    discount_rate: Rate used to calculate present value
    inflation_rate: Annual inflation rate for real terms calculation
    """

    # Create date range for analysis
    start_date = datetime.strptime(retrain_start_date, "%Y-%m-%d")
    dates = [start_date + timedelta(days=365 * i) for i in range(analysis_years)]

    # Initialize results dataframe
    results = pd.DataFrame({"date": dates, "year": range(analysis_years)})

    # Calculate immediate career path earnings
    results["immediate_career_nominal"] = (
        current_starting_salary * (1 + current_growth_rate) ** results["year"]
    )

    # Calculate retraining path earnings
    results["retrain_career_nominal"] = 0.0

    # During retraining period
    retraining_mask = results["year"] < course_duration_years
    results.loc[retraining_mask, "retrain_career_nominal"] = part_time_earnings
    results.loc[retraining_mask, "retraining_costs"] = course_annual_cost

    # After retraining period
    working_years_after_training = results["year"] - course_duration_years
    post_training_mask = results["year"] >= course_duration_years
    results.loc[post_training_mask, "retrain_career_nominal"] = (
        new_career_starting_salary
        * (1 + new_career_growth_rate)
        ** working_years_after_training[post_training_mask]
    )

    # Calculate present values
    results["discount_factor"] = 1 / (1 + discount_rate) ** results["year"]
    results["inflation_factor"] = 1 / (1 + inflation_rate) ** results["year"]

    # Calculate various metrics
    for career in ["immediate_career", "retrain_career"]:
        # Nominal values
        results[f"{career}_nominal_cumulative"] = results[f"{career}_nominal"].cumsum()

        # Real values (inflation adjusted)
        results[f"{career}_real"] = (
            results[f"{career}_nominal"] * results["inflation_factor"]
        )
        results[f"{career}_real_cumulative"] = results[f"{career}_real"].cumsum()

        # Present values
        results[f"{career}_pv"] = (
            results[f"{career}_nominal"] * results["discount_factor"]
        )
        results[f"{career}_pv_cumulative"] = results[f"{career}_pv"].cumsum()

    # Calculate differences between paths
    for metric in ["nominal", "real", "pv"]:
        results[f"earnings_diff_{metric}"] = (
            results[f"retrain_career_{metric}"] - results[f"immediate_career_{metric}"]
        )
        results[f"earnings_diff_{metric}_cumulative"] = (
            results[f"retrain_career_{metric}_cumulative"]
            - results[f"immediate_career_{metric}_cumulative"]
        )

    # Calculate key summary metrics
    summary_metrics = {
        "Total Nominal Cost": (
            results["earnings_diff_nominal_cumulative"].iloc[-1] * -1
        ),
        "Total Real Cost": (results["earnings_diff_real_cumulative"].iloc[-1] * -1),
        "Net Present Cost": (results["earnings_diff_pv_cumulative"].iloc[-1] * -1),
        "Break-even Year": results[results["earnings_diff_nominal_cumulative"] > 0][
            "year"
        ].min(),
        "Years to Break Even": len(
            results[results["earnings_diff_nominal_cumulative"] < 0]
        ),
        "Maximum Annual Earnings Gap": results["earnings_diff_nominal"].min() * -1,
        "Peak Cumulative Cost": results["earnings_diff_nominal_cumulative"].min() * -1,
    }

    return results, summary_metrics


def plot_earnings_comparison(results):
    """Create visualization of earnings paths and differences"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot cumulative earnings
    ax1.plot(
        results["year"],
        results["immediate_career_nominal_cumulative"],
        label=f"Immediate Career Path: {params['current_starting_salary']/1e3} -- {params['current_growth_rate']*100}%",
        linewidth=2,
    )
    ax1.plot(
        results["year"],
        results["retrain_career_nominal_cumulative"],
        label=f"Retraining Path: {params['new_career_starting_salary']/1e3} -- {params['new_career_growth_rate']*100}%",
        linewidth=2,
    )
    ax1.set_title("Cumulative Earnings Over Time")
    ax1.set_xlabel("Years")
    ax1.set_ylabel("Cumulative Earnings")
    ax1.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # Format y-axis
    ax1.legend()
    ax1.grid(True)
    sns.despine(ax=ax1)

    # Plot earnings difference
    ax2.plot(
        results["year"],
        results["earnings_diff_nominal_cumulative"],
        color="red"
        if results["earnings_diff_nominal_cumulative"].iloc[-1] < 0
        else "green",
        linewidth=2,
    )
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.set_title("Cumulative Earnings Difference (Retraining - Immediate)")
    ax2.set_xlabel("Years")
    ax2.set_ylabel("Earnings Difference")
    ax2.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # Format y-axis
    ax2.grid(True)
    sns.despine(ax=ax2)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("MacOSX")  # or try 'MacOSX' backend

    # Parameters for your specific case
    params = {
        "current_starting_salary": 50000,
        "current_growth_rate": 0.05,
        "retrain_start_date": "2025-09-01",
        "course_duration_years": 3,
        "course_annual_cost": 12500 + 20000,  # Combined tuition and living costs
        "part_time_earnings": 20000,
        "new_career_starting_salary": 100000,
        "new_career_growth_rate": 0.05,
        "analysis_years": 30,
        "discount_rate": 0.05,
        "inflation_rate": 0.02,
    }

    # Calculate results
    results, summary = calculate_career_paths(**params)

    plot_earnings_comparison(results)
    plt.show()
