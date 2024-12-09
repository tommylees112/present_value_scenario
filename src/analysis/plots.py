"""Visualization module for career change analysis."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..utils.plot_utils import thousands_formatter


def plot_earnings_comparison(
    results: pd.DataFrame,
    params: Dict,
    title: Optional[str] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Create visualization of earnings paths and differences.

    Parameters:
    -----------
    results: DataFrame containing career paths and earnings
    params: Dictionary of model parameters
    title: Optional custom title for the plot
    figsize: Figure size tuple (width, height)

    Returns:
    --------
    matplotlib.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot cumulative earnings
    _plot_cumulative_earnings(ax1, results, params)

    # Plot earnings difference
    _plot_earnings_difference(ax2, results)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


def _plot_cumulative_earnings(ax, results: pd.DataFrame, params: Dict):
    """Plot cumulative earnings for both career paths."""
    # Calculate cumulative earnings
    current_cum = results["current_career"].cumsum()
    new_cum = (results["new_career"] - results["course_costs"]).cumsum()

    # Create labels with formatted salary and growth rate
    current_label = (
        f"Current Career: £{params['current_starting_salary']:,.0f} "
        f"(+{params['current_growth_rate']*100:.1f}%/yr)"
    )
    new_label = (
        f"New Career: £{params['new_career_starting_salary']:,.0f} "
        f"(+{params['new_career_growth_rate']*100:.1f}%/yr)"
    )

    # Plot lines
    ax.plot(results["year"], current_cum, label=current_label, linewidth=2)
    ax.plot(results["year"], new_cum, label=new_label, linewidth=2)

    # Add study period shading
    study_years = int(params["course_duration_years"])
    ax.axvspan(
        0,
        study_years,
        color="gray",
        alpha=0.1,
        label=f"Study Period ({study_years} years, £{(params['course_annual_cost']*study_years) - (params['part_time_earnings']*study_years):,.0f} total cost)",
    )

    # Formatting
    # Add salary ratio to title
    salary_ratio = (
        params["new_career_starting_salary"] / params["current_starting_salary"]
    )
    ax.set_title(
        f"Cumulative Earnings Over Time \n(New/Current Salary Ratio: {salary_ratio:.1f}x)"
    )
    ax.set_xlabel("Years")
    ax.set_ylabel("Cumulative Earnings (£)")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def _plot_earnings_difference(
    ax, results: pd.DataFrame, display_years: list[int] = [3, 5, 10, 20, 29]
):
    """Plot cumulative earnings difference between career paths."""
    # Calculate cumulative difference
    diff = (
        results["new_career"] - results["course_costs"] - results["current_career"]
    ).cumsum()

    output = results[["date", "year"]].join(pd.Series(diff, name="diff"))

    # Determine color based on final outcome
    color = "green" if diff.iloc[-1] > 0 else "red"

    # Plot difference
    ax.plot(results["year"], diff, color=color, linewidth=2)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    # Add annotations
    for ix, year in enumerate(display_years):
        value = output.query(f"year == {year}")["diff"].values[0]
        color = "green" if value > 0 else "red"

        ax.text(
            0.02,
            0.98 - (0.035 * ix),
            f"{year}Y Difference: £{value:,.0f}",
            transform=ax.transAxes,
            verticalalignment="top",
            color=color,
            fontweight="bold",
        )

    # Formatting
    ax.set_title("Cumulative Earnings Difference (New - Current Career)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Earnings Difference (£)")
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


def plot_risk_analysis(risk_metrics: Dict, figsize: tuple = (10, 6)) -> plt.Figure:
    """Create visualization of risk analysis results.

    Parameters:
    -----------
    risk_metrics: Dictionary containing risk analysis results
    figsize: Figure size tuple (width, height)

    Returns:
    --------
    matplotlib.Figure: The created figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot risk factors
    risk_factors = risk_metrics["risk_factors"]
    factor_names = list(risk_factors.keys())
    factor_values = list(risk_factors.values())

    colors = ["red" if v > 0 else "green" for v in factor_values]
    ax1.barh(factor_names, factor_values, color=colors)
    ax1.set_title("Risk Factors")
    ax1.set_xlabel("Risk Score")

    # Plot financial metrics
    metrics = {
        "Max Earnings Drop": risk_metrics["max_earnings_drop"],
        "Opportunity Cost": risk_metrics["opportunity_cost"],
    }

    ax2.bar(metrics.keys(), metrics.values(), color="blue", alpha=0.6)
    ax2.set_title("Financial Impact")
    ax2.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))

    plt.tight_layout()
    return fig
