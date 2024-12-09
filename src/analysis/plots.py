"""Visualization module for career change analysis."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from ..utils.plot_utils import thousands_formatter
from .distributions import ParameterDistribution


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


def plot_parameter_distributions(
    parameter_distributions: Dict[str, ParameterDistribution],
) -> plt.Figure:
    """Create a side-by-side visualization of parameter distributions."""
    n_params = len(parameter_distributions)
    fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 5))

    # Handle single parameter case
    if n_params == 1:
        axes = [axes]

    fig.suptitle("Probability Distributions of Model Parameters", fontsize=12, y=1.05)

    # Generate x-ranges for each distribution
    for ax, (param_name, dist) in zip(axes, parameter_distributions.items()):
        if dist.distribution_type in ["normal", "truncated_normal"]:
            # Generate points for normal/truncated normal distribution
            x = np.linspace(
                dist.params["mean"] - 6 * dist.params["std"],
                dist.params["mean"] + 6 * dist.params["std"],
                1000,
            )
            if dist.distribution_type == "truncated_normal":
                a = (
                    dist.params.get("min", -np.inf) - dist.params["mean"]
                ) / dist.params["std"]
                b = (
                    dist.params.get("max", np.inf) - dist.params["mean"]
                ) / dist.params["std"]
                y = stats.truncnorm.pdf(
                    x, a, b, loc=dist.params["mean"], scale=dist.params["std"]
                )
            else:
                y = stats.norm.pdf(x, dist.params["mean"], dist.params["std"])

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Add vertical lines for mean and standard deviations
            ax.axvline(dist.params["mean"], color="r", linestyle="--", label="Mean")
            ax.axvline(
                dist.params["mean"] + dist.params["std"],
                color="g",
                linestyle=":",
                label="±1 SD",
            )
            ax.axvline(
                dist.params["mean"] - dist.params["std"], color="g", linestyle=":"
            )

            # Add annotations
            ax.text(
                0.05,
                0.95,
                f"Mean: {dist.params['mean']:,.2f}\nSD: {dist.params['std']:,.2f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # Set wider x limits (4 standard deviations)
            margin = 4 * dist.params["std"]
            ax.set_xlim(dist.params["mean"] - margin, dist.params["mean"] + margin)

        elif dist.distribution_type == "lognormal":
            # Generate points for lognormal distribution
            x = np.linspace(
                0, np.exp(dist.params["mu"] + 6 * dist.params["sigma"]), 1000
            )
            y = stats.lognorm.pdf(
                x, s=dist.params["sigma"], scale=np.exp(dist.params["mu"])
            )

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Calculate and plot key statistics
            median = np.exp(dist.params["mu"])
            mode = np.exp(dist.params["mu"] - dist.params["sigma"] ** 2)
            mean = np.exp(dist.params["mu"] + dist.params["sigma"] ** 2 / 2)

            ax.axvline(median, color="r", linestyle="--", label="Median")
            ax.axvline(mode, color="g", linestyle=":", label="Mode")

            # Add annotations
            ax.text(
                0.05,
                0.95,
                f"Median: {median:,.0f}\nMode: {mode:,.0f}\nMean: {mean:,.0f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        elif dist.distribution_type == "uniform":
            # Generate points for uniform distribution
            margin = 0.2 * (dist.params["max"] - dist.params["min"])
            x = np.linspace(
                dist.params["min"] - margin,
                dist.params["max"] + margin,
                1000,
            )
            y = stats.uniform.pdf(
                x, dist.params["min"], dist.params["max"] - dist.params["min"]
            )

            # Plot the distribution
            ax.plot(x, y, "b-", lw=2, label="PDF")

            # Add vertical lines for bounds
            ax.axvline(dist.params["min"], color="r", linestyle="--", label="Bounds")
            ax.axvline(dist.params["max"], color="r", linestyle="--")

            # Add annotations
            mean = (dist.params["min"] + dist.params["max"]) / 2
            ax.text(
                0.05,
                0.95,
                f"Min: {dist.params['min']:,.0f}\nMax: {dist.params['max']:,.0f}\nMean: {mean:,.0f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8),
            )

        # Customize each subplot
        ax.set_title(f"{dist.name}\n({param_name})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Remove scientific notation from axis
        ax.ticklabel_format(style="plain", axis="x")

    plt.tight_layout()
    return fig
