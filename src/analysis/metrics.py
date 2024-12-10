"""Shared metrics calculations for career analysis."""

import warnings
from typing import Dict, List

import numpy as np
from numpy.exceptions import RankWarning


def calculate_elasticity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate elasticity (% change in output per % change in input).

    Uses linear regression on log-transformed values.
    """
    # Avoid log(0) and negative values
    valid_mask = (x > 0) & (y != 0)
    if not valid_mask.any():
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", RankWarning)
        log_x = np.log(x[valid_mask])
        log_y = np.log(np.abs(y[valid_mask]))
        return np.polyfit(log_x, log_y, 1)[0]


def calculate_parameter_sensitivities(
    param_values: np.ndarray,
    output_values: np.ndarray,
    param_name: str,
) -> Dict[str, float]:
    """
    Calculate comprehensive sensitivity metrics for a parameter.

    Parameters:
    -----------
    param_values: Array of parameter values
    output_values: Array of corresponding output values
    param_name: Name of the parameter (for reporting)

    Returns:
    --------
    Dictionary containing:
    - correlation: Pearson correlation coefficient
    - elasticity: % change in output per % change in input
    - impact_10pct: Absolute change in output for 10% parameter change
    - mean_value: Mean parameter value
    """
    # Handle single-value case
    if len(param_values) <= 1:
        return {
            "parameter": param_name,
            "correlation": np.nan,  # Can't calculate correlation with single value
            "elasticity": np.nan,
            "impact_10pct": np.nan,
            "mean_value": param_values[0] if len(param_values) == 1 else np.nan,
        }

    # Calculate correlation with warning suppression for all numpy warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", RankWarning)
        correlation = np.corrcoef(param_values, output_values)[0, 1]
        elasticity = calculate_elasticity(param_values, output_values)

        # Calculate impact of 10% change
        param_mean = np.mean(param_values)
        param_shift = param_mean * 0.1
        gradient = np.polyfit(param_values, output_values, 1)[0]
        impact = gradient * param_shift

    return {
        "parameter": param_name,
        "correlation": correlation,
        "elasticity": elasticity,
        "impact_10pct": impact,
        "mean_value": param_mean,
    }


def calculate_break_even_metrics(
    current_values: np.ndarray,
    new_values: np.ndarray,
    costs: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Calculate break-even points and ratios.

    Parameters:
    -----------
    current_values: Array of current career values
    new_values: Array of new career values
    costs: 2D array of costs for each combination

    Returns:
    --------
    List of dictionaries containing break-even information
    """
    break_even_points = []

    for j, current_value in enumerate(current_values):
        cost_column = costs[:, j]
        if np.any(cost_column < 0):
            break_even_idx = np.where(cost_column < 0)[0][0]
            break_even_value = new_values[break_even_idx]
            break_even_points.append(
                {
                    "current_value": current_value,
                    "break_even_value": break_even_value,
                    "break_even_ratio": break_even_value / current_value,
                }
            )

    return break_even_points


def calculate_summary_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard summary statistics for a set of values.
    """
    summary_stats = {
        "mean": np.mean(values),
        "std_dev": np.std(values),
        "median": np.median(values),
        "percentile_5": np.percentile(values, 5),
        "percentile_95": np.percentile(values, 95),
        "probability_profitable": np.mean(values < 0),
    }

    return summary_stats
