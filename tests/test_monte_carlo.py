"""Tests for Monte Carlo simulation analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.analysis.monte_carlo import MonteCarloAnalysis, ParameterDistribution


def test_parameter_distribution_normal():
    """Test that normal distribution parameters are correctly validated and sampled."""
    dist = ParameterDistribution(
        name="test_param",
        distribution_type="normal",
        params={"mean": 100000, "std": 10000},
    )

    # Test sampling
    samples = dist.sample(size=1000)
    assert len(samples) == 1000
    assert 95000 < np.mean(samples) < 105000  # Should be close to mean
    assert 8000 < np.std(samples) < 12000  # Should be close to std


def test_parameter_distribution_invalid():
    """Test that invalid distribution types and parameters are caught."""
    # Test invalid distribution type
    with pytest.raises(ValueError, match="Unsupported distribution type"):
        ParameterDistribution(
            name="test_param",
            distribution_type="invalid_dist",
            params={"mean": 100000, "std": 10000},
        )

    # Test missing parameters
    with pytest.raises(ValueError, match="Missing required parameters"):
        ParameterDistribution(
            name="test_param",
            distribution_type="normal",
            params={"mean": 100000},  # Missing std
        )


def test_monte_carlo_basic_setup():
    """Test basic Monte Carlo analysis setup and initialization."""
    # Define test parameters
    param_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Salary",
            distribution_type="normal",
            params={"mean": 100000, "std": 10000},
        ),
        "new_career_starting_salary": ParameterDistribution(
            name="New Career Salary",
            distribution_type="normal",
            params={"mean": 120000, "std": 15000},
        ),
    }

    fixed_params = {"current_growth_rate": 0.03, "new_career_growth_rate": 0.05}

    analysis = MonteCarloAnalysis(
        parameter_distributions=param_distributions,
        fixed_parameters=fixed_params,
        time_horizons=[5, 10],
        course_costs=[10000, 20000],
        n_simulations=100,
        random_seed=42,
    )

    assert analysis.n_simulations == 100
    assert len(analysis.time_horizons) == 2
    assert len(analysis.course_costs) == 2


def test_monte_carlo_simulation():
    """Test running Monte Carlo simulation and checking results."""
    # Setup simple test case
    param_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Salary",
            distribution_type="normal",
            params={"mean": 100000, "std": 10000},
        )
    }

    fixed_params = {
        "current_growth_rate": 0.03,
        "new_career_growth_rate": 0.05,
        "new_career_starting_salary": 120000,
        "retrain_start_date": "2024-01-01",
        "course_duration_years": 1.0,
    }

    analysis = MonteCarloAnalysis(
        parameter_distributions=param_distributions,
        fixed_parameters=fixed_params,
        time_horizons=[5],
        course_costs=[10000],
        n_simulations=100,
        random_seed=42,
    )

    # Run simulation
    results = analysis.run_analysis()

    # Check results structure
    assert isinstance(results, dict)
    assert (5, 10000) in results
    assert isinstance(results[(5, 10000)], pd.DataFrame)

    # Check summary statistics
    summary = analysis.get_results_summary()
    assert isinstance(summary, pd.DataFrame)
    assert "probability_profitable" in summary.columns
    assert "mean_cost" in summary.columns


def test_monte_carlo_reproducibility():
    """Test that setting random seed produces reproducible results."""
    param_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Salary",
            distribution_type="normal",
            params={"mean": 100000, "std": 10000},
        )
    }

    fixed_params = {
        "current_growth_rate": 0.03,
        "new_career_growth_rate": 0.05,
        "new_career_starting_salary": 120000,
        "retrain_start_date": "2024-01-01",
        "course_duration_years": 1.0,
    }

    # Create two analyses with same seed
    analysis1 = MonteCarloAnalysis(
        parameter_distributions=param_distributions,
        fixed_parameters=fixed_params,
        time_horizons=[5],
        course_costs=[10000],
        n_simulations=100,
        random_seed=42,
    )

    analysis2 = MonteCarloAnalysis(
        parameter_distributions=param_distributions,
        fixed_parameters=fixed_params,
        time_horizons=[5],
        course_costs=[10000],
        n_simulations=100,
        random_seed=42,
    )

    results1 = analysis1.run_analysis()
    results2 = analysis2.run_analysis()

    # Sort DataFrames before comparison
    df1 = results1[(5, 10000)].sort_index(axis=1)
    df2 = results2[(5, 10000)].sort_index(axis=1)

    # Results should be identical after sorting
    pd.testing.assert_frame_equal(df1, df2)


def test_monte_carlo_plotting():
    """Test that plotting functions run without errors."""
    param_distributions = {
        "current_starting_salary": ParameterDistribution(
            name="Current Salary",
            distribution_type="normal",
            params={"mean": 100000, "std": 10000},
        )
    }

    fixed_params = {
        "current_growth_rate": 0.03,
        "new_career_growth_rate": 0.05,
        "new_career_starting_salary": 120000,
        "retrain_start_date": "2024-01-01",
        "course_duration_years": 1.0,
    }

    analysis = MonteCarloAnalysis(
        parameter_distributions=param_distributions,
        fixed_parameters=fixed_params,
        time_horizons=[5],
        course_costs=[10000],
        n_simulations=100,
        random_seed=42,
    )

    # Run simulation first
    analysis.run_analysis()

    # Test plotting functions
    fig1 = analysis.plot_distribution("total_nominal_cost")
    assert fig1 is not None

    fig2 = analysis.plot_parameter_sensitivity("total_nominal_cost")
    assert fig2 is not None
