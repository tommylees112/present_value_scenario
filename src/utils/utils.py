"""Shared utility functions across the career analysis package."""

import numpy as np


def thousands_formatter(x: float, p: int) -> str:
    """Format numbers in thousands with comma separator."""
    return f"{x:,.0f}"


def create_salary_grid(
    min_salary: float, max_salary: float, n_points: int
) -> np.ndarray:
    """Create a linear space of salary values."""
    return np.linspace(min_salary, max_salary, n_points)
