"""Utility functions for plotting."""


def thousands_formatter(x, pos):
    """Format numbers in thousands with comma separator."""
    return f"£{x*1e-3:,.0f}k"
