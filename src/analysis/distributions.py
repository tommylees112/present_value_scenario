"""Parameter distribution definitions for Monte Carlo analysis."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy import stats

ALLOWED_DISTRIBUTIONS = {
    "normal": {"required_params": ["mean", "std"]},
    "truncated_normal": {"required_params": ["mean", "std", "min", "max"]},
    "lognormal": {"required_params": ["mu", "sigma"]},
    "uniform": {"required_params": ["min", "max"]},
}


@dataclass
class ParameterDistribution:
    """
    Represents a probability distribution for a model parameter.

    Parameters:
    -----------
    name: Name of the parameter
    distribution_type: The statistical distribution to use
    params: Dictionary of parameters specific to the distribution type
    random_state: Random state for reproducibility

    Allowed Distributions:
    ---------------------
    normal: requires 'mean' and 'std' parameters
    truncated_normal: requires 'mean', 'std', 'min', and 'max' parameters
    lognormal: requires 'mu' and 'sigma' parameters
    uniform: requires 'min' and 'max' parameters
    """

    name: str
    distribution_type: str
    params: Dict[str, float]
    random_state: Optional[np.random.RandomState] = None

    def __post_init__(self):
        """Validate the distribution type and parameters after initialization."""
        if self.distribution_type not in ALLOWED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution type: {self.distribution_type}. "
                f"Allowed types are: {list(ALLOWED_DISTRIBUTIONS.keys())}"
            )

        required_params = ALLOWED_DISTRIBUTIONS[self.distribution_type][
            "required_params"
        ]
        missing_params = [
            param for param in required_params if param not in self.params
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {self.distribution_type} distribution: "
                f"{missing_params}. Required parameters are: {required_params}. "
                f"Provided parameters: {list(self.params.keys())}"
            )

    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution."""
        if self.distribution_type == "normal":
            return stats.norm.rvs(
                loc=self.params["mean"],
                scale=self.params["std"],
                size=size,
                random_state=self.random_state,
            )
        elif self.distribution_type == "truncated_normal":
            # Calculate the standardized bounds
            a = (self.params["min"] - self.params["mean"]) / self.params["std"]
            b = (self.params["max"] - self.params["mean"]) / self.params["std"]
            return stats.truncnorm.rvs(
                a=a,
                b=b,
                loc=self.params["mean"],
                scale=self.params["std"],
                size=size,
                random_state=self.random_state,
            )
        elif self.distribution_type == "lognormal":
            return stats.lognorm.rvs(
                s=self.params["sigma"],
                scale=np.exp(self.params["mu"]),
                size=size,
                random_state=self.random_state,
            )
        elif self.distribution_type == "uniform":
            return stats.uniform.rvs(
                loc=self.params["min"],
                scale=self.params["max"] - self.params["min"],
                size=size,
                random_state=self.random_state,
            )
        else:
            # This should never happen due to __post_init__ validation
            raise ValueError(
                f"Unsupported distribution type: {self.distribution_type}. "
                f"Allowed types are: {list(ALLOWED_DISTRIBUTIONS.keys())}"
            )
