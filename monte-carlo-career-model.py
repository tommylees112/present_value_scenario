import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import seaborn as sns

@dataclass
class ParameterDistribution:
    """
    Represents a probability distribution for a model parameter
    
    distribution_type: The statistical distribution to use (e.g., 'normal', 'lognormal', 'uniform')
    params: Dictionary of parameters specific to the distribution type
    """
    name: str
    distribution_type: str
    params: Dict[str, float]
    
    def sample(self, size: int = 1) -> np.ndarray:
        """Generate random samples from the distribution"""
        if self.distribution_type == 'normal':
            return stats.norm.rvs(
                loc=self.params['mean'],
                scale=self.params['std'],
                size=size
            )
        elif self.distribution_type == 'lognormal':
            return stats.lognorm.rvs(
                s=self.params['sigma'],
                scale=np.exp(self.params['mu']),
                size=size
            )
        elif self.distribution_type == 'uniform':
            return stats.uniform.rvs(
                loc=self.params['min'],
                scale=self.params['max'] - self.params['min'],
                size=size
            )
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

class MonteCarloCareerAnalysis:
    """
    Analyzes career paths using Monte Carlo simulation with probabilistic inputs
    """
    def __init__(self, 
                 parameter_distributions: Dict[str, ParameterDistribution],
                 fixed_parameters: Dict[str, float],
                 n_simulations: int = 1000,
                 analysis_years: int = 30):
        """
        Initialize the Monte Carlo analysis
        
        parameter_distributions: Dictionary mapping parameter names to their distributions
        fixed_parameters: Dictionary of parameters that don't vary
        n_simulations: Number of Monte Carlo simulations to run
        analysis_years: Number of years to project
        """
        self.parameter_distributions = parameter_distributions
        self.fixed_parameters = fixed_parameters
        self.n_simulations = n_simulations
        self.analysis_years = analysis_years
        
        # Storage for simulation results
        self.simulation_results = []
        self.summary_statistics = None
        
    def run_simulation(self):
        """Execute the Monte Carlo simulation"""
        # Generate parameter samples for each simulation
        parameter_samples = {
            name: dist.sample(self.n_simulations)
            for name, dist in self.parameter_distributions.items()
        }
        
        # Run simulations
        for i in range(self.n_simulations):
            # Combine sampled and fixed parameters for this simulation
            params = {
                **self.fixed_parameters,
                **{name: parameter_samples[name][i] for name in parameter_samples}
            }
            
            # Calculate results for this simulation
            results, summary = calculate_career_paths(**params)
            
            # Store results
            self.simulation_results.append({
                'simulation_id': i,
                'parameters': params,
                'results': results,
                'summary': summary
            })
            
        self._calculate_summary_statistics()
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics across all simulations"""
        # Extract key metrics from all simulations
        metrics = {}
        for metric in self.simulation_results[0]['summary'].keys():
            values = [sim['summary'][metric] for sim in self.simulation_results]
            metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_95': np.percentile(values, 95)
            }
        
        self.summary_statistics = metrics
    
    def plot_distribution(self, metric: str):
        """Plot the distribution of outcomes for a specific metric"""
        values = [sim['summary'][metric] for sim in self.simulation_results]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True)
        plt.title(f'Distribution of {metric}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        # Add percentile lines
        percentiles = [5, 50, 95]
        colors = ['r', 'g', 'r']
        styles = ['--', '-', '--']
        
        for p, c, s in zip(percentiles, colors, styles):
            percentile_value = np.percentile(values, p)
            plt.axvline(percentile_value, color=c, linestyle=s, 
                       label=f'{p}th percentile: {percentile_value:,.0f}')
        
        plt.legend()
        plt.grid(True)
        return plt.gcf()
    
    def plot_parameter_sensitivity(self, metric: str):
        """Create a tornado plot showing parameter sensitivity"""
        parameter_impacts = {}
        
        # Calculate impact of each parameter
        for param_name in self.parameter_distributions.keys():
            values = [sim['summary'][metric] for sim in self.simulation_results]
            parameters = [sim['parameters'][param_name] for sim in self.simulation_results]
            
            # Calculate correlation
            correlation = np.corrcoef(parameters, values)[0, 1]
            parameter_impacts[param_name] = correlation
        
        # Create tornado plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(parameter_impacts))
        
        # Sort by absolute impact
        sorted_impacts = sorted(parameter_impacts.items(), 
                              key=lambda x: abs(x[1]), 
                              reverse=True)
        
        names, impacts = zip(*sorted_impacts)
        
        plt.barh(y_pos, impacts)
        plt.yticks(y_pos, names)
        plt.xlabel('Correlation with ' + metric)
        plt.title('Parameter Sensitivity Analysis')
        plt.grid(True)
        
        return plt.gcf()

# Example usage
def create_example_analysis():
    """Create example analysis with probability distributions"""
    # Define parameter distributions
    parameter_distributions = {
        'current_starting_salary': ParameterDistribution(
            name='Current Starting Salary',
            distribution_type='normal',
            params={'mean': 100000, 'std': 5000}
        ),
        'current_growth_rate': ParameterDistribution(
            name='Current Growth Rate',
            distribution_type='normal',
            params={'mean': 0.05, 'std': 0.01}
        ),
        'new_career_starting_salary': ParameterDistribution(
            name='New Career Starting Salary',
            distribution_type='lognormal',
            params={'mu': np.log(100000), 'sigma': 0.2}
        ),
        'new_career_growth_rate': ParameterDistribution(
            name='New Career Growth Rate',
            distribution_type='normal',
            params={'mean': 0.05, 'std': 0.015}
        ),
        'part_time_earnings': ParameterDistribution(
            name='Part Time Earnings',
            distribution_type='uniform',
            params={'min': 0, 'max': 20000}
        )
    }
    
    # Define fixed parameters
    fixed_parameters = {
        'retrain_start_date': '2025-09-01',
        'course_duration_years': 3,
        'course_annual_cost': 32500,
        'analysis_years': 30,
        'discount_rate': 0.05,
        'inflation_rate': 0.02
    }
    
    return MonteCarloCareerAnalysis(
        parameter_distributions=parameter_distributions,
        fixed_parameters=fixed_parameters,
        n_simulations=1000
    )
