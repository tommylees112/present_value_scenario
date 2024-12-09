# 🎯 Career Change Financial Analyzer

A Python toolkit that helps answer the eternal question: "Should I quit my job to retrain for a new career?"

## 🤔 What Does It Do?

This toolkit helps you model and visualize the financial impact of a career change, including:
- Long-term earnings projections for both career paths
- The true cost of retraining (including opportunity costs)
- Risk analysis and "what-if" scenarios
- Break-even point analysis
- Monte Carlo simulations for uncertainty modeling

## 🏗️ Project Structure
```
├── src/
│ ├── analysis/
│ │ ├── plots.py # Visualization tools
│ │ ├── risk.py # Risk assessment calculations
│ │ ├── monte_carlo.py # Probabilistic modeling
│ │ └── parameter_space.py # Parameter sensitivity analysis
│ ├── core/
│ │ ├── career_model.py # Core calculation engine
│ │ └── career_model_params.py # Parameter definitions
│ └── utils/
│ └── plot_utils.py # Plotting utilities
├── examples/
│ ├── simple_financial_model.py
│ ├── parameter_space_example.py
│ └── probabilistic_example.py
```
## 🚀 How It Works

### Core Model
The heart of the system is `career_model.py`, which calculates:
- Earnings trajectories for both careers
- Training costs and part-time earnings during study
- Cumulative financial position over time

### Visualization
`plots.py` creates beautiful visualizations showing:
- Side-by-side earnings comparisons
- Break-even point analysis
- Risk factor visualization
- Monte Carlo simulation results

### Risk Analysis
The toolkit includes sophisticated risk assessment tools:
- Sensitivity analysis to identify critical factors
- Monte Carlo simulations to model uncertainty
- Risk scoring based on multiple factors

## 📊 Example Usage
```python
from src.core.career_model import CareerModel
from src.analysis.plots import plot_earnings_comparison
Define your career change scenario
params = {
"current_starting_salary": 45000,
"current_growth_rate": 0.03,
"new_career_starting_salary": 65000,
"new_career_growth_rate": 0.05,
"course_duration_years": 1,
"course_annual_cost": 9000,
"part_time_earnings": 10000
}
Run the analysis
model = CareerModel(params)
results = model.calculate_earnings(years=30)
Visualize the results
fig = plot_earnings_comparison(results, params)
fig.savefig('career_comparison.png')

```
## 🎓 Advanced Features

### Parameter Space Analysis
Explore how different variables affect the outcome:
```python
from src.analysis.parameter_space import analyze_parameter_sensitivity
sensitivity_results = analyze_parameter_sensitivity(
base_params=params,
param_ranges={'new_career_growth_rate': [0.02, 0.08]}
)
```

### Monte Carlo Simulation
Model uncertainty in your career change decision:
```python
from src.analysis.monte_carlo import run_monte_carlo_simulation
simulation_results = run_monte_carlo_simulation(
base_params=params,
num_simulations=1000
)
```
