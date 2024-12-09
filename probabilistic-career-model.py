from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fam1 import calculate_career_paths


@dataclass
class CareerScenario:
    """
    Represents a single possible career scenario with its probability
    """

    name: str
    probability: float
    current_starting_salary: float
    current_growth_rate: float
    retrain_start_date: str
    course_duration_years: float
    course_annual_cost: float
    part_time_earnings: float
    new_career_starting_salary: float
    new_career_growth_rate: float

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("Probability must be between 0 and 1")


class ProbabilisticCareerAnalysis:
    """
    Analyzes multiple possible career scenarios with their associated probabilities
    """

    def __init__(
        self,
        scenarios: List[CareerScenario],
        analysis_years: int = 30,
        discount_rate: float = 0.05,
        inflation_rate: float = 0.02,
    ):
        self.scenarios = scenarios
        self.analysis_years = analysis_years
        self.discount_rate = discount_rate
        self.inflation_rate = inflation_rate

        # Validate total probability sums to 1
        total_prob = sum(scenario.probability for scenario in scenarios)
        if not np.isclose(total_prob, 1.0, rtol=1e-5):
            raise ValueError(f"Scenario probabilities must sum to 1, got {total_prob}")

        # Calculate results for each scenario
        self.scenario_results = {}
        self.scenario_summaries = {}
        self._calculate_all_scenarios()

    def _calculate_all_scenarios(self):
        """Calculate results for each scenario"""
        for scenario in self.scenarios:
            # Use our existing calculation function with scenario parameters
            results, summary = calculate_career_paths(
                current_starting_salary=scenario.current_starting_salary,
                current_growth_rate=scenario.current_growth_rate,
                retrain_start_date=scenario.retrain_start_date,
                course_duration_years=scenario.course_duration_years,
                course_annual_cost=scenario.course_annual_cost,
                part_time_earnings=scenario.part_time_earnings,
                new_career_starting_salary=scenario.new_career_starting_salary,
                new_career_growth_rate=scenario.new_career_growth_rate,
                analysis_years=self.analysis_years,
                discount_rate=self.discount_rate,
                inflation_rate=self.inflation_rate,
            )
            self.scenario_results[scenario.name] = results
            self.scenario_summaries[scenario.name] = summary

    def calculate_expected_values(self) -> Dict[str, float]:
        """Calculate probability-weighted expected values for key metrics"""
        expected_values = {}

        # For each metric in our summary
        for metric in self.scenario_summaries[self.scenarios[0].name].keys():
            weighted_sum = sum(
                self.scenario_summaries[scenario.name][metric] * scenario.probability
                for scenario in self.scenarios
            )
            expected_values[metric] = weighted_sum

        return expected_values

    def calculate_value_at_risk(
        self, confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calculate Value at Risk for key metrics
        With our three scenarios, here's what happens:

        Worst Case (30% probability)
        Base Case (50% probability)
        Best Case (20% probability)

        When we try to find the 95% confidence level,
        we're looking for the value where the cumulative
        probability first exceeds 5% (1 - 0.95). With just
        three scenarios, we'll always select the worst case
        scenario because 30% > 5%.
        """
        var_metrics = {}

        # For each metric, calculate VaR
        for metric in self.scenario_summaries[self.scenarios[0].name].keys():
            values = [self.scenario_summaries[s.name][metric] for s in self.scenarios]
            probs = [s.probability for s in self.scenarios]

            # Sort values and accumulate probabilities
            sorted_pairs = sorted(zip(values, probs))
            cum_prob = 0

            for value, prob in sorted_pairs:
                cum_prob += prob
                if cum_prob >= 1 - confidence_level:
                    var_metrics[metric] = value
                    break

        return var_metrics

    def plot_scenario_comparison(self):
        """Create visualization comparing all scenarios"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot cumulative earnings for each scenario
        for scenario in self.scenarios:
            results = self.scenario_results[scenario.name]
            ax1.plot(
                results["year"],
                results["earnings_diff_nominal_cumulative"],
                label=f"{scenario.name} (p={scenario.probability:.2f})",
                alpha=0.7,
            )

        ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax1.set_title("Cumulative Earnings Difference by Scenario")
        ax1.set_xlabel("Years")
        ax1.set_ylabel("Earnings Difference")
        ax1.legend()
        ax1.grid(True)
        sns.despine(ax=ax1)

        # Create probability distribution of outcomes
        # Tuple[scenario.name, final_earnings_diff, scenario.probability]
        # get the "earnings_diff_nominal_cumulative" for each scenario, each result
        # create an output table for each scenario and years: 3, 5, 10, 15, 30
        scenario_results = defaultdict(list)

        for scenario in self.scenarios:
            for yr in [3, 5, 10, 15, 29]:
                scenario_results["scenario"].append(scenario.name)
                scenario_results["year"].append(yr)
                scenario_results["earnings_diff_nominal_cumulative"].append(
                    self.scenario_results[scenario.name]
                    .query(f"year == {yr}")["earnings_diff_nominal_cumulative"]
                    .values[0]
                )

        scenario_df = pd.DataFrame(scenario_results)

        sns.barplot(
            data=scenario_df,
            x="scenario",
            y="earnings_diff_nominal_cumulative",
            hue="year",
            palette="viridis",  # You can choose any color palette you like
            alpha=0.7,
            ax=ax2,
        )

        ax2.set_title("Cumulative Earnings Difference by Scenario and Year")
        ax2.set_xlabel("Scenarios")
        ax2.set_ylabel("Cumulative Earnings Difference (£)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

        ax2.legend(title="Year")
        sns.despine(ax=ax2)

        plt.tight_layout()
        return fig

    def generate_summary_report(self) -> str:
        """Generate a detailed summary report of the analysis"""
        expected_values = self.calculate_expected_values()
        var_metrics = self.calculate_value_at_risk()

        report = [
            "Probabilistic Career Path Analysis Summary",
            "=========================================\n",
        ]

        report.append("Scenario Overview:")
        for scenario in self.scenarios:
            report.append(
                f"\n{scenario.name} (Probability: {scenario.probability:.1%})"
            )
            report.append(
                f"  - Current Path: Start ${scenario.current_starting_salary:,.0f}, {scenario.current_growth_rate:.1%} growth"
            )
            report.append(
                f"  - New Path: Start ${scenario.new_career_starting_salary:,.0f}, {scenario.new_career_growth_rate:.1%} growth"
            )

        report.append("\nExpected Values:")
        for metric, value in expected_values.items():
            report.append(f"  - {metric}: ${value:,.0f}")

        report.append("\nValue at Risk (95% confidence):")
        for metric, value in var_metrics.items():
            report.append(f"  - {metric}: ${value:,.0f}")

        return "\n".join(report)


# Example usage:
def create_example_scenarios():
    """Create example scenarios for demonstration"""
    return [
        CareerScenario(
            name="Base Case",
            probability=0.5,
            current_starting_salary=100000,
            current_growth_rate=0.05,
            retrain_start_date="2025-09-01",
            course_duration_years=3,
            course_annual_cost=32500,
            part_time_earnings=0,
            new_career_starting_salary=100000,
            new_career_growth_rate=0.05,
        ),
        CareerScenario(
            name="Optimistic",
            probability=0.25,
            current_starting_salary=100000,
            current_growth_rate=0.05,
            retrain_start_date="2025-09-01",
            course_duration_years=3,
            course_annual_cost=32500,
            part_time_earnings=0,
            new_career_starting_salary=120000,
            new_career_growth_rate=0.07,
        ),
        CareerScenario(
            name="Pessimistic",
            probability=0.25,
            current_starting_salary=100000,
            current_growth_rate=0.05,
            retrain_start_date="2025-09-01",
            course_duration_years=3,
            course_annual_cost=32500,
            part_time_earnings=0,
            new_career_starting_salary=80000,
            new_career_growth_rate=0.04,
        ),
    ]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("MacOSX")

    # Create the analysis object
    scenarios = create_example_scenarios()
    analysis = ProbabilisticCareerAnalysis(scenarios)

    # Generate various insights
    expected_values = analysis.calculate_expected_values()
    var_metrics = analysis.calculate_value_at_risk()
    report = analysis.generate_summary_report()

    print(report)

    # Create visualizations
    analysis.plot_cumprob_VaR("Total Nominal Cost")
    plt.show()

    analysis.plot_scenario_comparison()
    plt.show()
