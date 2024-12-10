import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis.plots import plot_cumulative_earnings, plot_earnings_difference
from src.analysis.risk import RiskAnalyzer
from src.core.career_model import CareerModel
from src.core.career_model_params import CareerModelParams


def calculate_n_year_cost(career_model, n_years):
    # Calculate the summary metrics for the specified number of years
    results = career_model._create_results_dataframe(
        career_model._calculate_current_career(),
        career_model._calculate_new_career(),
        career_model._calculate_course_costs(),
    )
    summary = career_model._calculate_summary_metrics(results, n_years)
    return summary.get("total_nominal_cost", 0)


ANALYSIS_YEARS = 30


def main():
    st.title("Simple Financial Model for Career Change")

    # ================================
    # SIDEBAR
    # ================================
    # Sidebar for parameter inputs
    st.sidebar.header("Model Parameters")
    current_salary = st.sidebar.number_input(
        "Current Starting Salary", value=75_000, key="current_salary"
    )
    current_growth_rate = st.sidebar.slider(
        "Current Growth Rate", 0.0, 0.1, 0.05, key="current_growth_rate"
    )
    retrain_start_date = st.sidebar.date_input(
        "Retrain Start Date",
        value=pd.to_datetime("2025-09-01"),
        key="retrain_start_date",
    )
    course_duration_years = st.sidebar.number_input(
        "Course Duration (Years)", value=3, key="course_duration_years"
    )
    course_annual_cost = st.sidebar.number_input(
        "Course Annual Cost", value=25000, key="course_annual_cost"
    )
    part_time_earnings = st.sidebar.number_input(
        "Part Time Earnings", value=10000, key="part_time_earnings"
    )
    new_career_salary = st.sidebar.number_input(
        "New Career Starting Salary", value=100_000, key="new_career_salary"
    )
    new_growth_rate = st.sidebar.slider(
        "New Career Growth Rate", 0.0, 0.1, 0.05, key="new_growth_rate"
    )
    analysis_years = ANALYSIS_YEARS

    # Create parameters dictionary
    params_dict = {
        "current_starting_salary": current_salary,
        "current_growth_rate": current_growth_rate,
        "retrain_start_date": retrain_start_date.strftime("%Y-%m-%d"),
        "course_duration_years": course_duration_years,
        "course_annual_cost": course_annual_cost,
        "part_time_earnings": part_time_earnings,
        "new_career_starting_salary": new_career_salary,
        "new_career_growth_rate": new_growth_rate,
        "analysis_years": analysis_years,
    }
    params = CareerModelParams.from_dict(params_dict)

    # ================================
    # CALCULATIONS
    # ================================
    # Run the career model
    career_model = CareerModel(params)
    career_model_result = career_model.calculate()
    df, summary = career_model_result

    # Create two columns
    col1, col2 = st.columns(2)

    # Display "Years to Breakeven"
    col1.metric("Years to Breakeven", summary.get("years_to_break_even", "N/A"))

    # Create a placeholder for the metric
    metric_placeholder = col2.empty()

    # Dropdown for selecting N Year Cost
    n_years = col2.selectbox(
        "Select N Year Cost", options=[1, 3, 5, 10, 15, 20, 25, 29], key="n_years"
    )
    n_year_cost = calculate_n_year_cost(career_model, n_years)

    # Update the metric using the placeholder
    metric_placeholder.metric(f"Year {n_years} Cost", f"${n_year_cost:,.0f}")

    # ================================
    # BODY OF RESULTS
    # ================================
    # Display results
    st.subheader("Model Results")
    st.write(summary)

    # Plot results
    st.subheader("Earnings Comparison")
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_cumulative_earnings(ax, results=df, params=params_dict, display_breakeven=True)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_earnings_difference(ax, results=df, n_years=n_years)
    st.pyplot(fig)

    # Risk analysis
    risk_analyzer = RiskAnalyzer(career_model_result)
    risk_metrics = risk_analyzer.calculate_risk_metrics()
    st.subheader("Risk Metrics")
    st.write(risk_metrics)


if __name__ == "__main__":
    main()
