import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data, get_data_info
from src.analysis import PowerConsumptionAnalysis
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Household Power Consumption Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_resource
def get_analyzer():
    df = load_data()
    if df is not None:
        return PowerConsumptionAnalysis(df)
    return None

def main():
    st.markdown('<div class="main-header">⚡ Household Power Consumption Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    **Dataset**: [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

    This dashboard provides comprehensive analysis of household electricity consumption patterns including:
    - Exploratory Data Analysis
    - Time Series Forecasting
    - Anomaly Detection & Clustering
    - Rule-Based Energy Usage Recommendations
    """)

    # Load data
    with st.spinner("Loading dataset..."):
        analyzer = get_analyzer()

    if analyzer is None:
        st.error("Failed to load dataset. Please check if 'household_power_consumption.txt' exists.")
        return

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Analysis Section:",
                           ["Overview", "EDA - Time Series", "EDA - Patterns",
                            "Time Series Forecasting", "Anomaly Detection",
                            "Clustering Analysis", "AI Recommendations"])

    # Overview
    if page == "Overview":
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

        info = get_data_info(analyzer.df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{info['shape'][0]:,}")
        with col2:
            st.metric("Date Range", f"{info['date_range'][0].date()} to {info['date_range'][1].date()}")
        with col3:
            st.metric("Missing Values", info['missing_values'])
        with col4:
            st.metric("Features", len(info['columns']))

        st.subheader("Basic Statistics")
        stats = analyzer.get_basic_stats()
        st.dataframe(stats)

        st.subheader("Data Sample")
        st.dataframe(analyzer.df.head())

    # EDA - Time Series
    elif page == "EDA - Time Series":
        st.markdown('<div class="section-header">📈 Time Series Analysis</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", analyzer.df.index.min().date())
        with col2:
            end_date = st.date_input("End Date", analyzer.df.index.max().date())

        if st.button("Generate Time Series Plot"):
            with st.spinner("Generating plot..."):
                fig = analyzer.plot_time_series(start_date, end_date)
                st.pyplot(fig)

        st.markdown('<div class="section-header">📊 Distribution Analysis</div>', unsafe_allow_html=True)
        if st.button("Show Distribution Plots"):
            with st.spinner("Generating plots..."):
                fig = analyzer.plot_distribution()
                st.pyplot(fig)

    # EDA - Patterns
    elif page == "EDA - Patterns":
        st.markdown('<div class="section-header">⏰ Hourly Patterns</div>', unsafe_allow_html=True)

        if st.button("Analyze Hourly Patterns"):
            with st.spinner("Analyzing patterns..."):
                fig, hourly_avg, high_usage, low_usage = analyzer.analyze_hourly_patterns()
                st.pyplot(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("High Usage Hours")
                    st.write(", ".join(map(str, high_usage)))
                with col2:
                    st.subheader("Low Usage Hours")
                    st.write(", ".join(map(str, low_usage)))

        st.markdown('<div class="section-header">📅 Daily Patterns</div>', unsafe_allow_html=True)

        if st.button("Analyze Daily Patterns"):
            with st.spinner("Analyzing patterns..."):
                fig, daily_avg = analyzer.analyze_daily_patterns()
                st.pyplot(fig)

    # Time Series Forecasting
    elif page == "Time Series Forecasting":
        st.markdown('<div class="section-header">🔮 Time Series Forecasting</div>', unsafe_allow_html=True)

        st.write("Predicting next-hour Global_active_power using ARIMA model")

        if st.button("Train Forecasting Model"):
            with st.spinner("Training model... (this may take a few minutes)"):
                results = analyzer.train_forecasting_model()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"{results['mae']:.4f}")
                with col2:
                    st.metric("RMSE", f"{results['rmse']:.4f}")
                with col3:
                    st.metric("Training Hours", f"{len(results['train']):,}")

        if analyzer.forecast_results is not None:
            st.markdown('<div class="section-header">📈 Forecast Results</div>', unsafe_allow_html=True)

            last_n = st.slider("Show last N hours", 50, 500, 100)
            fig = analyzer.plot_forecast(last_n)
            st.pyplot(fig)

    # Anomaly Detection
    elif page == "Anomaly Detection":
        st.markdown('<div class="section-header">🔍 Anomaly Detection</div>', unsafe_allow_html=True)

        st.write("Detecting unusual consumption patterns using Isolation Forest")

        if st.button("Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                n_anomalies, anomaly_df = analyzer.detect_anomalies()

                st.metric("Anomalous Days Detected", n_anomalies)

                # Show anomalous dates
                anomalous_dates = anomaly_df[anomaly_df['anomaly'] == -1].index
                st.subheader("Anomalous Dates")
                st.dataframe(pd.DataFrame({'Date': anomalous_dates, 'Power': anomaly_df.loc[anomalous_dates, 'Global_active_power']}))

    # Clustering Analysis
    elif page == "Clustering Analysis":
        st.markdown('<div class="section-header">📊 Clustering Analysis</div>', unsafe_allow_html=True)

        st.write("Clustering daily consumption profiles using K-means")

        n_clusters = st.selectbox("Number of Clusters", [3, 4, 5], index=0)

        if st.button("Perform Clustering"):
            with st.spinner("Performing clustering..."):
                clusters, kmeans = analyzer.perform_clustering(n_clusters)

                st.success(f"Clustered {len(clusters)} daily profiles into {n_clusters} groups")

                fig = analyzer.plot_clusters()
                st.pyplot(fig)

                characteristics = analyzer.get_cluster_characteristics()
                st.subheader("Cluster Characteristics")

                for cluster_id, char in characteristics.items():
                    with st.expander(f"Cluster {cluster_id}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Size", char['size'])
                        with col2:
                            st.metric("Avg Consumption", f"{char['mean_consumption']:.3f} kW")
                        with col3:
                            st.metric("Peak Hour", char['peak_hour'])

                        st.write(f"Peak Power: {char['peak_power']:.3f} kW at hour {char['peak_hour']}")
                        st.write(f"Low Power: {char['low_power']:.3f} kW at hour {char['low_hour']}")

    # AI Recommendations
    elif page == "AI Recommendations":
        st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)

        st.write("Rule-based system for energy usage categorization and recommendations")

        if st.button("Generate Recommendation Example"):
            with st.spinner("Generating recommendation..."):
                example = analyzer.get_prediction_example()

                st.subheader("Example Prediction & Recommendation")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Power", f"{example['predicted_power']:.3f} kW")
                with col2:
                    st.metric("Category", example['category'])

                st.info(f"💡 **Suggestion**: {example['suggestion']}")

        st.markdown("""
        ### Usage Categories:
        - **Low Usage** (< 1.0 kW): Energy-efficient household
        - **Medium Usage** (1.0 - 3.0 kW): Moderate consumption
        - **High Usage** (> 3.0 kW): High consumption, potential for optimization
        """)

if __name__ == "__main__":
    main()