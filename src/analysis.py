import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

class PowerConsumptionAnalysis:
    def __init__(self, df):
        self.df = df
        self.hourly_data = None
        self.daily_data = None
        self.daily_profiles = None
        self.clusters = None
        self.anomalies = None
        self.forecast_model = None
        self.forecast_results = None

    def prepare_hourly_data(self):
        """Prepare hourly aggregated data for forecasting."""
        if self.hourly_data is None:
            self.hourly_data = self.df['Global_active_power'].resample('H').mean()
            self.hourly_data.dropna(inplace=True)
        return self.hourly_data

    def prepare_daily_data(self):
        """Prepare daily aggregated data."""
        if self.daily_data is None:
            self.daily_data = self.df['Global_active_power'].resample('D').mean()
            self.daily_data.dropna(inplace=True)
        return self.daily_data

    def get_basic_stats(self):
        """Get basic statistics of Global_active_power."""
        return self.df['Global_active_power'].describe()

    def plot_time_series(self, start_date=None, end_date=None, max_points=10000):
        """Plot time series of Global_active_power."""
        fig, ax = plt.subplots(figsize=(15, 6))

        data = self.df['Global_active_power']
        if start_date and end_date:
            data = data[start_date:end_date]

        # Sample data if too large
        if len(data) > max_points:
            data = data.sample(max_points).sort_index()

        ax.plot(data.index, data.values)
        ax.set_title('Global Active Power Time Series')
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Global Active Power (kW)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_distribution(self):
        """Plot distribution of Global_active_power."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(self.df['Global_active_power'], bins=50, alpha=0.7)
        ax1.set_title('Global Active Power Distribution')
        ax1.set_xlabel('Power (kW)')
        ax1.set_ylabel('Frequency')

        # Box plot
        ax2.boxplot(self.df['Global_active_power'])
        ax2.set_title('Global Active Power Box Plot')
        ax2.set_ylabel('Power (kW)')

        plt.tight_layout()
        return fig

    def analyze_hourly_patterns(self):
        """Analyze hourly consumption patterns."""
        self.df['Hour'] = self.df.index.hour
        hourly_avg = self.df.groupby('Hour')['Global_active_power'].mean()

        fig, ax = plt.subplots(figsize=(12, 6))
        hourly_avg.plot(kind='bar', ax=ax)
        ax.set_title('Average Global Active Power by Hour')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Power (kW)')
        plt.tight_layout()

        # Identify high/low usage hours
        mean_power = hourly_avg.mean()
        high_usage = hourly_avg[hourly_avg > mean_power].index.tolist()
        low_usage = hourly_avg[hourly_avg < mean_power].index.tolist()

        return fig, hourly_avg, high_usage, low_usage

    def analyze_daily_patterns(self):
        """Analyze daily consumption patterns."""
        self.df['DayOfWeek'] = self.df.index.dayofweek
        daily_avg = self.df.groupby('DayOfWeek')['Global_active_power'].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        daily_avg.plot(kind='bar', ax=ax)
        ax.set_title('Average Global Active Power by Day of Week')
        ax.set_xlabel('Day of Week (0=Monday)')
        ax.set_ylabel('Average Power (kW)')
        plt.tight_layout()

        return fig, daily_avg

    def detect_anomalies(self):
        """Detect anomalies using Isolation Forest."""
        daily_data = self.prepare_daily_data()
        daily_data_df = daily_data.to_frame()

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(daily_data_df)
        daily_data_df['anomaly'] = anomalies

        self.anomalies = daily_data_df

        return (anomalies == -1).sum(), daily_data_df

    def create_daily_profiles(self):
        """Create daily consumption profiles for clustering."""
        if self.daily_profiles is not None:
            return self.daily_profiles

        daily_profiles = []
        dates = []

        for date, group in self.df.groupby(self.df.index.date):
            if len(group) >= 24:  # Only full days
                profile = group['Global_active_power'].resample('H').mean()[:24]
                if len(profile) == 24 and not profile.isnull().any():
                    daily_profiles.append(profile.values)
                    dates.append(date)

        self.daily_profiles = np.array(daily_profiles)
        return self.daily_profiles, dates

    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering on daily profiles."""
        if self.daily_profiles is None:
            self.create_daily_profiles()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.daily_profiles)

        self.clusters = clusters
        return clusters, kmeans

    def plot_clusters(self):
        """Plot cluster profiles."""
        if self.clusters is None:
            self.perform_clustering()

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        for i in range(3):
            cluster_data = self.daily_profiles[self.clusters == i]
            if len(cluster_data) > 0:
                avg_profile = cluster_data.mean(axis=0)
                axes[i].plot(avg_profile, label=f'Cluster {i}')
                axes[i].set_title(f'Cluster {i} Average Profile (n={len(cluster_data)})')
                axes[i].set_xlabel('Hour')
                axes[i].set_ylabel('Power (kW)')
                axes[i].legend()

        plt.tight_layout()
        return fig

    def get_cluster_characteristics(self):
        """Get characteristics of each cluster."""
        if self.clusters is None:
            self.perform_clustering()

        characteristics = {}
        for i in range(3):
            cluster_data = self.daily_profiles[self.clusters == i]
            if len(cluster_data) > 0:
                avg_profile = cluster_data.mean(axis=0)
                characteristics[i] = {
                    'size': len(cluster_data),
                    'mean_consumption': avg_profile.mean(),
                    'peak_hour': avg_profile.argmax(),
                    'peak_power': avg_profile.max(),
                    'low_hour': avg_profile.argmin(),
                    'low_power': avg_profile.min()
                }

        return characteristics

    def train_forecasting_model(self, test_size=0.2):
        """Train ARIMA forecasting model."""
        hourly_data = self.prepare_hourly_data()

        train_size = int(len(hourly_data) * (1 - test_size))
        train, test = hourly_data[:train_size], hourly_data[train_size:]

        # Train ARIMA model
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=len(test))
        forecast.index = test.index

        # Evaluate
        mae = mean_absolute_error(test, forecast)
        rmse = np.sqrt(mean_squared_error(test, forecast))

        self.forecast_model = model_fit
        self.forecast_results = {
            'train': train,
            'test': test,
            'forecast': forecast,
            'mae': mae,
            'rmse': rmse
        }

        return self.forecast_results

    def plot_forecast(self, last_n=100):
        """Plot forecasting results."""
        if self.forecast_results is None:
            self.train_forecasting_model()

        fig, ax = plt.subplots(figsize=(15, 6))

        test = self.forecast_results['test'][-last_n:]
        forecast = self.forecast_results['forecast'][-last_n:]

        ax.plot(test.index, test.values, label='Actual', alpha=0.7)
        ax.plot(forecast.index, forecast.values, label='Predicted', alpha=0.7)
        ax.set_title(f'Predicted vs Actual Global Active Power (Last {last_n} hours)')
        ax.set_xlabel('DateTime')
        ax.set_ylabel('Global Active Power (kW)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def categorize_usage(self, power):
        """Rule-based usage categorization."""
        if power < 1.0:
            return "Low Usage", "Consider using energy-efficient appliances to maintain low consumption."
        elif power < 3.0:
            return "Medium Usage", "Monitor usage during peak hours to optimize energy costs."
        else:
            return "High Usage", "Reduce unnecessary electrical loads or shift usage to off-peak hours."

    def get_prediction_example(self):
        """Get an example prediction with categorization."""
        if self.forecast_results is None:
            self.train_forecasting_model()

        last_predicted = self.forecast_results['forecast'].iloc[-1]
        category, suggestion = self.categorize_usage(last_predicted)

        return {
            'predicted_power': last_predicted,
            'category': category,
            'suggestion': suggestion
        }