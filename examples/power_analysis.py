import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

print("Loading dataset...")
df = pd.read_csv('household_power_consumption.txt', sep=';', low_memory=False)
print("Dataset loaded. Shape:", df.shape)

# Convert date and time to datetime
print("Preprocessing data...")
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('DateTime', inplace=True)

# Convert numeric columns
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop missing values
df.dropna(inplace=True)
print("After cleaning. Shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())

# Task 1: EDA - Time Series Trend
print("\n=== Task 1: EDA ===")
plt.figure(figsize=(15, 6))
plt.plot(df.index[:10000], df['Global_active_power'][:10000])  # Plot first 10000 points to avoid memory issues
plt.title('Global Active Power Time Series (First 10k points)')
plt.xlabel('DateTime')
plt.ylabel('Global Active Power (kW)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('global_active_power_trend.png', dpi=150)
plt.show()

# Identify missing/abnormal readings
print("Checking for abnormal readings...")
print("Global_active_power stats:")
print(df['Global_active_power'].describe())

# Box plot for outliers
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['Global_active_power'])
plt.title('Global Active Power Distribution')
plt.savefig('power_distribution.png')
plt.show()

# Daily patterns
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month

hourly_avg = df.groupby('Hour')['Global_active_power'].mean()
plt.figure(figsize=(10, 6))
hourly_avg.plot(kind='bar')
plt.title('Average Global Active Power by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Average Power (kW)')
plt.savefig('hourly_patterns.png')
plt.show()

daily_avg = df.groupby('DayOfWeek')['Global_active_power'].mean()
plt.figure(figsize=(10, 6))
daily_avg.plot(kind='bar')
plt.title('Average Global Active Power by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Average Power (kW)')
plt.savefig('daily_patterns.png')
plt.show()

print("High usage hours:", hourly_avg[hourly_avg > hourly_avg.mean()].index.tolist())
print("Low usage hours:", hourly_avg[hourly_avg < hourly_avg.mean()].index.tolist())

# Task 2: Forecasting
print("\n=== Task 2: Time Series Forecasting ===")
# Downsample to hourly for forecasting (reduce data size)
hourly_data = df['Global_active_power'].resample('H').mean()
hourly_data.dropna(inplace=True)

# Prepare data for forecasting (use last 1000 hours for speed)
train_size = int(len(hourly_data) * 0.8)
train, test = hourly_data[:train_size], hourly_data[train_size:]

print("Training on", len(train), "hours, testing on", len(test), "hours")

# Simple ARIMA model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
print("ARIMA model fitted")

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Plot predicted vs actual
plt.figure(figsize=(15, 6))
plt.plot(test.index[-100:], test[-100:], label='Actual', alpha=0.7)
plt.plot(forecast.index[-100:], forecast[-100:], label='Predicted', alpha=0.7)
plt.title('Predicted vs Actual Global Active Power (Last 100 hours)')
plt.xlabel('DateTime')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast_vs_actual.png', dpi=150)
plt.show()

# Task 3: Unsupervised Learning
print("\n=== Task 3: Unsupervised Learning ===")
# Daily consumption profiles
daily_data = df['Global_active_power'].resample('D').mean()
daily_data.dropna(inplace=True)

# Anomaly detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
daily_data_df = daily_data.to_frame()
anomalies = iso_forest.fit_predict(daily_data_df)
daily_data_df['anomaly'] = anomalies

print("Anomalies detected:", (anomalies == -1).sum())

# Clustering on daily profiles (using hourly averages per day)
# Create daily profiles
daily_profiles = []
dates = []
for date, group in df.groupby(df.index.date):
    if len(group) >= 24:  # Only full days
        profile = group['Global_active_power'].resample('H').mean()[:24]  # First 24 hours
        if len(profile) == 24 and not profile.isnull().any():  # Check for NaN
            daily_profiles.append(profile.values)
            dates.append(date)

daily_profiles = np.array(daily_profiles)
print("Daily profiles shape:", daily_profiles.shape)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(daily_profiles)

# Visualize clusters
plt.figure(figsize=(12, 8))
for i in range(3):
    cluster_data = daily_profiles[clusters == i]
    plt.subplot(3, 1, i+1)
    plt.plot(cluster_data.mean(axis=0), label=f'Cluster {i}')
    plt.title(f'Cluster {i} Average Profile')
    plt.xlabel('Hour')
    plt.ylabel('Power (kW)')
    plt.legend()
plt.tight_layout()
plt.savefig('clusters.png', dpi=150)
plt.show()

# Cluster characteristics
for i in range(3):
    cluster_avg = daily_profiles[clusters == i].mean(axis=0)
    print(f"Cluster {i}: Mean daily consumption = {cluster_avg.mean():.3f} kW")
    print(f"  Peak hour: {cluster_avg.argmax()}, Peak power: {cluster_avg.max():.3f}")
    print(f"  Low hour: {cluster_avg.argmin()}, Low power: {cluster_avg.min():.3f}")
    print()

# Task 4: Rule-Based AI
print("\n=== Task 4: Rule-Based AI ===")
def categorize_usage(power):
    if power < 1.0:
        return "Low Usage", "Consider using energy-efficient appliances to maintain low consumption."
    elif power < 3.0:
        return "Medium Usage", "Monitor usage during peak hours to optimize energy costs."
    else:
        return "High Usage", "Reduce unnecessary electrical loads or shift usage to off-peak hours."

# Example with last predicted value
last_predicted = forecast.iloc[-1]
category, suggestion = categorize_usage(last_predicted)
print(f"Predicted Power: {last_predicted:.3f} kW")
print(f"Category: {category}")
print(f"Suggestion: {suggestion}")

print("\nAnalysis complete! Check the saved PNG files for visualizations.")