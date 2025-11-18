# Household Power Consumption Analysis

Interactive Streamlit dashboard for analyzing household electricity consumption patterns.

## Dataset

Uses the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) or

fetch data by using the code

from ucimlrepo

import fetch_ucirepo 

individual_household_electric_power_consumption = fetch_ucirepo(id=235) 

X = individual_household_electric_power_consumption.data.features 

y = individual_household_electric_power_consumption.data.targets 

print(individual_household_electric_power_consumption.metadata)  

print(individual_household_electric_power_consumption.variables) 


## Setup

1. Place the dataset file `household_power_consumption.txt` in the `data/` directory.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- Exploratory Data Analysis (EDA)
- Time Series Forecasting
- Anomaly Detection
- Clustering Analysis
- AI-powered Recommendations

## Project Structure

- `app.py`: Main Streamlit application
- `src/`: Source code modules
- `data/`: Dataset files
- `examples/`: Original analysis scripts
- `results/`: Generated plots and results

<img width="1920" height="932" alt="image" src="https://github.com/user-attachments/assets/b47e9132-b1ac-4920-a888-1d132a2be665" />


<img width="1920" height="932" alt="image" src="https://github.com/user-attachments/assets/f88aad93-27d6-4c99-90f5-6f9d4e6a861f" />

