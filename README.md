# Household Power Consumption Analysis

Interactive Streamlit dashboard for analyzing household electricity consumption patterns.

This project focuses on predicting energy usage, detecting anomalies, and grouping daily consumption patterns using ML models. I used two methods and got **better accuracy of** **96.29%** with the MLP regressor (Consumption Analysis 2.ipynb) 

---

## SUPERVISED LEARNING

### **Method 1. Random Forest Regressor**
File: Consumption Analysis Task.ipynb
- Uses past 24 hours as input window.
- No scaling required.
- Stable baseline model.
- **Performance:** MAE = 0.5018, RMSE = 0.7361

### **Method 2. MLP Regressor (Neural Network)**
File: Consumption Analysis 2.ipynb
- Uses scaled 24-hour windows.
- Learns smooth and complex time patterns better.
- **Performance:** MAE = 0.0785, RMSE = 0.2130  
- **Much more accurate than Random Forest**

---

## Difference (Short & Simple)
- **Random Forest:** Easy, stable baseline but less accurate for time-series.
- **MLP:** Needs scaling but gives **much better predictions**.
- ✔️ **MLP is the better forecasting model** in this project.

---

## UNSUPERVISED LEARNING

### **Anomaly Detection (Isolation Forest)**
- Detects abnormal power usage points.
- Total anomalies detected: **22365**

### **Clustering (KMeans)**
- Groups days into 3 usage clusters:
  - Cluster 0 → Low usage  
  - Cluster 1 → Medium usage  
  - Cluster 2 → High usage  

---

## RULE-BASED CATEGORY SYSTEM
Automatically classifies predicted usage into:
- Low Usage  
- Medium Usage  
- High Usage  

Classification accuracy: **96.29%**

---

## Result 
- **MLP > Random Forest** for forecasting accuracy.  
- RF is useful as a quick baseline.  
- Unsupervised models help identify unusual days and patterns.


## Dataset

Uses the [Individual Household Electric Power Consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) or

## fetch data by using the code

`from ucimlrepo

import fetch_ucirepo 

individual_household_electric_power_consumption = fetch_ucirepo(id=235) 

X = individual_household_electric_power_consumption.data.features 

y = individual_household_electric_power_consumption.data.targets 

print(individual_household_electric_power_consumption.metadata)  

print(individual_household_electric_power_consumption.variables) `


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


### Final Conclusion

- **With a large TXT dataset given → MLP gives better and smoother accuracy for this project.**
- **If the same dataset were cleaned and stored as CSV → Random Forest accuracy could improve and might match or beat MLP for simpler patterns.**


