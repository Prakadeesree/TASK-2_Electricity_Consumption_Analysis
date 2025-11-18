import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    """Load and preprocess the household power consumption dataset."""
    try:
        # Load the dataset
        df = pd.read_csv('data/household_power_consumption.txt', sep=';', low_memory=False)

        # Convert date and time to datetime
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
        df.set_index('DateTime', inplace=True)

        # Convert numeric columns
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop missing values
        df.dropna(inplace=True)

        return df

    except FileNotFoundError:
        st.error("Dataset file 'data/household_power_consumption.txt' not found. Please ensure the data file is placed in the data/ directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_data_info(df):
    """Get basic information about the dataset."""
    if df is None:
        return {}

    info = {
        'shape': df.shape,
        'date_range': (df.index.min(), df.index.max()),
        'missing_values': df.isnull().sum().sum(),
        'columns': df.columns.tolist()
    }
    return info