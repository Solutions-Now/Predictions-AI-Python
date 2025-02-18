import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import time
import sys

# Function to display the progress bar with percentage
def show_progress(current, total, task_name="Processing"):
    progress = (current / total) * 100
    bar = ('#' * int(progress // 2)).ljust(50)  # Adjust the bar length
    sys.stdout.write(f'\r{task_name}: [{bar}] {progress:.2f}%')
    sys.stdout.flush()

# Loading message to show the process is starting
print("Loading data and starting forecast... Please wait.")

# Load the data from the Excel file (adjust the file path)
data = pd.read_excel('assets/excel/jena_climate_2009_2015.xlsx')

# Clean column names (remove any leading/trailing spaces)
data.columns = data.columns.str.strip()

# Convert 'Date Time' column to datetime format
data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')

# Sort by date to ensure the index is monotonic
data.sort_values('Date Time', inplace=True)

# Set 'Date Time' as the index
data.set_index('Date Time', inplace=True)

# 🔴 FIX: Remove duplicate timestamps to avoid reindexing issues
data = data[~data.index.duplicated(keep='first')]

# Set the frequency of the DateTime index (assuming 10-minute intervals)
data = data.asfreq('10min')  # Adjust if needed

# Function to forecast a variable using ARIMA
def forecast_variable(variable, data, steps=365):
    model = ARIMA(data[variable], order=(5, 1, 0))  # Adjust (p, d, q) based on the dataset
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Forecast for multiple columns
columns_to_forecast = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)']
total_tasks = len(columns_to_forecast)
print(f"Starting forecast for {total_tasks} variables...")

# Simulating the loading with progress percentage
for i, variable in enumerate(columns_to_forecast):
    print(f"\nForecasting for '{variable}'...")
    forecast = forecast_variable(variable, data)
    show_progress(i + 1, total_tasks, task_name=f"Forecasting {variable}")
    time.sleep(1)  # Simulate some time for each forecast

# Generate future dates for the next year (starting from the last date in your dataset)
future_dates = pd.date_range(data.index[-1], periods=366, freq='10min')[1:]  # Adjust '10min' for 10-minute intervals

# Ensure the forecast length matches the number of future dates
forecast_length = len(future_dates)

# Adjust if the number of forecast steps is smaller than required
forecast_p = forecast[:forecast_length]
forecast_T = forecast[:forecast_length]
forecast_Tpot = forecast[:forecast_length]
forecast_Tdew = forecast[:forecast_length]

# Create a DataFrame with the predicted data
predicted_data = pd.DataFrame({
    'Date Time': future_dates,
    'PREDICTED_p (mbar)': forecast_p,
    'PREDICTED_T (degC)': forecast_T,
    'PREDICTED_Tpot (K)': forecast_Tpot,
    'PREDICTED_Tdew (degC)': forecast_Tdew
})

# Save the predicted data to an Excel file
print("\nSaving predicted data to 'predicted_weather_readings.xlsx'...")
predicted_data.to_excel('predicted_weather_readings.xlsx', index=False)

# Final message when the script finishes
print("\nPredicted data saved to 'predicted_weather_readings.xlsx'. Process completed.")
