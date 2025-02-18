import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load your data (replace 'your_data.xlsx' with your actual file path)
data = pd.read_excel('assets/excel/wehdah_data_2021.xlsx')
data['DAM_DAILY_DATE'] = pd.to_datetime(data['DAM_DAILY_DATE'])

# Set the 'DAM_DAILY_DATE' column as the index
data.set_index('DAM_DAILY_DATE', inplace=True)

# Function to forecast using ARIMA
def forecast_variable(variable, data, steps=365):
    model = ARIMA(data[variable], order=(5, 1, 0))  # Adjust (p, d, q) as necessary
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Forecast for each variable
forecast_inflow = forecast_variable('TOTAL_INFLOW', data)
forecast_outflow = forecast_variable('TOTAL_OUTFLOW', data)
forecast_level = forecast_variable('DAM_DAILY_LEVEL', data)

# Generate future dates for the next year (starting from the last date in your dataset)
future_dates = pd.date_range(data.index[-1], periods=366, freq='D')[1:]

# Create a DataFrame with the predicted data for all variables
predicted_data = pd.DataFrame({
    'DAM_DAILY_DATE': future_dates,
    'PREDICTED_TOTAL_INFLOW': forecast_inflow,
    'PREDICTED_TOTAL_OUTFLOW': forecast_outflow,
    'PREDICTED_DAM_DAILY_LEVEL': forecast_level
})

# Save the predicted data to an Excel file
predicted_data.to_excel("predicted_dam_readings.xlsx", index=False)

# Optionally, plot the original and forecasted data for one of the variables
# plt.figure(figsize=(10, 6))
# plt.plot(data.index, data['DAM_DAILY_LEVEL'], label='Original DAM_DAILY_LEVEL', color='blue')
# plt.plot(future_dates, forecast_level, label='Predicted DAM_DAILY_LEVEL', color='red')
# plt.xlabel('Date')
# plt.ylabel('DAM_DAILY_LEVEL')
# plt.legend()
# plt.title('Predicted DAM_DAILY_LEVEL for the Next Year')
# plt.show()

print("Predicted data for TOTAL_INFLOW, TOTAL_OUTFLOW, and DAM_DAILY_LEVEL saved to 'predicted_dam_readings.xlsx'")
