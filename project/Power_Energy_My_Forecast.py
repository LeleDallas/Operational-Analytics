import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def seasonal_decompose(data, period):
    n = len(data)
    seasonal = np.zeros(n)
    residual = np.zeros(n)
    trend = np.zeros(n)
    deseasonality = np.zeros(n)
    
    for i in range(period, n):
        seasonal[i] = np.mean(data[i-period:i])
        residual[i] = data[i] - seasonal[i]
    
    for i in range(period, n):
        trend[i] = np.mean(residual[i-period:i])
        deseasonality[i] = data[i] - seasonal[i] - trend[i]
    
    return seasonal, residual, trend, deseasonality


stream_options = {
    '1': 'Energy',
    '2': 'Power',
    '0': 'Exit',
}

data_options = {
    '1': 'Electricity Out',
    '2': 'Total Output Factor',
    '3': 'AC Efficiency (LHV)',
    '4': 'Heat Rate (HHV)',
    '0': 'Exit',
}

parameters = [
    {
        'path': 'San Leandro Power.xlsx',
        'period': 96,
        'order': (0, 0, 0),
        'seasonal_order': (0, 1, 0, 96),
        'n': 60
    },
    {
        'path': 'San Leandro Energy.xlsx',
        'period': 31,
        'order': (0, 1, 0),
        'seasonal_order': (1, 1, 0, 96),
        'n': 31
    }
]

data_label = None

while data_label is None:
    print("Select which data stream to check:")
    for number, label in stream_options.items():
        print(f"{number}: {label}")

    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)

    data_label = stream_options.get(data_number)

    if data_label:
        print("You selected", data_label, "Stream data for Kaiser Permanente - San Leandro")
        break
    else:
        print("Invalid choice. Please select a valid number.")


selectedParameters = parameters[int(data_number)-1]
path = selectedParameters['path']
period = selectedParameters['period']
order = selectedParameters['order']
seasonal_order = selectedParameters['seasonal_order']
n = selectedParameters['n']

data_label = None

while data_label is None:
    print("Select which data type check:")
    for number, label in data_options.items():
        print(f"{number}: {label}")

    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)
    
    data_label = data_options.get(data_number)

    if data_label:
        print("You selected:", data_label)
        print("Data rendering...")
    else:
        print("Invalid choice. Please select a valid number.")


data = pd.read_excel(path, skiprows=10)

# Preprocess the data
df = data.iloc[1:, :].copy()

df['Date (Local)'] = pd.to_datetime(df['Date (Local)'], format='%d/%m/%y %H:%M')
df.set_index('Date (Local)', inplace=True)
df[data_label] = pd.to_numeric(df[data_label], errors='coerce')

seasonality, residual, trend, deseasonality = seasonal_decompose(df[data_label].values, period)

# Plot the original data, seasonality, residual, trend, and deseasonality
fig, axes = plt.subplots(5, 1, figsize=(10, 9))
axes[0].plot(df.index, df[data_label].values, color='blue')
axes[0].set_title('Original Data')

axes[1].plot(df.index, seasonality, color='red')
axes[1].set_title('Seasonality')

axes[2].plot(df.index, residual, color='green')
axes[2].set_title('Residual')

axes[3].plot(df.index, trend, color='orange')
axes[3].set_title('Trend')

axes[4].plot(df.index, deseasonality, color='purple')
axes[4].set_title('Deseasonalized')

plt.tight_layout()
plt.show()

# Forecast using simple average of previous seasonal values
forecast = deseasonality[-period:] + seasonality[-period:]

# Plot the forecast
plt.plot(df.index, df[data_label].values, label='Actual')
plt.plot(df.index[-period:], forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel(data_label)
plt.title(data_label + ' Forecast')
plt.legend()
plt.show()
