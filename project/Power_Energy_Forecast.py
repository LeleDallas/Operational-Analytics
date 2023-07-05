import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import scipy.stats as stats

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
        'path': 'San Leandro Energy.xlsx',
        'period': 31,
        'order': (0, 1, 0),
        'seasonal_order': (1, 1, 0, 96),
        'n': 31
    },
    {
        'path': 'San Leandro Power.xlsx',
        'period': 96,
        'order': (0, 0, 0),
        'seasonal_order': (0, 1, 0, 96),
        'n': 60
    },
]
data_label = None

while data_label == None:
    print("Select which data stream to check:")
    for number, label in stream_options.items():
        print(f"{number}: {label}")

    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)

    data_label = stream_options.get(data_number)

    if data_label:
        print("You selected", data_label,
              "Stream data for Kaiser Permanente - San Leandro")
        break
    else:
        print("Invalid choice. Please select a valid number.")


selectedParameters = parameters[int(data_number)-1]
path = selectedParameters['path']
period = selectedParameters['period']
order = selectedParameters['order']
seasonal_order = selectedParameters['seasonal_order']
n = selectedParameters['n']  # Number of steps ahead to forecast
data_label = None

while data_label == None:
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

df['Date (Local)'] = pd.to_datetime(
    df['Date (Local)'],
    format='%d/%m/%y %H:%M'
)

df.set_index('Date (Local)', inplace=True)

# Convert data_label column to numeric type
df[data_label] = pd.to_numeric(
    df[data_label],
    errors='coerce'
)

decomposition = seasonal_decompose(
    df[data_label],
    model='additive',
    # model='multiplicative',
    period=period
)

seasonality = decomposition.seasonal
residual = decomposition.resid
trend = decomposition.trend
deseasonality = df[data_label] - seasonality
df_smoothed = df[data_label].rolling(window=n, center=True).mean()
# multiplicative decomposition
# deseasonality = df[data_label] / seasonality


# Plot the original data, seasonality, residual, trend, and deseasonality
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# Plot the original data
axes[0, 0].plot(df.index, df[data_label], color='blue')
axes[0, 0].set_title('Original Data')

# Plot the smoothed data
axes[0, 1].plot(df_smoothed.index, df_smoothed, color='gray')
axes[0, 1].set_title('Smoothed Data (Moving Average)')

# Plot the seasonality
axes[1, 0].plot(seasonality.index, seasonality, color='red')
axes[1, 0].set_title('Seasonality')

# Plot the residual
axes[1, 1].plot(residual.index, residual, color='green')
axes[1, 1].set_title('Residual')

# Plot the trend
axes[2, 0].plot(trend.index, trend, color='orange')
axes[2, 0].set_title('Trend')

# Plot the deseasonality
axes[2, 1].plot(deseasonality.index, deseasonality, color='purple')
axes[2, 1].set_title('Deseasonalized')

plt.tight_layout()
plt.show()

# Plot PACF
# fig, ax = plt.subplots(figsize=(10, 4))
# plot_pacf(df[data_label], ax=ax, lags=100)
# [ax.axvline(lag, color="blue", lw=10, alpha=0.1) for lag in range(1, 4)]

# ax.set_title('Partial Autocorrelation Function (PACF)')
# plt.tight_layout()
# plt.show()

# Fit SARIMA model
model = SARIMAX(
    df[data_label],
    order=order,
    seasonal_order=seasonal_order
)
model_fit = model.fit()

# #Residuals
# residuals = model_fit.resid
# # Plot the distribution of residuals
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # Distribution plot
# axes[0].hist(residuals, bins=30, color='blue', alpha=0.7)
# axes[0].set_title('Distribution of Residuals')
# axes[0].set_xlabel('Residuals')
# axes[0].set_ylabel('Frequency')

# # QQ plot
# stats.probplot(residuals, dist="norm", plot=axes[1])
# axes[1].set_title('QQ Plot of Residuals')

# plt.tight_layout()
# plt.show()

# model_median_error = residuals.median()
# print("Model Median Error:", model_median_error)


# Forecast
forecast = model_fit.predict(
    start=len(df),
    end=len(df) + n,
    dynamic=True
)

# Plot the forecast
plt.plot(df.index, df[data_label], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel(data_label)
plt.title(data_label + ' Forecast')
plt.legend()
plt.show()
