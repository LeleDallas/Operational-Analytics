import statsmodels.api as sm
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed

# Read the data from Excel file
data = pd.read_excel('San Leandro Energy.xlsx', skiprows=10)

# Preprocess the data
df = data.iloc[1:150, :].copy()  # Adjust the range for 2 days of data
df['Date (Local)'] = pd.to_datetime(df['Date (Local)'], format='%d/%m/%y %H:%M')
df.set_index('Date (Local)', inplace=True)
df['Electricity Out'] = pd.to_numeric(df['Electricity Out'], errors='coerce')

# Perform seasonal decomposition
result = seasonal_decompose(df['Electricity Out'], model='additive', period=31)
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# Define the range of values for p, d, q, P, D, Q
p_values = range(0, 2, 2)  # AR order with step of 2
d_values = range(0, 2)  # Differencing order
q_values = range(0, 2, 2)  # MA order with step of 2
P_values = range(0, 2)  # Seasonal AR order
D_values = range(0, 2)  # Seasonal differencing order
Q_values = range(0, 2)  # Seasonal MA order
s_values = [96]  # Seasonal period

# Create all possible combinations of parameters
parameters = itertools.product(p_values, d_values, q_values, P_values, D_values, Q_values, s_values)

# Define the number of parallel jobs
num_jobs = 4  # Adjust the number of jobs based on your machine's capacity

# Define a function to fit SARIMA model for a parameter combination
def fit_sarima(param):
    order = param[:3]
    seasonal_order = param[3:]
    try:
        model = sm.tsa.SARIMAX(seasonal, order=order, seasonal_order=seasonal_order)
        results = model.fit()
        return results.aic, order, seasonal_order
    except:
        return float("inf"), None, None

# Iterate over parameter combinations using parallel processing
results = Parallel(n_jobs=num_jobs)(delayed(fit_sarima)(param) for param in parameters)

# Find the best AIC and corresponding orders
best_aic, best_order, best_seasonal_order = min(results, key=lambda x: x[0])

print("Best order:", best_order)
print("Best seasonal_order:", best_seasonal_order)

# Results for 15 minutes interval data
# Best order: (0, 0, 0)
# Best seasonal_order: (0, 1, 0, 96)