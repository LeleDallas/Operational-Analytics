import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# Define data options
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

# Define parameters for different datasets
parameters = {
    '2': {
        'path': 'San Leandro Power.xlsx',
        'period': 96,
        'n': 20
    },
    '1': {
        'path': 'San Leandro Energy.xlsx',
        'period': 31,
        'n': 31
    }
}

# Select data stream
while True:
    print("Select which data stream to check:")
    for number, label in stream_options.items():
        print(f"{number}: {label}")
    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)
    data_label = stream_options.get(data_number)
    if data_label:
        print(
            f"You selected {data_label} Stream data for Kaiser Permanente - San Leandro")
        break
    else:
        print("Invalid choice. Please select a valid number.")

selected_parameters = parameters[data_number]
path = selected_parameters['path']
print(path)

# Select data type
while True:
    print("Select which data type to check:")
    for number, label in data_options.items():
        print(f"{number}: {label}")
    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)
    data_label = data_options.get(data_number)
    if data_label:
        print(f"You selected: {data_label}")
        print("Data rendering...")
        break
    else:
        print("Invalid choice. Please select a valid number.")

n = selected_parameters['n']  # Number of steps ahead to forecast

# Read and preprocess the data
data = pd.read_excel(path, skiprows=10)
df = data.iloc[1:, :].copy()
df['Date (Local)'] = pd.to_datetime(
    df['Date (Local)'], format='%d/%m/%y %H:%M')
df.set_index('Date (Local)', inplace=True)
df[data_label] = pd.to_numeric(df[data_label], errors='coerce')

# Create lagged features
for i in range(1, n+1):
    df[f'lag_{i}'] = df[data_label].shift(i-1)

# Remove missing values
df.drop("Unnamed: 0", axis=1, inplace=True)
df = df.dropna()

# Split the data into training and testing sets
train_size = int(0.8 * len(df))
train = df[:train_size]
test = df[train_size:]

# Split the features and target variables
X_train = train.iloc[:, 1:]
y_train = train[data_label]
X_test = test.iloc[:, 1:]
y_test = test[data_label]

# Define the hyperparameters grid
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(
    DecisionTreeRegressor(),
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)

# Train the decision tree regressor with the best hyperparameters
model = DecisionTreeRegressor(**best_params)
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Calculate root mean squared error
train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

print("Root Mean Squared Error (Train):", train_rmse)
print("Root Mean Squared Error (Test):", test_rmse)

# Plot the actual and predicted values, including the forecast
plt.plot(df.index[:train_size], y_train, lw=5,
         label='Actual (Train)', color="#FFCC80")
plt.plot(df.index[train_size:], y_test, lw=5, label='Actual (Test)', color = "#d3b8fd")
plt.plot(df.index[:train_size], train_predictions,
         lw=1, label='Predicted (Train)', color = "#3050BB")
plt.plot(df.index[train_size:], test_predictions,
         lw=1, label='Predicted (Test)', color = "#FF7080")
plt.xlabel('Date')
plt.ylabel(data_label)
plt.title('Regression Tree: Actual vs Predicted (Including Forecast)')
plt.legend()
plt.show()
