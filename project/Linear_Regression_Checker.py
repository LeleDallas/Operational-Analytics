import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

stream_options = {
    '1': 'Energy',
    '2': 'Power',
    '0': 'Exit',
}

features_options = {
    '1': 'AC Efficiency (LHV)',
    '2': 'Heat Rate (HHV)',
    '3': 'Electricity Out',
    '4': 'Total Output Factor',
    '0': 'Exit',
}

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
        print("You selected", data_label,
              "Stream data for Kaiser Permanente - San Leandro")
        break
    else:
        print("Invalid choice. Please select a valid number.")


# Load the Excel file into a DataFrame, skipping the first 10 rows (header and image)
data = pd.read_excel('San Leandro ' + data_label + '.xlsx', skiprows=10)

# Specify the column names you want to access (starting from the second column)
features = ['AC Efficiency (LHV)', 'Heat Rate (HHV)',
            'Electricity Out', 'Total Output Factor']


data_label = None

while data_label is None:
    print("Select which data stream to check:")
    for number, label in features_options.items():
        print(f"{number}: {label}")

    data_number = input("Enter the corresponding number: ")
    if data_number == '0':
        exit(0)

    data_label = features_options.get(data_number)

    if data_label:
        print("You selected", data_label,
              "Stream data for Kaiser Permanente - San Leandro")
        break
    else:
        print("Invalid choice. Please select a valid number.")


features.remove(data_label)
target = data_label

# Create a new DataFrame with the desired columns
df = data.iloc[:, 1:].drop(index=0)

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the columns to float
df[features] = df[features].astype('float')

# Split the data into training, validation, and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    df[features],
    df[target],
    test_size=0.1,  # 10% for the test set
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1,  # 10% for the validation set (10% of 90%)
)

# Create and train the predictive model (e.g., Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
predictions_val = model.predict(X_val)
predictions_test = model.predict(X_test)

# Evaluate the model's performance on the validation set (e.g., calculate R^2 score)
r2_score_val = model.score(X_val, y_val)
print('R^2 Score (Validation Set):', r2_score_val)
# Evaluate the model's performance on the test set (e.g., calculate R^2 score)
r2_score_test = model.score(X_test, y_test)
print('R^2 Score (Test Set):', r2_score_test)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Plot predicted vs. actual values for the validation set
ax1.scatter(X_val.index, y_val, color='blue', label='Actual (Validation)')
ax1.scatter(X_val.index, predictions_val, color='red',
            label='Predicted (Validation)')
ax1.set_ylabel('Total Output Factor (Validation)')
ax1.set_title('Predicted vs. Actual Values')

# Plot predicted vs. actual values for the test set
ax2.scatter(X_test.index, y_test, color='blue', label='Actual (Test)')
ax2.scatter(X_test.index, predictions_test,
            color='red', label='Predicted (Test)')
ax2.set_xlabel('Index')
ax2.set_ylabel('Total Output Factor (Test)')

plt.subplots_adjust(hspace=0.4)
plt.show()
