# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('credit_data.csv')

# Check the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())  # Check for missing data

# If there are missing values, handle them (e.g., drop rows with missing values)
df = df.dropna()  # You can also choose to fill missing values with df.fillna() if preferred

# Defining features (X) and target (y)
X = df[['Age', 'Income', 'LoanAmount']]  # Features (independent variables)
y = df['CreditScore']  # Target (dependent variable)

# Scaling the features (optional but can help for better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nMean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Create a DataFrame to compare Actual vs Predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Display the first few rows of Predicted vs Actual values
print("\nPredicted vs Actual values:")
print(comparison_df.head())

# Plotting the Predicted vs Actual values (using matplotlib)
plt.figure(figsize=(10, 6))
plt.scatter(comparison_df['Actual'], comparison_df['Predicted'], color='blue')
plt.plot([comparison_df['Actual'].min(), comparison_df['Actual'].max()],
         [comparison_df['Actual'].min(), comparison_df['Actual'].max()],
         color='red', linestyle='--')  # Line of perfect prediction
plt.title('Predicted vs Actual Credit Score')
plt.xlabel('Actual Credit Score')
plt.ylabel('Predicted Credit Score')
plt.grid(True)
plt.show()

# Plotting the residuals (Predicted - Actual)
residuals = y_pred - y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')  # Zero line for residuals
plt.title('Residual Plot')
plt.xlabel('Predicted Credit Score')
plt.ylabel('Residuals (Predicted - Actual)')
plt.grid(True)
plt.show()
