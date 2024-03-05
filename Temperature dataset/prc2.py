import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/home/rushabh/Documents/ML practicals/Temperature dataset/temperatures.csv')

# Display the first few rows to confirm it's loaded correctly
print(df.head())

# Define independent variable
X = df[['YEAR']].values

# Month names
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Store metrics
metrics = []

for month in months:
    # Define dependent variable
    y = df[month].values
    
    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict temperatures
    predictions = model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r_square = r2_score(y, predictions)
    
    metrics.append((month, mse, mae, r_square))
    
    # Plotting
    plt.scatter(X, y, color='gray')
    plt.plot(X, predictions, color='red', linewidth=2)
    plt.title(f'Temperature Trend for {month}')
    plt.xlabel('Year')
    plt.ylabel('Temperature')
    plt.show()
    
# Convert metrics into a DataFrame for better readability
metrics_df = pd.DataFrame(metrics, columns=['Month', 'MSE', 'MAE', 'R^2'])
print(metrics_df)



