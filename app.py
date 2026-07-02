import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import numpy as np

# Load Dataset
df = pd.read_csv("car data.csv")

# # First 5 Rows
print(df.head())

# # Shape
print("\nShape:")
print(df.shape)

# # Columns
print("\nColumns:")
print(df.columns)

# # Information
print("\nInformation:")
print(df.info())

# # Description
print("\nDescription:")
print(df.describe())

# # Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nCar Name Count")

print(df["Car_Name"].value_counts())

print("\nTotal Unique Cars")

print(df["Car_Name"].nunique())

print("\nFuel Type")

print(df["Fuel_Type"].value_counts())

print("\nSelling Type")

print(df["Selling_type"].value_counts())

print("\nTransmission")

print(df["Transmission"].value_counts())

print("\nOwner")

print(df["Owner"].value_counts())

print("\nUnique Fuel Types")

print(df["Fuel_Type"].unique())

print("\nUnique Transmission")

print(df["Transmission"].unique())

# Histogram

df.hist(figsize=(12,10))

plt.suptitle("Car Dataset Histogram")

plt.show()

# Fuel Type Bar Chart

df["Fuel_Type"].value_counts().plot(kind="bar")

plt.title("Fuel Type Distribution")

plt.xlabel("Fuel Type")

plt.ylabel("Count")

plt.show()

# Transmission Bar Chart

df["Transmission"].value_counts().plot(kind="bar")

plt.title("Transmission Distribution")

plt.xlabel("Transmission")

plt.ylabel("Count")

plt.show()

# Selling Price vs Present Price

plt.figure(figsize=(8,6))

plt.scatter(

    df["Present_Price"],

    df["Selling_Price"]

)

plt.xlabel("Present Price")

plt.ylabel("Selling Price")

plt.title("Present Price vs Selling Price")

plt.show()

# Create Copy

car_df = df.copy()

# # Remove Car Name

car_df = car_df.drop("Car_Name", axis=1)

# # Fuel Type Encoding

car_df["Fuel_Type"] = car_df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

# # Selling Type Encoding

car_df["Selling_type"] = car_df["Selling_type"].map({
    "Dealer": 0,
    "Individual": 1
})

# # Transmission Encoding

car_df["Transmission"] = car_df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# # Final Dataset

print("\nEncoded Dataset")

print(car_df.head())

# Features (X)

X = car_df.drop("Selling_Price", axis=1)

# Target (y)

y = car_df["Selling_Price"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

print("\nFeatures (X)")
print(X.head())

print("\nTarget (y)")
print(y.head())

print("\nTraining Data:", X_train.shape)

print("Testing Data:", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredicted Prices")

print(predictions[:10])

# R2 Score

r2 = r2_score(y_test, predictions)

print("\nR2 Score:", r2)

# MAE

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

# MSE

mse = mean_squared_error(y_test, predictions)

print("\nMean Squared Error:", mse)

# RMSE

rmse = np.sqrt(mse)

print("\nRoot Mean Squared Error:", rmse)

# Actual vs Predicted (Line Graph)

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title("Actual vs Predicted Car Price")

plt.xlabel("Cars")

plt.ylabel("Price")

plt.legend()

plt.show()

# Scatter Plot

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.show()

# Prediction Error

errors = y_test - predictions

plt.figure(figsize=(8,6))

plt.hist(errors, bins=20)

plt.title("Prediction Error Distribution")

plt.xlabel("Error")

plt.ylabel("Frequency")

plt.show()