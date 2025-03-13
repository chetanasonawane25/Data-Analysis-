#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1. Data Exploration and Preprocessing

#Impot necessary libearies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

#1.1 Load the dataset
data = pd.read_csv(r"C:\Users\HP\Downloads\TASK-ML-INTERN (1).csv")
print("Data loaded successfully.")
print(data.head())

#1.2 Check for Data Structure
print("Shape of the dataset:", data.shape)
print("Data types:\n", data.dtypes)

#1.3 Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

if missing_values.any():
    print("There are missing values in the dataset.")
else:
    print("No missing values found in the dataset.")

#1.4 Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")
if duplicates > 0:
    data.drop_duplicates(inplace=True)

#1.5 Summary Statistics
print("\nDataset Summary:")
print(data.describe())

#1.6 Drop non-numeric columns (if any)
data = data.select_dtypes(include=['number'])

# Simulate DON concentration using a predefined function (Approach 2)
# Assume DON concentration is a polynomial function of ALL spectral bands
# Generate random weights for all spectral bands
weights = np.random.rand(data.shape[1])  # Random weights for all spectral bands

# Simulate DON concentration as a weighted sum of all spectral bands
data["DON_concentration"] = data.dot(weights) + np.random.normal(0, 0.1, size=len(data))

# Add some noise to make it realistic
data["DON_concentration"] += np.random.normal(0, 0.1, size=len(data))

# Check the generated DON concentration
print("\nSimulated DON Concentration:")
print(data["DON_concentration"].head())

#1.7 Boxplot for target variable (DON concentration)
plt.figure(figsize=(12, 5))
sns.boxplot(data=data[["DON_concentration"]])
plt.title("Boxplot for DON Concentration")
plt.xticks(rotation=90)
plt.show()

#1.8 Handle outliers in the target variable (DON concentration)
Q1 = data["DON_concentration"].quantile(0.25)
Q3 = data["DON_concentration"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data["DON_concentration"] < lower_bound) | (data["DON_concentration"] > upper_bound)]
print(f"Number of Outliers Before Treatment: {outliers.shape[0]}")

#1.9 Apply log transformation to normalize the distribution
data["DON_concentration_log"] = np.log1p(data["DON_concentration"])

#1.10 Replace outliers in log-transformed data
data["DON_concentration_log"] = np.where(data["DON_concentration_log"] < lower_bound, lower_bound, data["DON_concentration_log"])
data["DON_concentration_log"] = np.where(data["DON_concentration_log"] > upper_bound, upper_bound, data["DON_concentration_log"])

#1.11 Boxplot after log transformation and outlier handling
sns.boxplot(data=data[["DON_concentration_log"]])
plt.title("Boxplot After Log + Outlier Treatment")
plt.show()

#1.12 Normalize the spectral data
scaler = StandardScaler()
spectral_data = data.iloc[:, :-2]  # Exclude target and log-transformed target
scaled_spectral_data = scaler.fit_transform(spectral_data)

#1.13 Visualize spectral bands
plt.figure(figsize=(12, 6))
x_values = spectral_data.columns
selected_xticks = x_values[::10]  # Reduce x-axis labels for better readability

for i in range(5):
    plt.plot(x_values, scaled_spectral_data[i], label=f"Sample {i+1}", linewidth=2)

plt.title("Spectral Reflectance Visualization")
plt.xlabel("Wavelength Bands")
plt.ylabel("Normalized Reflectance")
plt.xticks(selected_xticks, rotation=45)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

#1.14 Heatmap for correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(scaled_spectral_data).corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

#2. Dimensionality Reduction

#2.1 Dimensionality Reduction using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_spectral_data)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

#2.2 Visualizing PCA results
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['DON_concentration_log'], cmap='viridis')
plt.colorbar(label='DON Concentration (log)')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Dimensionality Reduction")
plt.show()

#2.3 Dimensionality Reduction using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(scaled_spectral_data)

#2.4 Visualizing t-SNE results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=data['DON_concentration_log'], cmap='plasma')
plt.colorbar(label='DON Concentration (log)')
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE - Dimensionality Reduction")
plt.show()

#3. Model Training

#3.1 Split dataset into training and testing
X = scaled_spectral_data
y = data['DON_concentration_log']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3.2 Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#3.3 Make predictions
y_pred_rf = rf_model.predict(X_test)

#4. Model Evaluation

#4.1 Evaluate model performance
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
print(f"\nRandom Forest Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"Root Mean Squared Error (RMSE): {rmse_rf}")
print(f"R² Score: {r2_rf}")

#4.2 Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual DON Concentration (log)")
plt.ylabel("Predicted DON Concentration (log)")
plt.title("Actual vs Predicted DON Concentration (Random Forest)")
plt.show()

#4.3 Train XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

#4.4 Evaluate XGBoost model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)
print(f"\nXGBoost Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae_xgb}")
print(f"Root Mean Squared Error (RMSE): {rmse_xgb}")
print(f"R² Score: {r2_xgb}")

#4.5 Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)
plt.xlabel("Actual DON Concentration (log)")
plt.ylabel("Predicted DON Concentration (log)")
plt.title("Actual vs Predicted DON Concentration (XGBoost)")
plt.show()

#4.6 Compare Random Forest and XGBoost
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7, label='Random Forest', color='blue')
plt.scatter(y_test, y_pred_xgb, alpha=0.7, label='XGBoost', color='red')
plt.xlabel("Actual DON Concentration (log)")
plt.ylabel("Predicted DON Concentration (log)")
plt.title("Actual vs Predicted DON Concentration (Random Forest vs XGBoost)")
plt.legend()
plt.show()

#4.6 Define Model Evaluation Metrics
results = {
    "Metric": ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score"],
    "Random Forest": [mae_rf, rmse_rf, r2_rf],
    "XGBoost": [mae_xgb, rmse_xgb, r2_xgb],
    "Best Model": [
        "XGBoost" if mae_xgb < mae_rf else "Random Forest",
        "XGBoost" if rmse_xgb < rmse_rf else "Random Forest",
        "Random Forest" if r2_rf > r2_xgb else "XGBoost"
    ]
}

# Convert results to DataFrame
df_results = pd.DataFrame(results)

#4.7 Display Results
print("\n================= Final Model Interpretation =================")
print(df_results.to_string(index=False))
print("==============================================================")

#4.8 Best model decision
if results["Best Model"].count("Random Forest") > results["Best Model"].count("XGBoost"):
    best_model = "Random Forest"
elif results["Best Model"].count("XGBoost") > results["Best Model"].count("Random Forest"):
    best_model = "XGBoost"
else:
    best_model = "Stacking Model (Ensemble of RF & XGB)"

print(f"\n **Best Model Based on Overall Performance: {best_model}** ")


# In[ ]:




