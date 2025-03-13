import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import os

# Function to preprocess the data
def preprocess_data(data):
    # Drop non-numeric columns (if any)
    data = data.select_dtypes(include=['number'])

    # Simulate DON concentration using a predefined function (Approach 2)
    weights = np.random.rand(data.shape[1])  # Random weights for all spectral bands
    data["DON_concentration"] = data.dot(weights) + np.random.normal(0, 0.1, size=len(data))
    data["DON_concentration"] += np.random.normal(0, 0.1, size=len(data))

    # Handle outliers in the target variable (DON concentration)
    Q1 = data["DON_concentration"].quantile(0.25)
    Q3 = data["DON_concentration"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Apply log transformation to normalize the distribution
    data["DON_concentration_log"] = np.log1p(data["DON_concentration"])
    data["DON_concentration_log"] = np.where(data["DON_concentration_log"] < lower_bound, lower_bound, data["DON_concentration_log"])
    data["DON_concentration_log"] = np.where(data["DON_concentration_log"] > upper_bound, upper_bound, data["DON_concentration_log"])

    # Normalize the spectral data
    scaler = StandardScaler()
    spectral_data = data.iloc[:, :-2]  # Exclude target and log-transformed target
    scaled_spectral_data = scaler.fit_transform(spectral_data)

    return data, scaled_spectral_data, scaler

# Function to train the model and save PCA/model
def train_and_save_model(X_train, y_train):
    # Apply PCA
    pca = PCA(n_components=5)  # Reduce to 5 features
    pca.fit(X_train)

    # Save PCA to a file
    with open('pca.pkl', 'wb') as file:
        pickle.dump(pca, file)

    # Train a model (e.g., Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(pca.transform(X_train), y_train)

    # Save the model to a file
    with open('rf_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    st.success("Model and PCA saved successfully!")

# Function to load the trained model and PCA
@st.cache_resource
def load_model_and_preprocessing():
    # Load PCA
    with open('pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    # Load the model
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return pca, model

# Streamlit app
def main():
    st.title("DON Concentration Prediction App")
    st.write("Upload a CSV file containing spectral data to predict DON concentration.")

    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the data
        data = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data")
        st.write(data.head())

        # Preprocess the data
        data, scaled_spectral_data, scaler = preprocess_data(data)
        st.write("### Preprocessed Data")
        st.write(pd.DataFrame(scaled_spectral_data).head())

        # Split dataset into training and testing
        X = scaled_spectral_data
        y = data['DON_concentration_log']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model (if not already trained)
        if not (os.path.exists('pca.pkl') and os.path.exists('rf_model.pkl')):
            st.warning("Model and PCA not found. Training the model...")
            train_and_save_model(X_train, y_train)

        # Load the trained model and PCA
        pca, model = load_model_and_preprocessing()

        # Make predictions
        y_pred = model.predict(pca.transform(X_test))

        # Evaluate model performance
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write("### Model Evaluation")
        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R² Score: {r2}")

        # Scatter plot of actual vs predicted values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.set_xlabel("Actual DON Concentration (log)")
        ax.set_ylabel("Predicted DON Concentration (log)")
        ax.set_title("Actual vs Predicted DON Concentration")
        st.pyplot(fig)

        # Option to download predictions
        predictions_df = pd.DataFrame(y_pred, columns=["Predicted DON Concentration (log)"])
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

# Run the app
if __name__ == "__main__":
    main()
