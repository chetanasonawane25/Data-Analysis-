DON Concentration Prediction Using Spectral Data


This repository contains a Python script for predicting Dissolved Organic Nitrogen (DON) concentration using spectral data. The script includes data preprocessing, dimensionality reduction, model training, and evaluation using machine learning techniques like Random Forest and XGBoost.

Table of Contents
Project Overview
Prerequisites
Installation Instructions
Running the Code
Repository Structure
Dependencies
Methodology
Results
Contributing


Project Overview
This project focuses on predicting DON concentration from spectral data using machine learning models. The script performs the following tasks:

Loads and preprocesses spectral data.
Simulates DON concentration using a weighted sum of spectral bands.
Applies data preprocessing techniques like outlier handling, normalization, and log transformation.
Reduces dimensionality using PCA and t-SNE.
Trains and evaluates Random Forest and XGBoost models.
Compares model performance and selects the best model.
Prerequisites
Before running the code, ensure you have the following:

Python 3.6 or higher installed.
A dataset in CSV format containing spectral data (e.g., TASK-ML-INTERN (1).csv).
A working internet connection to install dependencies.
Installation Instructions
Follow these steps to set up the environment and install the required dependencies:

Clone the Repository (if applicable): git clone https://github.com/chetanasonawane25/Imago_AI_Tasks.git
Create a Virtual Environment (optional but recommended): python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate
Install Dependencies: The script requires several Python libraries. Install them using the following command:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Verify Installation: Ensure all libraries are installed by running:
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, xgboost; print('All libraries installed successfully!')"
Running the Code

Prepare the Dataset:
Ensure your dataset (TASK-ML-INTERN (1).csv) is placed in the correct directory.
Update the file path in the script if needed: data = pd.read_csv(r"C:\Users\HP\Downloads\TASK-ML-INTERN (1).csv") Replace the path with the actual location of your dataset.

Run the Script:
Open a terminal in the project directory.
Execute the script: python don_prediction.py
The script will load the data, preprocess it, train models, and display results with visualizations.

Expected Output:
Data summary statistics, missing value reports, and duplicate checks.
Visualizations like boxplots, spectral reflectance plots, correlation heatmaps, PCA/t-SNE scatter plots, and actual vs. predicted scatter plots.
Model evaluation metrics (MAE, RMSE, R² Score) for Random Forest and XGBoost.
A final table comparing model performance and the best model recommendation.
Repository Structure

Below is the structure of the repository:

don-prediction/
│
├── don_prediction.py           # Main Python script for DON concentration prediction
├── TASK-ML-INTERN (1).csv      # Sample dataset (not included, user-provided)
├── README.md                   # This README file
└── venv/                       # Virtual environment (optional, created by user)

don_prediction.py: The core script that performs data preprocessing, visualization, model training, and evaluation.
TASK-ML-INTERN (1).csv: The dataset file containing spectral data (replace with your dataset).
README.md: Documentation for setting up and running the project.
Dependencies


The following Python libraries are required:

pandas: For data manipulation and analysis.
numpy: For numerical computations.
matplotlib: For plotting and visualization.
seaborn: For enhanced data visualization.
scikit-learn: For preprocessing, dimensionality reduction, and machine learning models.
xgboost: For the XGBoost regressor model.
Install them using:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

Methodology
The script follows these steps:

Data Exploration and Preprocessing:
Loads the dataset and checks for missing values, duplicates, and data types.
Simulates DON concentration as a weighted sum of spectral bands with added noise.
Handles outliers using IQR and applies log transformation for normalization.
Normalizes spectral data using StandardScaler.

Visualization:
Generates boxplots for DON concentration before and after preprocessing.
Plots spectral reflectance for a subset of samples.
Creates a correlation heatmap for spectral bands.

Dimensionality Reduction:
Applies PCA and t-SNE to reduce the dimensionality of spectral data.
Visualizes the results with scatter plots colored by DON concentration.

Model Training and Evaluation:
Splits the data into training and testing sets.
Trains Random Forest and XGBoost models on the log-transformed DON concentration.
Evaluates models using MAE, RMSE, and R² Score.
Visualizes actual vs. predicted values for both models.

Model Comparison:
Compares Random Forest and XGBoost performance using a table.
Selects the best model based on overall metrics.

Results
The script outputs detailed evaluation metrics for both Random Forest and XGBoost models.
Visualizations help understand the data distribution, spectral patterns, and model performance.
A final table summarizes the metrics and identifies the best model (Random Forest, XGBoost, or a potential ensemble).
Contributing

Contributions are welcome! To contribute:

Fork the repository.
Create a new branch for your feature or bugfix.
Submit a pull request with a detailed description of your changes.
