DON Concentration Prediction Using Spectral Data 📊


A Python-based machine learning project to predict Dissolved Organic Nitrogen (DON) concentration from spectral data, featuring advanced preprocessing, visualization, and model evaluation.

🚀 Simulated DON Concentration

Generates DON concentration using a weighted sum of spectral bands with added noise.


📈 Data Preprocessing & Visualization

Includes outlier handling, log transformation, normalization, and detailed visualizations like boxplots and heatmaps.


🛠️ Dimensionality Reduction

Applies PCA and t-SNE to simplify spectral data while preserving key patterns.


🤖 Machine Learning Models

Trains and evaluates Random Forest and XGBoost models for accurate predictions.


📊 Model Comparison

Compares model performance with metrics like MAE, RMSE, and R² Score.
This project provides a comprehensive pipeline for predicting DON concentration using spectral data, leveraging machine learning techniques to deliver actionable insights for environmental analysis.


Setup
Get started with the DON Prediction project! Just follow these steps to set up and run the code:

Clone the repository (if applicable):
git clone https://github.com/chetanasonawane25/Imago_AI_Tasks.git
cd Imago_AI_Tasks


Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost


Update the dataset path in the script (don_prediction.py):
Replace the path in the following line with the location of your dataset:

data = pd.read_csv(r"C:\Users\HP\Downloads\TASK-ML-INTERN (1).csv")


Run the script:

python don_prediction.py



DON Concentration Prediction Comprehensive Report :-

[DON_Concentration_Prediction_Comprehensive_Report.pdf](https://github.com/user-attachments/files/19229769/DON_Concentration_Prediction_Comprehensive_Report.pdf)
