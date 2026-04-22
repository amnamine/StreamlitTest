# 🌾 Crop Yield Prediction — Streamlit App

## About the Project
This repository contains a Streamlit-based web application designed as an interactive inference demo for predicting crop yields. The model calculates the expected yield measured in hectograms per hectare (hg/ha) based on various environmental and agricultural factors. 

## Features
* **Interactive User Interface**: A clean, web-based dashboard built with Streamlit.
* **Real-Time Inference**: Instantly predict crop yields by adjusting input parameters such as temperature, rainfall, and pesticide usage.
* **Dynamic Form Inputs**: Automatically loads available countries (Areas) and crops (Items) directly from the project's dataset.

## Technology Stack
* **Framework**: Streamlit, Flask, Gunicorn
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (DecisionTreeRegressor, StandardScaler), Joblib

## Machine Learning Pipeline
The inference pipeline in this application strictly mirrors the preprocessing steps used during the model's training phase:
1. **Feature Engineering**: Calculates a custom ratio using the formula `rain_temp_ratio = rain / (temp + 0.1)`.
2. **Categorical Encoding**: Applies One-Hot Encoding to the `Area` and `Item` categories (utilizing `drop_first=True`).
3. **Scaling**: Transforms the feature matrix using a pre-fitted `StandardScaler`.
4. **Prediction**: Feeds the scaled data into a tuned `DecisionTreeRegressor` (optimized via GridSearchCV) to generate the final yield prediction.

## Prerequisites & Required Files
To run this application successfully, the following serialized model artifacts and datasets must be located in the project's root directory (alongside `streamlit_app.py`):
* `crop_yield_prediction_model.pkl` (The trained Decision Tree model)
* `crop_yield_scaler.pkl` (The fitted StandardScaler)
* `model_features.pkl` (A list of feature names to ensure the input matrix matches the training phase exactly)
* `yield_df.csv` (Optional but highly recommended; used to populate the country and crop dropdown menus)

## Installation & Setup

1. **Install Dependencies:**
   Ensure you have Python installed, then install the required packages using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application:**
   Launch the Streamlit server locally by executing the following command in your terminal:
   ```bash
   streamlit run streamlit_app.py
   ```
   The application will automatically open in your default web browser.
