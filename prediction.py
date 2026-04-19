# Electricity Theft Detection Inference Script (Frontend-Ready)
# -----------------------------------------------------------
# PURPOSE:
# This script is designed to be used after model training.
# It takes a CSV file as input and outputs all locations/consumers
# where electricity theft is possible.
#
# It is structured like an API/backend so a frontend (Streamlit, Flask,
# web form, or desktop UI) can easily call the main function.
#
# INPUT:
#   CSV file containing consumption data
#
# OUTPUT:
#   1. Printed list of suspicious consumers
#   2. Saved CSV file with theft predictions
#
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
from scipy.stats import zscore


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "theft_model.pkl"
SCALER_FILE = "scaler.pkl"
OUTPUT_FILE = "possible_theft_locations.csv"


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

def load_model_and_scaler():
    """
    Loads the trained ML model and scaler.
    """

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    print("Model and scaler loaded successfully")

    return model, scaler


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(data):
    """
    Cleans and prepares input data.
    """

    # Convert values to numeric
    data = data.apply(pd.to_numeric, errors="coerce")

    # Handle missing values
    data = data.ffill()
    data = data.bfill()

    # Select numeric columns
    numeric_data = data.select_dtypes(include=["number"])

    return data, numeric_data


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data, numeric_data):
    """
    Creates behavioral features used by the model.
    """

    data["avg_consumption"] = numeric_data.mean(axis=1)

    data["std_consumption"] = numeric_data.std(axis=1)

    data["max_consumption"] = numeric_data.max(axis=1)

    data["min_consumption"] = numeric_data.min(axis=1)

    data["consumption_range"] = (
        data["max_consumption"]
        - data["min_consumption"]
    )

    return data


# ============================================================
# ANOMALY / THEFT DETECTION
# ============================================================

def predict_theft(data, model, scaler):
    """
    Predicts theft probability and risk score.
    """

    features = [
        "avg_consumption",
        "std_consumption",
        "max_consumption",
        "min_consumption",
        "consumption_range"
    ]

    X = data[features]

    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)

    data["theft_probability"] = probabilities[:, 1]

    data["risk_score"] = (
        data["theft_probability"] * 100
    ).round(2)

    # =====================================
# ESTIMATED REVENUE LOSS
# =====================================

    cost_per_unit = 7

    data["estimated_loss"] = (
        data["avg_consumption"]
        * data["theft_probability"]
        * cost_per_unit
    ).round(2)


    # =====================================
# EXPLAINABLE AI — REASONS
# =====================================

    data["reason"] = ""

    data.loc[
        data["consumption_range"] > 50,
        "reason"
    ] += "Sudden consumption change; "

    data.loc[
        data["std_consumption"] > 2 * data["avg_consumption"],
        "reason"
    ] += "High variability; "

    data.loc[
        data["min_consumption"] == 0,
        "reason"
    ] += "Frequent zero usage; "



    return data


# ============================================================
# PRIORITY CLASSIFICATION
# ============================================================

def assign_priority(data):
    """
    Assigns inspection priority level.
    """

    conditions = [
        data["risk_score"] >= 80,
        data["risk_score"].between(50, 79)
    ]

    levels = [
        "HIGH",
        "MEDIUM"
    ]

    data["inspection_priority"] = np.select(
        conditions,
        levels,
        default="LOW"
    )

    return data


# ============================================================
# FILTER SUSPICIOUS LOCATIONS
# ============================================================

def get_possible_theft_locations(data):
    """
    Returns only suspicious consumers.
    """

    suspicious = data[
        data["inspection_priority"] != "LOW"
    ]

    return suspicious


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(suspicious_data):
    """
    Saves detected theft locations to CSV.
    """

    suspicious_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Results saved to:", OUTPUT_FILE)


# ============================================================
# MAIN PIPELINE FUNCTION (Frontend will call this)
# ============================================================

def predict_from_csv(file_path):
    """
    Main function to run full detection pipeline.

    This is the function your frontend will call.
    """

    print("Reading input file...")

    data = pd.read_csv(file_path)

    model, scaler = load_model_and_scaler()

    data, numeric_data = preprocess_data(data)

    data = create_features(data, numeric_data)

    data = predict_theft(data, model, scaler)

    data = assign_priority(data)

    suspicious = get_possible_theft_locations(data)

    save_results(suspicious)

    print("\nPossible theft locations:")

    print(suspicious.head(10))

    return data, suspicious


# ============================================================
# COMMAND LINE EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = input(
        "Enter CSV file path: "
    )

    predict_from_csv(file_path)
