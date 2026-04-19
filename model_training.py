import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =============================
# STEP 1 — LOAD DATA
# =============================

file_path = "data 1.csv"

data = pd.read_csv(file_path)

print("Dataset shape:", data.shape)
print(data.head())


# =============================
# STEP 2 — HANDLE MISSING VALUES
# =============================

data = data.ffill()
data = data.bfill()

print("Missing values handled")


# =============================
# STEP 3 — FEATURE ENGINEERING
# =============================

# =============================
# STEP — SELECT NUMERIC COLUMNS
# =============================

# Convert everything to numeric where possible
data = data.apply(pd.to_numeric, errors="coerce")

# Handle missing values again after conversion
data = data.ffill()
data = data.bfill()

# Select only numeric columns
numeric_data = data.select_dtypes(include=["number"])

print("Numeric columns selected:", numeric_data.shape)

# =============================
# FEATURE ENGINEERING
# =============================

data["avg_consumption"] = numeric_data.mean(axis=1)

data["std_consumption"] = numeric_data.std(axis=1)

data["max_consumption"] = numeric_data.max(axis=1)

data["min_consumption"] = numeric_data.min(axis=1)

data["consumption_range"] = (
    data["max_consumption"]
    - data["min_consumption"]
)

print("Features created successfully")


# =============================
# STEP 4 — OUTLIER DETECTION
# =============================
date_columns = data.select_dtypes(include="number").columns
z_scores = data[date_columns].apply(
    lambda x: zscore(x, nan_policy="omit")
)

z_scores = pd.DataFrame(
    z_scores,
    columns=date_columns
)

outliers = data[
    (np.abs(z_scores) > 3).any(axis=1)
]

print("Number of outliers:", len(outliers))


# =============================
# STEP 5 — CREATE LABEL
# =============================

data["theft_flag"] = 0

data.loc[outliers.index, "theft_flag"] = 1


# =============================
# STEP 6 — MODEL FEATURES
# =============================

features = [
    "avg_consumption",
    "std_consumption",
    "max_consumption",
    "min_consumption",
    "consumption_range"
]

X = data[features]

y = data["theft_flag"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =============================
# STEP 7 — TRAIN MODEL
# =============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_scaled, y)

print("Model trained")


# =============================
# STEP 8 — RISK SCORE
# =============================

probabilities = model.predict_proba(X_scaled)

data["theft_probability"] = probabilities[:, 1]

data["risk_score"] = (
    data["theft_probability"] * 100
).round(2)


# =============================
# STEP 9 — PRIORITY LEVEL
# =============================

conditions = [
    data["risk_score"] >= 80,
    data["risk_score"] >= 50,
    data["risk_score"] < 50
]

levels = [
    "HIGH",
    "MEDIUM",
    "LOW"
]

data["inspection_priority"] = np.select(
    conditions,
    levels,
    default = "LOW"
)


# =============================
# STEP 10 — SAVE RESULTS
# =============================

data = data.sort_values(
    by="risk_score",
    ascending=False
)

data.to_csv(
    "electricity_theft_results.csv",
    index=False
)

print("Results saved successfully")

import joblib

joblib.dump(model, "theft_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved")