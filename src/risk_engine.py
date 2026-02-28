import pickle
import pandas as pd
import numpy as np
import os

# -----------------------------
# Load Engineered Feature Data
# -----------------------------
df = pd.read_csv(r"D:\Behavioural_Hackathon\data\engineered_behaviour_features.csv")

# -----------------------------
# Load Trained Models
# -----------------------------
with open("models/burnout_model.pkl", "rb") as f:
    burnout_model = pickle.load(f)

with open("models/dropout_model.pkl", "rb") as f:
    dropout_model = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# -----------------------------
# Prepare Feature Matrix
# -----------------------------
X = df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

# -----------------------------
# Model Predictions
# -----------------------------
burnout_probs = burnout_model.predict_proba(X)
dropout_probs = dropout_model.predict_proba(X)[:, 1]

# -----------------------------
# Normalize Drift Components (0–1 scale)
# -----------------------------

# Clip percentage-based features safely
login_norm = df["login_drop_pct"].abs().clip(0, 1)
att_norm = df["attendance_drop_pct"].abs().clip(0, 1)

# Normalize delay increase relative to max value
delay_norm = df["delay_increase_pct"].abs() / df["delay_increase_pct"].abs().max()

# Combine drift signals
drift = (login_norm + att_norm + delay_norm) / 3

# -----------------------------
# Final Risk Score (0–100)
# -----------------------------
risk_score = (
    0.4 * burnout_probs.max(axis=1) +
    0.4 * dropout_probs +
    0.2 * drift
) * 100

# Ensure strictly 0–100 range
risk_score = np.clip(risk_score, 0, 100)

# -----------------------------
# Store Results
# -----------------------------
df["risk_score"] = np.round(risk_score, 2)
df["burnout_prediction"] = le.inverse_transform(burnout_model.predict(X))
df["dropout_probability"] = np.round(dropout_probs, 3)

# Create output folder if not exists
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/final_predictions.csv", index=False)

print("Risk Scoring Complete")
print("Risk Score Range:", df["risk_score"].min(), "to", df["risk_score"].max())