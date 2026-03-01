import shap
import pickle
import pandas as pd
import os
import numpy as np

print("Loading data...")

df = pd.read_csv("data/engineered_behaviour_features.csv")
X = df.drop(["student_id", "burnout_label", "dropout_label"], axis=1)

with open("models/burnout_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Computing SHAP values (TreeExplainer)...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ---------------------------------------------------
# Handle Multi-Class Shape Safely
# ---------------------------------------------------

# If shap_values is list (old style)
if isinstance(shap_values, list):
    high_shap = shap_values[2]   # High class index
else:
    # New style: (samples, features, classes)
    high_shap = shap_values[:, :, 2]

print("SHAP shape:", high_shap.shape)

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------
# Save student-level SHAP values
# ---------------------------------------------------
shap_df = pd.DataFrame(high_shap, columns=X.columns)
shap_df["student_id"] = df["student_id"]

shap_df.to_csv("outputs/shap_student_level_values.csv", index=False)

# ---------------------------------------------------
# Save global importance
# ---------------------------------------------------
mean_importance = np.abs(high_shap).mean(axis=0)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": mean_importance
}).sort_values("mean_abs_shap", ascending=False)

importance_df.to_csv("outputs/shap_global_importance.csv", index=False)

print("SHAP precomputation complete.")