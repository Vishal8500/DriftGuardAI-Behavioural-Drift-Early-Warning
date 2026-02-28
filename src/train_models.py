import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

df = pd.read_csv(r"D:\Behavioural_Hackathon\data\engineered_behaviour_features.csv")

le = LabelEncoder()
df["burnout_encoded"] = le.fit_transform(df["burnout_label"])

X = df.drop(["student_id", "burnout_label", "dropout_label", "burnout_encoded"], axis=1)
y = df["burnout_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

burnout_model = RandomForestClassifier(n_estimators=200, max_depth=8)
burnout_model.fit(X_train, y_train)

print(classification_report(y_test, burnout_model.predict(X_test)))

dropout_model = LogisticRegression(max_iter=1000)
dropout_model.fit(X_train, df.loc[X_train.index, "dropout_label"])

with open("models/burnout_model.pkl", "wb") as f:
    pickle.dump(burnout_model, f)

with open("models/dropout_model.pkl", "wb") as f:
    pickle.dump(dropout_model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Models Saved")