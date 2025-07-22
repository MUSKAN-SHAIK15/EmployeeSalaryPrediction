import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("adult 3.csv")

# Replace '?' with NaN and drop missing rows
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# ✅ Select only the relevant features + target
selected_features = [
    'age',
    'workclass',
    'education',
    'occupation',
    'gender',
    'hours-per-week',
    'income'  # this is the target
]
df = df[selected_features]

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'income':  # we’ll encode target separately
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["income"] = target_encoder.fit_transform(df["income"])  # <=50K = 0, >50K = 1

# Split into features and label
X = df.drop("income", axis=1)
y = df["income"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "salary_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")  # in case you want to decode prediction

print("✅ Model trained with selected features and saved.")
