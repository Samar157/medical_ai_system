import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# Load datasets (fix path using raw string r"")
df1 = pd.read_csv(r"C:\Users\abhis\Downloads\heart.csv")
# Combine
data = pd.concat([df1], ignore_index=True)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Select ONLY useful features
features = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "fbs",
    "thalach",
    "exang",
    "oldpeak"
]

# Keep only required columns
data = data[features + ["target"]]

# Remove missing target
data = data.dropna(subset=["target"])

# Fill missing values
data = data.fillna(data.median(numeric_only=True))

# Split X, y
X = data[features]
y = data["target"]

# Model pipeline
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# Cross validation
scores = cross_val_score(pipe, X, y, cv=5)
print("Accuracy:", scores.mean())

# Train full model
pipe.fit(X, y)

# Save model
joblib.dump(pipe, "heart_model.pkl")

print("Model saved successfully ✅")