"""
feature_engineering.py
- Input : data/processed/cleaned_customer_churn.csv
- Output: data/processed/engineered_features.csv
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLEANED_PATH = os.path.join("data", "processed", "cleaned_customer_churn.csv")
ENGINEERED_PATH = os.path.join("data", "processed", "engineered_features.csv")

os.makedirs(os.path.dirname(ENGINEERED_PATH), exist_ok=True)

def main():
    df = pd.read_csv(CLEANED_PATH)
    print("Loaded cleaned data:", df.shape)

    # Ensure target exists
    if "Churn" not in df.columns:
        raise Exception("Target column 'Churn' not found in cleaned data.")

    # Convert Yes/No to 1/0 if necessary
    if df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")
    df = df.dropna(subset=["Churn"])

    # Detect numeric columns (exclude target)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Churn"]
    print("Numeric cols:", numeric_cols)

    # Fill missing numeric values with median
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())

    # Scale numeric columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Categorical columns
    cat_cols = [c for c in df.columns if df[c].dtype == object and c != "Churn"]
    print("Categorical cols:", cat_cols)

    # Simple imputation for categorical: fillna with 'Unknown'
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df.to_csv(ENGINEERED_PATH, index=False)
    print("Engineered features saved to:", ENGINEERED_PATH)
    print("Engineered shape:", df.shape)

if __name__ == "__main__":
    main()
