import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "customer_churn.csv")
CLEANED_PATH = os.path.join("data", "processed", "cleaned_customer_churn.csv")

os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)

def main():
    df = pd.read_csv(RAW_PATH)
    print("Initial shape:", df.shape)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Print columns for visibility
    print("Columns:", list(df.columns))

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # If Churn column exists and is Yes/No ensure consistent capitalization
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.title()

    # Convert empty strings in numeric-like columns to NaN
    for col in df.columns:
        if df[col].dtype == object:
            # Attempt to detect numeric columns stored as text
            sample = df[col].dropna().head(10).astype(str)
            if sample.str.match(r"^\d+(\.\d+)?$").all():
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Basic missing-value handling: drop rows without target
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])

    # Save cleaned dataset
    df.to_csv(CLEANED_PATH, index=False)
    print("Cleaned data saved to:", CLEANED_PATH)
    print("Cleaned shape:", df.shape)

if __name__ == "__main__":
    main()
