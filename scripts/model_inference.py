"""
model_inference.py
Usage:
    python scripts/model_inference.py --example
    or import the `predict` function into your app.
"""
import os
import joblib
import pandas as pd
import argparse

MODEL_PATH = os.path.join("models", "best_churn_model.pkl")

def predict(df):
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["feature_columns"]
    X = df[features]
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    res = df.copy()
    res["churn_pred"] = preds
    if probs is not None:
        res["churn_prob"] = probs
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", action="store_true", help="Run an example prediction with the first 5 rows")
    args = parser.parse_args()

    if args.example:
        # Use engineered dataset for an example (first rows)
        engineered = pd.read_csv(os.path.join("data", "processed", "engineered_features.csv"))
        example_df = engineered.head(5)
        print("Running example prediction on first 5 rows")
        print(predict(example_df))

if __name__ == "__main__":
    main()
