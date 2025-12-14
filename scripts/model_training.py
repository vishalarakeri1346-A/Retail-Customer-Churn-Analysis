"""
model_training.py
- Input : data/processed/engineered_features.csv
- Output: models/best_churn_model.pkl
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import joblib

ENGINEERED_PATH = os.path.join("data", "processed", "engineered_features.csv")
MODEL_DIR = os.path.join("models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")

def main():
    df = pd.read_csv(ENGINEERED_PATH)
    print("Loaded engineered data:", df.shape)

    # Detect target
    target_candidates = [c for c in df.columns if c.lower() == "churn"]
    if not target_candidates:
        raise Exception("Target column 'Churn' not found.")
    target = target_candidates[0]

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # train/test split (stratify if possible)
    stratify_arg = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

    print("Train/test sizes:", X_train.shape, X_test.shape)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0))
        }
        trained_models[name] = model
        print(name, results[name])

    # Select best model by F1
    best = max(results.keys(), key=lambda k: results[k]["f1"])
    print("Best model:", best, "metrics:", results[best])

    # Save best model + metadata
    joblib.dump({
        "model": trained_models[best],
        "model_name": best,
        "metrics": results[best],
        "feature_columns": list(X.columns)
    }, MODEL_PATH)
    print("Saved best model to:", MODEL_PATH)

if __name__ == "__main__":
    main()
