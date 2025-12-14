"""
evaluate.py
Generates evaluation metrics and saves a confusion matrix plot.
"""
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

ENGINEERED_PATH = os.path.join("data", "processed", "engineered_features.csv")
MODEL_PATH = os.path.join("models", "best_churn_model.pkl")
VIS_DIR = os.path.join("visuals")
os.makedirs(VIS_DIR, exist_ok=True)

def main():
    data = pd.read_csv(ENGINEERED_PATH)
    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    features = model_bundle["feature_columns"]

    X = data[features]
    y = data[[c for c in data.columns if c.lower()=="churn"][0]]

    # Simple evaluation on full dataset; for final evaluation use test set saved earlier
    y_pred = model.predict(X)
    print("Classification Report:\n", classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix (on full dataset)")
    plt.savefig(os.path.join(VIS_DIR, "confusion_matrix.png"))
    print("Saved confusion matrix to visuals/confusion_matrix.png")

if __name__ == "__main__":
    main()
