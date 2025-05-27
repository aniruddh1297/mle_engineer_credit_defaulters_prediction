import argparse
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, fbeta_score
)
import pandas as pd

def load_data(input_data, model_path):
    loaded_obj = joblib.load(os.path.join(input_data, "test.pkl"))

    print("📦 Loaded test.pkl type:", type(loaded_obj))

    if isinstance(loaded_obj, tuple):
        X_test, y_test = loaded_obj
    else:
        raise ValueError("❌ test.pkl must contain a tuple (X_test, y_test). Got: {}".format(type(loaded_obj)))

    model = joblib.load(os.path.join(model_path, "best_model.pkl"))

    print("🧪 X_test shape:", X_test.shape)
    print("🧪 y_test shape:", y_test.shape)
    print("🧪 Model type:", type(model))

    return X_test, y_test, model

def optimize_threshold(y_true, probas, beta=1.0):
    best_thresh = 0.5
    best_fscore = 0.0
    for t in np.arange(0.2, 0.7, 0.01):
        preds = (probas >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=beta)
        if score > best_fscore:
            best_fscore = score
            best_thresh = t
    print(f"🎯 Best Threshold for F{beta}-Score: {best_thresh:.2f} | F{beta}: {best_fscore:.4f}")
    return best_thresh

def evaluate_model(model, X_test, y_test, threshold):
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)
    return preds, probas, cm, y_test

def plot_metrics(cm, probas, y_test, output_path):
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "confusion_matrix.png"))
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "roc_curve.png"))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pr_curve.png"))
    plt.close()

def main(args):
    X_test, y_test, model = load_data(args.input_data, args.model_path)
    probas = model.predict_proba(X_test)[:, 1]
    threshold = optimize_threshold(y_test, probas, beta=1.0)
    preds, probas, cm, y_test = evaluate_model(model, X_test, y_test, threshold)
    plot_metrics(cm, probas, y_test, args.output_path)
    print("✅ Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
