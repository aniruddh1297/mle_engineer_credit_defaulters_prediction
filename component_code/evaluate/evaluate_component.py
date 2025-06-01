import argparse
import joblib
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mlflow
import shap
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

def load_data(input_data, model_path):
    X_test, y_test = joblib.load(os.path.join(input_data, "test.pkl"))
    model = joblib.load(os.path.join(model_path, "best_model.pkl"))
    return X_test, y_test, model

def optimize_cost_threshold(y_true, probas, cost_fp=1000, cost_fn=900):
    best_thresh = 0.5
    best_cost = float('inf')
    for t in np.arange(0.2, 0.7, 0.01):
        preds = (probas >= t).astype(int)
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        if total_cost < best_cost:
            best_cost = total_cost
            best_thresh = t
    print(f"💸 Best Threshold for Cost: {best_thresh:.2f} | Min Cost: {best_cost}")
    return best_thresh, best_cost

def evaluate_model(model, X_test, y_test, threshold):
    probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= threshold).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probas)
    cm = confusion_matrix(y_test, preds)
    return preds, probas, cm, y_test, acc, f1, roc_auc

def plot_metrics(cm, probas, y_test, output_path):
    os.makedirs(output_path, exist_ok=True)
    cm_path = os.path.join(output_path, "confusion_matrix.png")
    roc_path = os.path.join(output_path, "roc_curve.png")
    pr_path = os.path.join(output_path, "pr_curve.png")

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    return cm_path, roc_path, pr_path

def generate_shap_plot(model, X_test, output_path):
    shap_path = os.path.join(output_path, "shap_beeswarm.png")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)  # safer with summary_plot
        plt.tight_layout()
        plt.savefig(shap_path)
        plt.close()

        print("SHAP summary plot saved.")
        return shap_path
    except Exception as e:
        print(f"SHAP explainability failed: {e}")
        return None


def write_notes(output_path, cm, cost):
    notes_path = os.path.join(output_path, "model_notes.txt")
    with open(notes_path, "w") as f:
        f.write("Model Limitations and Potential Biases:\n")
        f.write("-- Class imbalance may affect precision/recall.\n")
        f.write("-- Cost-based threshold used instead of F1 optimization.\n")
        f.write("-- Model doesn't consider temporal trends or behavioral drift.\n")
        f.write(f"-- Estimated total cost of misclassification: {cost}\n")
        f.write(f"-- Confusion Matrix: {cm.tolist()}\n")
    return notes_path

def main(args):
    X_test, y_test, model = load_data(args.input_data, args.model_path)
    probas = model.predict_proba(X_test)[:, 1]
    threshold, cost = optimize_cost_threshold(y_test, probas)
    preds, probas, cm, y_test, acc, f1, roc_auc = evaluate_model(model, X_test, y_test, threshold)
    cm_path, roc_path, pr_path = plot_metrics(cm, probas, y_test, args.output_path)
    notes_path = write_notes(args.output_path, cm, cost)
    shap_path = generate_shap_plot(model, X_test, args.output_path)

    mlflow.log_param("evaluation_threshold", threshold)
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)
    mlflow.log_metric("test_samples", len(y_test))
    mlflow.log_metric("estimated_misclassification_cost", cost)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(pr_path)
    mlflow.log_artifact(notes_path)
    if shap_path:
        mlflow.log_artifact(shap_path)

    print("Evaluation complete. Business impact optimized and logged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
