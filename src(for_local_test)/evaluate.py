import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve, fbeta_score
)

def load_data():
    X_test, y_test = joblib.load("data/processed/test.pkl")
    model = joblib.load("data/model/best_model.pkl")
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

def plot_metrics(cm, probas, y_test):
    os.makedirs("doc", exist_ok=True)

    # Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("doc/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("doc/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig("doc/pr_curve.png")
    plt.close()

def main():
    beta = 1.0  # Set beta = 1 for F1, beta = 2 for F2
    X_test, y_test, model = load_data()
    probas = model.predict_proba(X_test)[:, 1]
    threshold = optimize_threshold(y_test, probas, beta=beta)
    preds, probas, cm, y_test = evaluate_model(model, X_test, y_test, threshold)
    plot_metrics(cm, probas, y_test)
    print("✅ Evaluation v4 complete with F-beta optimization.")

if __name__ == "__main__":
    main()
