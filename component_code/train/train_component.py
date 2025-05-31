import argparse
import os
import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn

def load_data(processed_path):
    X_train, y_train = joblib.load(os.path.join(processed_path, "train.pkl"))
    X_val, y_val = joblib.load(os.path.join(processed_path, "val.pkl"))
    return X_train, y_train, X_val, y_val

def tune_model(model, param_dist, X_train, y_train):
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_

def main(args):
    X_train, y_train, X_val, y_val = load_data(args.input_data)

    # ✅ Calculate class imbalance ratio for XGBoost
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    print("🔍 Tuning XGBoost...")
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight  # ✅ imbalance handling
    )
    xgb_params = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(50, 200),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }
    xgb_best, xgb_best_params, xgb_score = tune_model(xgb, xgb_params, X_train, y_train)

    print("\n🔍 Tuning RandomForest...")
    rf = RandomForestClassifier(class_weight='balanced')  # ✅ imbalance handling
    rf_params = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'max_features': ['sqrt', 'log2']
    }
    rf_best, rf_best_params, rf_score = tune_model(rf, rf_params, X_train, y_train)

    print("\n🔍 Tuning LogisticRegression...")
    lr = LogisticRegression(solver='liblinear', class_weight='balanced')  # ✅ imbalance handling
    lr_params = {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2']
    }
    lr_best, lr_best_params, lr_score = tune_model(lr, lr_params, X_train, y_train)

    # Select best model
    scores = {'XGBoost': xgb_score, 'RandomForest': rf_score, 'LogisticRegression': lr_score}
    best_name = max(scores, key=scores.get)
    best_model = {'XGBoost': xgb_best, 'RandomForest': rf_best, 'LogisticRegression': lr_best}[best_name]
    best_score = scores[best_name]
    best_params = {'XGBoost': xgb_best_params, 'RandomForest': rf_best_params, 'LogisticRegression': lr_best_params}[best_name]

    print(f"\n🏆 Selected Model: {best_name} with F1: {round(best_score, 4)}")

    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_prob = best_model.predict_proba(X_val)[:, 1] if hasattr(best_model, "predict_proba") else None

    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else float('nan')
    conf_matrix = confusion_matrix(y_val, y_val_pred)

    # Save best model
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, "best_model.pkl")
    joblib.dump(best_model, model_path)

    # Save confusion matrix to file
    conf_matrix_file = os.path.join(args.output_path, "confusion_matrix.txt")
    with open(conf_matrix_file, "w") as f:
        f.write(np.array2string(conf_matrix))

    # Log to MLflow
    mlflow.log_param("selected_model", best_name)
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)
    mlflow.log_metric("val_accuracy", accuracy)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)
    mlflow.log_metric("val_roc_auc", roc_auc)
    mlflow.log_artifact(conf_matrix_file)
    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="credit-default-model")

    print("✅ Model training complete and all metrics logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)