import argparse
import os
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import pandas as pd

def load_data(processed_path):
    X_train, y_train = joblib.load(os.path.join(processed_path, "train.pkl"))
    return X_train, y_train

def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2.0)

    param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3)
    }

    print("🔍 Tuning XGBoost for RECALL...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='recall',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print(f"\n✅ Best Recall: {search.best_score_:.4f}")
    print(f"✅ Best Params: {search.best_params_}")
    return search.best_estimator_

def save_model(model, output_path):
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(model, os.path.join(output_path, "best_model.pkl"))
    print(f"✅ Model saved at {output_path}/best_model.pkl")

def main(args):
    X_train, y_train = load_data(args.input_data)
    model = train_model(X_train, y_train)
    save_model(model, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
