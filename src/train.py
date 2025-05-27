import joblib
import os
from sklearn.metrics import recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def load_data(processed_path):
    X_train, y_train = joblib.load(os.path.join(processed_path, "train.pkl"))
    X_val, y_val = joblib.load(os.path.join(processed_path, "val.pkl"))
    return X_train, y_train, X_val, y_val

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

def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "best_model.pkl"))
    print(f"✅ Model saved at {path}/best_model.pkl")

def main():
    processed_path = "data/processed"
    model_output_path = "data/model"
    X_train, y_train, X_val, y_val = load_data(processed_path)
    model = train_model(X_train, y_train)
    save_model(model, model_output_path)

if __name__ == "__main__":
    main()
