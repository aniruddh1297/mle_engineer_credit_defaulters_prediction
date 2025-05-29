import joblib
import os
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import warnings

warnings.filterwarnings("ignore")


def load_data(processed_path):
    X_train, y_train = joblib.load(os.path.join(processed_path, "train.pkl"))
    X_val, y_val = joblib.load(os.path.join(processed_path, "val.pkl"))
    return X_train, y_train, X_val, y_val


def train_and_select_best(X_train, y_train):
    model_configs = {
        "XGBoost": {
            "estimator": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=2.0),
            "param_dist": {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 8),
                'learning_rate': uniform(0.01, 0.2),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3)
            }
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(),
            "param_dist": {
                'n_estimators': randint(100, 300),
                'max_depth': randint(3, 10),
                'max_features': ['auto', 'sqrt', 'log2']
            }
        },
        "LogisticRegression": {
            "estimator": LogisticRegression(solver='liblinear'),
            "param_dist": {
                'C': uniform(0.1, 10.0),
                'penalty': ['l1', 'l2']
            }
        }
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, config in model_configs.items():
        print(f"\n🔍 Tuning {name}...")
        search = RandomizedSearchCV(
            estimator=config["estimator"],
            param_distributions=config["param_dist"],
            n_iter=20,
            scoring=make_scorer(f1_score),
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"✅ {name} Best F1: {search.best_score_:.4f}")
        print(f"✅ {name} Best Params: {search.best_params_}")

        if search.best_score_ > best_score:
            best_score = search.best_score_
            best_model = search.best_estimator_
            best_name = name

    print(f"\n🏆 Selected Model: {best_name} with F1: {best_score:.4f}")
    return best_model


def save_model(model, path):
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "best_model.pkl"))
    print(f"✅ Model saved at {path}/best_model.pkl")


def main():
    processed_path = "data/processed"
    model_output_path = "data/model"
    X_train, y_train, X_val, y_val = load_data(processed_path)
    model = train_and_select_best(X_train, y_train)
    save_model(model, model_output_path)


if __name__ == "__main__":
    main()
