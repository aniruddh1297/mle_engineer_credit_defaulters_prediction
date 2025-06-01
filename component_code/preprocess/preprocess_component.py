import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow

def load_data(file_path):
    df = pd.read_excel(file_path, header=1)
    df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)
    return df

def clean_data(df):
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['education'] = df['education'].replace(0, np.nan)
    df['marriage'] = df['marriage'].replace(0, np.nan)
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    df['avg_bill_amt'] = df[[f'bill_amt{i}' for i in range(1, 7)]].mean(axis=1)
    df['avg_pay_amt'] = df[[f'pay_amt{i}' for i in range(1, 7)]].mean(axis=1)
    df['pay_ratio'] = (df['avg_pay_amt'] / df['avg_bill_amt']).replace([np.inf, -np.inf], 0).fillna(0)
    df['recent_default_flag'] = (df['pay_0'] >= 1).astype(int)
    df['max_pay_delay'] = df[['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']].max(axis=1)
    df['bill_trend_up'] = (df['bill_amt6'] > df['bill_amt1']).astype(int)
    df['pay_stability'] = df[[f'pay_amt{i}' for i in range(1, 7)]].std(axis=1).fillna(0)

    y = df['default_payment_next_month']
    X = df.drop(columns=['default_payment_next_month'])

    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, columns=X.columns)

    return X_scaled, y, scaler

def split_and_save(X, y, output_dir):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((X_train, y_train), os.path.join(output_dir, "train.pkl"))
    joblib.dump((X_val, y_val), os.path.join(output_dir, "val.pkl"))
    joblib.dump((X_test, y_test), os.path.join(output_dir, "test.pkl"))

    return {
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "num_features": X.shape[1]
    }

def main(args):
    df = load_data(args.input_data)
    df = clean_data(df)
    X, y, scaler = feature_engineering(df)
    split_stats = split_and_save(X, y, args.output_path)
    scaler_path = os.path.join(args.output_path, "scaler.pkl")
    joblib.dump(scaler, scaler_path)

    
    mlflow.log_param("scaler", "StandardScaler")
    mlflow.log_param("num_rows", len(df))
    mlflow.log_param("num_features", split_stats["num_features"])
    mlflow.log_metric("train_size", split_stats["train_size"])
    mlflow.log_metric("val_size", split_stats["val_size"])
    mlflow.log_metric("test_size", split_stats["test_size"])

    
    class_counts = y.value_counts().to_dict()
    for label, count in class_counts.items():
        mlflow.log_metric(f"label_{label}_count", count)

    
    mlflow.log_artifact(scaler_path)

    print("Preprocessing complete and parameters logged with MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
