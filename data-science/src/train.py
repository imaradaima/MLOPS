# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


def _resolve_csv(path_like: str, default_name: str) -> Path:
    """
    Accept either a directory (containing a single CSV or a known file name)
    or a direct path to a CSV, and return a concrete CSV Path.
    """
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_dir():
        # prefer a conventional file name (train.csv / test.csv) if present
        candidate = p / default_name
        if candidate.exists():
            return candidate
        # otherwise pick the first .csv we find
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found under: {p}")
        return csvs[0]
    raise FileNotFoundError(f"Path not found: {p}")


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("train")

    # -------- arguments --------
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (folder with train.csv or a CSV file)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data (folder with test.csv or a CSV file)")
    parser.add_argument("--model_output", type=str, required=True,
                        help="Output directory to save the MLflow model")

    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees in the RandomForest")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Maximum depth of trees (None expands fully)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallelism for RandomForest (-1 uses all cores)")
    return parser.parse_args()


def main(args):
    """Read train/test, train model, evaluate, save trained model"""

    # -------- Step 2: load datasets --------
    train_csv = _resolve_csv(args.train_data, "train.csv")
    test_csv  = _resolve_csv(args.test_data, "test.csv")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # -------- Step 3: split X / y (target column is 'price') --------
    target_col = "price"
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in training data columns: {list(train_df.columns)}")
    if target_col not in test_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in test data columns: {list(test_df.columns)}")

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    # -------- Step 4: initialize & train model --------
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    model.fit(X_train, y_train)

    # -------- Step 5: log params to MLflow --------
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth if args.max_depth is not None else "None")
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("n_jobs", args.n_jobs)
    mlflow.log_param("n_features", X_train.shape[1])

    # -------- Step 6: predict & evaluate --------
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (test): {mse:.4f}")

    # -------- Step 7: log metric & save model --------
    mlflow.log_metric("MSE", float(mse))

    out_dir = Path(args.model_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save as MLflow model directory (portable across steps)
    mlflow.sklearn.save_model(sk_model=model, path=str(out_dir))

    print(f"Saved MLflow model to: {out_dir.resolve()}")


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path : {args.test_data}",
        f"Model output path       : {args.model_output}",
        f"Number of Estimators    : {args.n_estimators}",
        f"Max Depth               : {args.max_depth}",
    ]
    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
