# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser("prep")  # Create an ArgumentParser object
    parser.add_argument("--raw_data", type=str, help="Path to raw data")           # str path to input CSV
    parser.add_argument("--train_data", type=str, help="Path to train dataset")    # str output folder
    parser.add_argument("--test_data", type=str, help="Path to test dataset")      # str output folder
    parser.add_argument(
        "--test_train_ratio", type=float, default=0.2,
        help="Test-train ratio (e.g., 0.2 means 20% test, 80% train)"
    )
    args = parser.parse_args()
    return args


def main(args):
    """Read, preprocess, split, and save datasets"""

    # Reading Data
    df = pd.read_csv(args.raw_data)

    # ------- PREPROCESSING STEPS -------

    # Normalize column names gently (strip spaces, lower) — helps avoid case/space mismatches.
    df.columns = (
        df.columns.str.strip()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.lower()
    )

    # Identify target if present; default to 'price' per project
    target_col = "price" if "price" in df.columns else None

    # Step 1: Label-encode categorical features (excluding target if it's categorical by accident)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    for col in cat_cols:
        le = LabelEncoder()
        # cast to string to avoid issues with mixed types
        df[col] = le.fit_transform(df[col].astype(str))

    # Step 2: Split train/test (stratify on segment if available and has >1 class)
    # handle either 'segment' or legacy 'type'
    seg_col = None
    for candidate in ["segment", "type"]:
        if candidate in df.columns:
            seg_col = candidate
            break

    stratify = None
    if seg_col is not None and df[seg_col].nunique() > 1:
        stratify = df[seg_col]

    test_size = float(args.test_train_ratio)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=stratify
    )

    # Step 3: Save train/test CSVs to their respective folders
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_path = Path(args.train_data) / "train.csv"
    test_path = Path(args.test_data) / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train dataset to: {train_path}  (rows={len(train_df)})")
    print(f"Saved test dataset to : {test_path}   (rows={len(test_df)})")

    # Step 4: Log simple metrics to MLflow
    mlflow.log_metric("train_rows", int(len(train_df)))
    mlflow.log_metric("test_rows", int(len(test_df)))
    if target_col and target_col in df.columns:
        mlflow.log_param("target", target_col)
    if seg_col:
        mlflow.log_param("stratify_by", seg_col)


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()  # Call the function to parse arguments

    lines = [
        f"Raw data path: {args.raw_data}",           # Print the raw_data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",      # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",     # Print the test_train_ratio
    ]
    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

  
