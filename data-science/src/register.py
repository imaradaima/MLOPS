# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
import json


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')  # str
    parser.add_argument('--model_path', type=str, help='Model directory')                             # str
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")   # str
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')
    return args


def main(args):
    """Loads the best-trained model from the sweep job and registers it"""

    print("Registering", args.model_name)

    # --- Step 1: Load the model directory as an MLflow sklearn model ---
    # args.model_path should point to a folder containing MLmodel metadata
    model = mlflow.sklearn.load_model(args.model_path)

    # --- Step 2: Log the loaded model into the current MLflow run ---
    # Use a stable artifact subfolder name
    artifact_path = "random_forest_price_regressor"
    logged = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path
    )
    # logged.model_uri is like "runs:/<run_id>/<artifact_path>"
    model_uri = logged.model_uri

    # --- Step 3: Register the logged model to the Model Registry ---
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    # mv.version holds the registered version

    print(f"Registered model '{args.model_name}' as version {mv.version} from {model_uri}")

    # --- Step 4: Write registration details to JSON for downstream steps ---
    out_dir = Path(args.model_info_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "model_name": args.model_name,
        "registered_version": mv.version,
        "model_uri": model_uri,
        "run_id": logged.run_id if hasattr(logged, "run_id") else mlflow.active_run().info.run_id,
        "artifact_path": artifact_path,
    }
    with open(out_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("Wrote model info to:", out_dir / "model_info.json")


if __name__ == "__main__":
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}",
    ]
    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

 
