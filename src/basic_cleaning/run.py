#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # Single W&B run
    run = wandb.init(job_type="basic_cleaning", save_code=True, project="nyc_airbnb", group="cleaning")
    run.config.update(vars(args))

    # 1) Download the input artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # 2) Drop price outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # 3) Convert dates
    df["last_review"] = pd.to_datetime(df["last_review"])

    # 4) Keep only NYC bounding box
    in_box = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[in_box].copy()

    # 5) Save and log cleaned artifact
    output_path = "clean_sample.csv"
    df.to_csv(output_path, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,              # expect "clean_sample"
        description=args.output_description
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument("--input_artifact", type=str,
                        help="Input artifact to download, e.g., 'sample.csv:latest'", required=True)
    parser.add_argument("--output_artifact", type=str,
                        help="Name for the output artifact, e.g., 'clean_sample.csv'", required=True)
    parser.add_argument("--output_type", type=str,
                        help="Type of artifact, use 'clean_sample'", required=True)
    parser.add_argument("--output_description", type=str,
                        help="Short description for the cleaned dataset", required=True)
    parser.add_argument("--min_price", type=float,
                        help="Minimum nightly price to keep", required=True)
    parser.add_argument("--max_price", type=float,
                        help="Maximum nightly price to keep", required=True)

    args = parser.parse_args()
    go(args)

