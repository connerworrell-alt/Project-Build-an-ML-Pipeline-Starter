import json
import os
import tempfile

import hydra
import mlflow
import wandb  # noqa: F401
from omegaconf import DictConfig
from hydra.utils import get_original_cwd

# Steps that run when main.steps == "all"
_STEPS = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_regression_model",
]


@hydra.main(config_path=".", config_name="config", version_base=None)
def go(config: DictConfig) -> None:
    # Group runs in W&B
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Determine which steps to run
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _STEPS

    # Get repo root explicitly (since Hydra changes working dir)
    here = get_original_cwd()
    orig_cwd = os.getcwd()

    try:
        # Work in a temporary clean directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)

            # 1) Download raw data
            if "download" in active_steps:
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/get_data",
                    "main",
                    version="main",
                    env_manager="conda",
                    parameters={
                        "sample": config["etl"]["sample"],
                        "artifact_name": "sample.csv",
                        "artifact_type": "raw_data",
                        "artifact_description": "Raw file as downloaded",
                    },
                )

            # 2) Basic cleaning
            if "basic_cleaning" in active_steps:
                _ = mlflow.run(
                    os.path.join(here, "src", "basic_cleaning"),
                    "main",
                    env_manager="local",
                    parameters={
                        "input_artifact": "sample.csv:latest",
                        "output_artifact": "clean_sample.csv",
                        "output_type": "clean_sample",
                        "output_description": "Data with outliers and non NYC rows removed",
                        "min_price": config["basic_cleaning"]["min_price"],
                        "max_price": config["basic_cleaning"]["max_price"],
                    },
                )

            # 3) Data validation
            if "data_check" in active_steps:
                _ = mlflow.run(
                    os.path.join(here, "src", "data_check"),
                    "main",
                    env_manager="conda",
                    parameters={
                        "csv": "clean_sample.csv:latest",
                        "ref": "clean_sample.csv:reference",
                        "kl_threshold": config["data_check"]["kl_threshold"],
                        "min_price": config["data_check"]["min_price"],
                        "max_price": config["data_check"]["max_price"],
                    },
                )

            # 4) Split dataset
            if "data_split" in active_steps:
                _ = mlflow.run(
                    f"{config['main']['components_repository']}/train_val_test_split",
                    "main",
                    version="main",
                    env_manager="conda",
                    parameters={
                        "input": "clean_sample.csv:latest",
                        "test_size": config["data_split"]["test_size"],
                        "random_seed": config["data_split"]["random_state"],
                        "stratify_by": config["data_split"]["stratify_by"],
                    },
                )

            # 5) Train the Random Forest
            if "train_random_forest" in active_steps:
                rf_config_path = os.path.abspath("rf_config.json")
                with open(rf_config_path, "w", encoding="utf-8") as fp:
                    json.dump(dict(config["modeling"]["random_forest"].items()), fp)

                _ = mlflow.run(
                    os.path.join(here, "src", "train_random_forest"),
                    "main",
                    env_manager="local",
                    parameters={
                        "trainval_artifact": "trainval_data.csv:latest",
                        "val_size": config["data_split"]["val_size"],
                        "random_seed": config["data_split"]["random_state"],
                        "stratify_by": config["data_split"]["stratify_by"],
                        "rf_config": rf_config_path,
                        "max_tfidf_features": config["data_split"]["max_tfidf_features"],
                        "output_artifact": "random_forest_export",
                    },
                )

            # 6) Test the trained model
            if "test_regression_model" in active_steps:
                # Force Python 3.10 for compatibility with training environment
                os.environ["MLFLOW_CONDA_CREATE_FLAGS"] = "python=3.10"

                _ = mlflow.run(
                    f"{config['main']['components_repository']}/test_regression_model",
                    "main",
                    version="main",
                    env_manager="conda",  # Use its own env so wandb_utils installs
                    parameters={
                        "mlflow_model": "random_forest_export:production",
                        "test_dataset": "test_data.csv:latest",
                    },
                )

    finally:
        # Ensure we return to the original directory
        try:
            os.chdir(orig_cwd)
        except Exception:
            os.chdir(here)


if __name__ == "__main__":
    go()
