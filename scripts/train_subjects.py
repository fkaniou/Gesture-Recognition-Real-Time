import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf
import yaml
import os
import random
from utils.train_functions import train_subject
from datetime import datetime

def set_seeds(seed_value=42):

    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def main(config_path):

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seeds(config["training_parameters"]["seed"])

    subjects = range(config["training_data"]["subject_start"], config["training_data"]["subject_end"] + 1)
    n_classes = config["training_data"]["classes"]
    training_params = config["training_parameters"]
    network_params = config["network"]

    total_acc, total_loss = [], []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config["logging"]["log_dir"]) / f"experiment_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:

        print(f"Training for Subject: {subject_id}")

        subject_metrics = train_subject(
            subject_id=subject_id,
            data_params=config["training_data"],
            training_params=training_params,
            network_params=network_params,
            log_dir=log_dir
        )

        total_acc.append(subject_metrics["val_accuracy"])
        total_loss.append(subject_metrics["val_loss"])


    average_acc = np.mean(total_acc)
    average_loss = np.mean(total_loss)

    print(f"\n Mean Validation Accuracy: {average_acc:.4f}")
    print(f" Mean Validation Loss: {average_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AtzoriNet model")
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    main(args.config)
