from modules.AtzoriNet import AtzoriNet
from utils.data_utils import get_data_generators
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from modules.logger import Logger, EpochLogger
import sys
import csv
import os
import yaml

def step_decay(epoch):
    drop_rate = 0.5  
    epochs_drop = 25
    initial_lr = 0.001
    lr = initial_lr * (drop_rate ** (epoch // epochs_drop))
    print(f"Epoch {epoch+1}: Learning Rate = {lr}")
    return lr

def save_history_csv(history, out_path):
    csv_file = os.path.join(out_path, "training_history.csv")

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "accuracy", "val_loss", "val_accuracy"])

        n_epochs = len(history.history["loss"])
        for i in range(n_epochs):
            writer.writerow([
                i + 1,
                history.history["loss"][i],
                history.history["accuracy"][i],
                history.history["val_loss"][i],
                history.history["val_accuracy"][i],
            ])

    print(f"Saved CSV --> {csv_file}")

def save_params_yaml(data_params, training_params, network_params, out_path):
    yaml_path = os.path.join(out_path, "params.yaml")

    all_params = {
        "data_params": data_params,
        "training_params": training_params,
        "network_params": network_params
    }

    with open(yaml_path, "w") as f:
        yaml.dump(all_params, f)

    print(f"Saved config --> {yaml_path}")

def train_subject(subject_id, data_params, training_params, network_params, log_dir):

    # === Create subject-specific directory ===
    subject_dir = os.path.join(log_dir, f"Subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)

    # Redirect stdout to log file inside subject folder
    sys.stdout = Logger(os.path.join(subject_dir, "training_log.txt"))
    logger = EpochLogger()

    train_generator, val_generator = get_data_generators(
        subject_id,
        data_params["data_path"],
        data_params["gestures"],
        data_params["reps"],
        training_params["train_ratio"],
        subject_dir,
        training_params["seed"],
        training_params["batch_size"],
        training_params["window_size"],
        training_params["window_step"]
    )

    model = AtzoriNet(
        input_shape=network_params["input_shape"],
        classes=data_params["classes"],
        n_pool=network_params["n_pool"],
        n_dropout=network_params["n_dropout"],
        n_l2=network_params["n_l2"],
        n_init=network_params["n_init"],
        batch_norm=network_params["batch_norm"]
    )

    optimizer = Adam(learning_rate=training_params["initial_lr"]) \
        if training_params["optimizer"] == "adam" \
        else SGD(learning_rate=training_params["initial_lr"], momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(subject_dir, "best_model.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    tensorboard_callback = TensorBoard(log_dir=subject_dir)
    lr_scheduler = LearningRateScheduler(step_decay)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=training_params["epochs"],
        callbacks=[checkpoint, logger, tensorboard_callback, lr_scheduler],
        verbose=0  # no batch output
    )

    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Subject {subject_id} --> Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss:.4f}")

    final_model_path = os.path.join(subject_dir, "final_model.h5")
    print(final_model_path)
    model.save(final_model_path)

    save_history_csv(history, subject_dir)
    save_params_yaml(data_params, training_params, network_params, subject_dir)

    return {
        "subject": subject_id,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss
    }