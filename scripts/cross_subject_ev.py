import os
import numpy as np
from tensorflow.keras.models import load_model
from utils.data_utils import get_mindrove_data_generators
from utils.load_logs import load_minmax, load_model_for_subject

def cross_subject_eval(experiment_dir, mindrove_data_dir, mindrove_subjects=range(1, 5),batch_size=32, window_size=15, window_step=6):
    results = []

    for subject_folder in sorted(os.listdir(experiment_dir), key=lambda x: int(x.split("_")[1])):
        subject_dir = os.path.join(experiment_dir, subject_folder)
        if not os.path.isdir(subject_dir):
            continue

        if not subject_folder.startswith("Subject_"):
            continue
        ninapro_id = int(subject_folder.split("_")[1])

        try:
            model, model_path = load_model_for_subject(subject_dir)
            min_val, max_val, minmax_path = load_minmax(subject_dir, ninapro_id)
        except FileNotFoundError as e:
            print(f"Skipping {subject_folder}: {e}")
            continue

        print(f"\n=== Evaluating {subject_folder} ===")
        print(f"Model: {model_path}")
        print(f"MinMax: {minmax_path} | min={min_val:.6f}, max={max_val:.6f}")

        for mindrove_id in mindrove_subjects:
            test_gen = get_mindrove_data_generators(
                mindrove_data_dir, mindrove_id, min_val, max_val,
                batch_size=batch_size, window_size=window_size, window_step=window_step
            )

            loss, acc = model.evaluate(test_gen, verbose=0)
            print(f"Ninapro subject: {ninapro_id} -> Mindrove subject: {mindrove_id}: Acc={acc:.4f}, Loss={loss:.4f}")
            results.append((ninapro_id, mindrove_id, acc, loss))

    if results:
        accs = [r[2] for r in results]
        losses = [r[3] for r in results]
        print(f"\nOverall Mean Accuracy: {np.mean(accs):.4f}")
        print(f"Overall Mean Loss: {np.mean(losses):.4f}")

    return results

if __name__ == "__main__":
    experiment_dir = "C:\\Users\\User\\Desktop\\Gesture-Recognition-Real-Time\\logs\\experiment_20251117_133316"
    mindrove_data_dir = "C:\\Users\\User\\Desktop\\Gesture-Recognition-Real-Time\\data\\mindrove_data\\data"

    results = cross_subject_eval(experiment_dir, mindrove_data_dir, mindrove_subjects=range(1, 5))


