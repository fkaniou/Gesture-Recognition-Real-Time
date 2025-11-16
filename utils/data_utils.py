import os
import warnings
import numpy as np
from scipy.io import loadmat
from utils.preprocessing_functions import lpf, trim_data, augmentation
from modules.DataGenerator import DataGenerator

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def load_emg_data(data_dir, subject_id, gestures, reps, log_dir, augment = False, l = 180, fs = 100, train=True):
    x, y = [], []
  
    for gesture in gestures:
        gesture_dir = os.path.join(data_dir, f"subject-{subject_id:02d}", f"gesture-{gesture:02d}", "rms")

        for rep in reps: 
            if gesture == 0:               
                file_path = os.path.join(gesture_dir, f"rep-{rep:02d}_01.mat")
            else:
                file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")

            if os.path.exists(file_path):
                data = loadmat(file_path)['emg']
                if len(data) < l:
                    l = len(data)                           
     
    print(f"Minimum length across all trials: {l} samples")
    print(f"Central seconds that I keep: {l / fs:.2f} s")

    print(f"Minimum length across all samples: {l}")

    # Load, trim & lpf

    for gesture in gestures:
        gesture_dir = os.path.join(data_dir, f"subject-{subject_id:02d}", f"gesture-{gesture:02d}", "rms")
                        
        for rep in reps:    
            if gesture == 0:               
                file_path = os.path.join(gesture_dir, f"rep-{rep:02d}_01.mat")
            else:
                file_path = os.path.join(gesture_dir, f"rep-{rep:02d}.mat")

            if os.path.exists(file_path):
                data = loadmat(file_path)['emg']
                data = trim_data(data, l)
                data = lpf(data)
                gesture_stim = int(loadmat(file_path)['stimulus'][0][0])
                # if augment and np.random.rand() < 0.8:
                #     data = augmentation(data)
                x.append(data)
                y.append(gesture_stim)                   
 
    
    print("Distribution", np.unique(y, return_counts=True))

    if train:
        max_x = max([arr.max() for arr in x])
        min_x = min([arr.min() for arr in x])

        log_file = os.path.join(log_dir, f"subject-{subject_id:02d}_minmax.txt")
        with open(log_file, "w") as f:
            f.write(f"min: {min_x:.6f}\n")
            f.write(f"max: {max_x:.6f}\n")
        
        print(f"Subject {subject_id:02d} | Min: {min_x:.4f} | Max: {max_x:.4f}")
    
        x = [(arr - min_x) / (max_x - min_x) for arr in x]
    else:
        log_file = os.path.join(log_dir, f"subject-{subject_id:02d}_minmax.txt")
        with open(log_file, "r") as f:
            lines = f.readlines()
            min_x = float(lines[0].strip().split(": ")[1])
            max_x = float(lines[1].strip().split(": ")[1])
        
        print(f"Subject {subject_id:02d} | Min: {min_x:.4f} | Max: {max_x:.4f}")

        x = [(arr - min_x) / (max_x - min_x) for arr in x]

    return np.array(x, dtype=object), np.array(y, dtype=object)


def get_data_generators(subject_id, data_path, gestures, reps, train_ratio, log_dir, seed=42, batch_size = 32, window_size=15, window_step=6):
    reps = range(1, reps+1)
    reps = np.array(reps)
    rng = np.random.default_rng(seed)  
    rng.shuffle(reps)
    gestures = range(0, gestures)

    n_train = int(len(reps) * train_ratio)
    train_reps = reps[:n_train].tolist()
    val_reps = reps[n_train:].tolist()

    print(f"Training Reps: {train_reps}")
    print(f"Validation Reps: {val_reps}")
    print(f"Subject ID: {subject_id}")
    
    x_train, y_train = load_emg_data(data_path, subject_id, gestures, train_reps, log_dir, augment = True, train=True)
    x_val, y_val = load_emg_data(data_path, subject_id, gestures, val_reps, log_dir, augment= False, train=False)
    print(f"Loaded EMG data shape (train): {x_train.shape}")
    print(f"Loaded EMG data shape (val): {x_val.shape}")

    train_generator = DataGenerator(x_train, y_train, batch_size=batch_size, window_size=window_size, window_step=window_step)
    X, y = next(iter(train_generator))
    print("Batch X shape:", X.shape)
    print("Batch y shape:", y.shape)

    val_generator = DataGenerator(x_val, y_val, batch_size=batch_size, window_size=window_size, window_step=window_step, shuffle=False)
    
    return train_generator, val_generator
    
        