import os
import warnings
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils.preprocessing_mindrove import to_volts, remove_dc, downsample, rms, anti_aliasing_filter, plot, plot_fft, lpf_post_fft, plot_pipeline_stages
from utils.preprocessing_functions import lpf, trim_data, augmentation, rearrange_channels
from modules.DataGenerator import DataGenerator
import glob
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
                #print("Data shape before rearranging:", data.shape)
                #print("First 5 rows of all channels:\n", data[:5])
                data = rearrange_channels(data, n_channels=data.shape[1])
                #print("Data shape after rearranging:", data.shape)
                #print("First 5 rows of all channels:\n", data[:5])
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

def load_mindrove_data(mindrove_data_dir,subject_id, min_value, max_value):
    x, y = [], []
    downsample_factor = 5
    fs_mindrove = 500

    subject_prefix = f"s{subject_id}_"
    
    subject_dirs = sorted(
        d for d in os.listdir(mindrove_data_dir)
        if os.path.isdir(os.path.join(mindrove_data_dir, d)) and d.startswith(subject_prefix)
    )

    #print(f"Found {len(subject_dirs)} folders for subject {subject_id}:")
    #for d in subject_dirs:
    #    print("  -", d)

    for folder in subject_dirs:
        folder_path = os.path.join(mindrove_data_dir, folder)

        mat_files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))

        for mat_file in mat_files:
            stages=[]
            titles=[]
            #print(f"  Loading: {os.path.basename(mat_file)}")
            mat_data = loadmat(mat_file)
            data = mat_data['emg']
            stages.append(data); titles.append("Raw Signal")
            #plot(data, title = "Raw Signal")
            data = to_volts(data)
            stages.append(data); titles.append("After converting to volts")
            #plot(data, title = "After converting to volts")
            data = remove_dc(data)
            stages.append(data); titles.append("After DC removal")
            #plot(data, title = "After DC removal")
            data = rms(data, window_size=10, step=1)
            stages.append(data); titles.append("After RMS")
            #plot(data, title = "After RMS")
            data = anti_aliasing_filter(data, cutoff=45., fs=fs_mindrove, order=4)
            stages.append(data); titles.append("After Anti-aliasing filter")
            #plot(data, title = "After Anti-aliasing filter")
            data = downsample(data, downsample_factor=downsample_factor)
            stages.append(data); titles.append("After downsampling")

            #plot(data, title = "After downsampling")
            #plot_fft(data[:,0], fs=100)

            #data = lpf(data, f=1., fs=100)
            data = lpf_post_fft(data, cutoff=10., fs=100, order=2)
            stages.append(data); titles.append("After LPF")
            #plot(data, title = "After LPF")

            data = (data - min_value) / (max_value - min_value)
            stages.append(data); titles.append("After Min-Max Normalization")
            #plot(data, title = "After Min-Max Normalization")

            #print(f"Data shape before rearranging: {data.shape}")
            #print("First 5 rows of all channels:\n", data[:5])
            data = rearrange_channels(data, n_channels=data.shape[1])
            #plot_pipeline_stages(stages, titles, fs=fs_mindrove, file_name=mat_file)

            #plot(data, title = "Final preprocessed signal")
            #print(f"Data shape after rearranging: {data.shape}")
            #print("First 5 rows of all channels:\n", data[:5])
            x.append(data)

            gesture_stim = int(mat_data['stimulus'][0][0])
            y.append(gesture_stim)

            #print(f"  Done preprocessing {os.path.basename(mat_file)}")

    #print("\nAll files processed for subject", subject_id)

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
    
def get_mindrove_data_generators(mindrove_data_dir, subject_id, min_value, max_value, batch_size=32, window_size=15, window_step=6):
    x_test, y_test = load_mindrove_data(mindrove_data_dir, subject_id, min_value, max_value)
    
    print(f"Loaded Mindrove EMG data shape (test): {x_test.shape}")

    test_generator = DataGenerator(x_test, y_test, batch_size=batch_size, window_size=window_size, window_step=window_step, shuffle=False)
    return test_generator
    