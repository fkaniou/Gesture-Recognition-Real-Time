import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.signal
from scipy.signal import welch
import matplotlib.pyplot as plt


# Butterworth 1 Hz low-pass filter
# def lpf(x, f=1., fs=100):
#     f = f / (fs / 2)
#     x = np.abs(x)
#     b, a = scipy.signal.butter(1, f, 'low')
#     output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
#     return output

def lpf_post_fft(x, cutoff=10., fs=100, order=2):
    nyq = fs / 2
    b, a = scipy.signal.butter(order, cutoff/nyq, btype='low')
    return scipy.signal.filtfilt(b, a, x, axis=0)

def anti_aliasing_filter(x, cutoff=45., fs=500, order=4):
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low')
    return scipy.signal.filtfilt(b, a, x, axis=0)

#0.045e-6
def to_volts(x, lsb=0.045e-6):
    return x * lsb

def remove_dc(x):
    mean = np.mean(x, axis=0, keepdims=True)
    return x - mean

def rms(x, window_size=5, step=1):
    out =[]
    for start in range(0, x.shape[0] - window_size + 1, step):
        seg = x[start:start + window_size, :]
        out.append(np.sqrt(np.mean(seg ** 2, axis=0)))
    return np.array(out)

def downsample(x, downsample_factor):
    return x[::downsample_factor, :]

def plot(data, title="EMG Signal"):
    plt.figure(figsize=(10,4))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("amplitude(volts)")
    plt.grid()
    plt.show()

def plot_fft(signal, fs=100):
    N = len(signal)
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    mag = np.abs(X)

    plt.figure(figsize=(10,4))
    plt.plot(freqs, mag)
    plt.title("FFT Spectrum")
    plt.xlabel("f (Hz)")
    plt.ylabel("size |X(f)|")
    plt.xlim(0, fs/2)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pipeline_stages(stages, titles, fs=500, n_samples=1000, n_cols=2, file_name=None):
    n = len(stages)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3*n_rows), sharex=False)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            length = min(n_samples, stages[i].shape[0])
            time = np.arange(length) / fs
            # plot all channels
            ax.plot(time, stages[i][:length, :])
            ax.set_title(titles[i])
            ax.grid(alpha=0.3)
        else:
            ax.axis("off")  # hide unused subplot

    # Add file name as figure-level header
    if file_name is not None:
        fig.text(0.5, 0.98, f"Preprocessing pipeline for {os.path.basename(file_name)}",
                 ha='center', va='top', fontsize=14)

    plt.tight_layout(rect=[0,0,1,0.95])  # leave space for header
    plt.show()
