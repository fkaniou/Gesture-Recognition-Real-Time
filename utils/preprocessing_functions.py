import numpy as np
from tensorflow import keras
import tensorflow as tf
import scipy.signal

# Butterworth 1 Hz low-pass filter
def lpf(x, f=1., fs=100):
    f = f / (fs / 2)
    x = np.abs(x)
    b, a = scipy.signal.butter(1, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output

def trim_data(data, target_length):
    if len(data) > target_length:
        cut = (len(data) - target_length) // 2
        return data[cut:cut + target_length]
    else:
        return data

def jitter(x, snr_db=25):
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    snr = 10 ** (snr_db/10)
    Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn

def augmentation(x):
    return jitter(x)

def rearrange_channels(data, n_channels):
    T = data.shape[0]
    new_data = np.zeros((T, 10), dtype=data.dtype)
    if n_channels == 8:
        new_data[:, 1:9] = data
    elif n_channels == 10:
        new_data[:, 1:9] = data[:, 0:8]
    else:
        raise ValueError(f"Unexpected number of channels: {n_channels}")
    return new_data