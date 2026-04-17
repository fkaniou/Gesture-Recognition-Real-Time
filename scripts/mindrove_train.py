import sys
import os
import numpy as np
import scipy.io
from datetime import datetime
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.DataGenerator import DataGenerator
from modules.AtzoriNet import AtzoriNet

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Softmax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess data
def load_my_data(base_path):
    signals = []
    labels = []
    
    subjects_reps = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for folder in subjects_reps:
        path_to_preprocessed = os.path.join(base_path, folder, 'preprocessed')
        if not os.path.exists(path_to_preprocessed):
            continue
            
        print(f"Reading folder: {folder}...", end='\r')
        
        for file in os.listdir(path_to_preprocessed):
            if file.endswith(".mat") and "gesture" in file:
                file_path = os.path.join(path_to_preprocessed, file)
                mat_data = scipy.io.loadmat(file_path)
                
                emg_signal = mat_data.get('emg') 
                if emg_signal is None:
                    keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    emg_signal = mat_data[keys[0]]

                #PREPROCESSING
                emg_signal = emg_signal.astype('float32')
                
                # Rectification 
                emg_signal = np.abs(emg_signal)
                
                # Min-Max Normalization στο [0, 1]
                min_val = np.min(emg_signal)
                max_val = np.max(emg_signal)
                if max_val - min_val > 0:
                    emg_signal = (emg_signal - min_val) / (max_val - min_val)

                # CHANNEL PADDING 
                if emg_signal.shape[1] == 8:
                    padding = np.zeros((emg_signal.shape[0], 2), dtype='float32')
                    emg_signal = np.hstack((emg_signal, padding))

                # Εξαγωγή Label από το όνομα (gestureXX.mat)
                label = int(file.replace('gesture', '').replace('.mat', ''))
                
                signals.append(emg_signal)
                labels.append(label)
                
    return signals, labels


# 2. TRAINING PIPELINE
def run_pipeline():
    base_path = 'C:\\Users\\User\\Desktop\\semg_recording\\NaviFlame\\naviflame\\data\\recorded_ninapro'
    
    signals, labels = load_my_data(base_path)
    
    if len(signals) == 0:
        print("Error!No data found!")
        return

    # Split 80-10-10
    train_sig, temp_sig, train_lab, temp_lab = train_test_split(
        signals, labels, test_size=0.20, random_state=42, stratify=labels
    )
    val_sig, test_sig, val_lab, test_lab = train_test_split(
        temp_sig, temp_lab, test_size=0.50, random_state=42, stratify=temp_lab
    )

    W_SIZE = 100
    CHANNELS = 10 
    
    train_gen = DataGenerator(train_sig, train_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), window_size=W_SIZE, shuffle=True)
    val_gen   = DataGenerator(val_sig, val_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), window_size=W_SIZE, shuffle=False)
    test_gen  = DataGenerator(test_sig, test_lab, batch_size=64, dim=(W_SIZE, CHANNELS, 1), window_size=W_SIZE, shuffle=False)

    base_atzori = AtzoriNet(input_shape=(W_SIZE, CHANNELS, 1), classes=30, batch_norm=True)
    
    # ΠΑΡΕΜΒΑΣΗ: Παίρνουμε την έξοδο ΠΡΙΝ το Reshape για να αποφύγουμε το σφάλμα [64, 330]
    # Το index -3 μας δίνει το layer b5_soft (πριν το καταστροφικό reshape)
    x = base_atzori.layers[-3].output
    
    # Χρησιμοποιούμε GlobalAveragePooling2D για να "κλειδώσουμε" την έξοδο στις 30 κλάσεις
    x = GlobalAveragePooling2D(name='global_fix')(x)
    new_output = Softmax(name='final_softmax')(x)
    
    # Αυτό είναι το τελικό μοντέλο που θα εκπαιδεύσουμε
    model = Model(inputs=base_atzori.input, outputs=new_output, name='Fixed_AtzoriNet')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    # στ. Callbacks
    os.makedirs('mindrove_models', exist_ok=True)
    dt = datetime.now().strftime("%Y%m%d_%H%M")
    model_path = os.path.join('mindrove_models', f"atzorinet_{dt}.keras")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    # ζ. Εκπαίδευση
    print(f"\nStarting training (Window: {W_SIZE}, Channels: {CHANNELS})...")
    model.fit(train_gen, validation_data=val_gen, epochs=100, callbacks=callbacks)

    # η. Τελική Αξιολόγηση
    print("\n--- TEST EVALUATION ---")
    loss, acc = model.evaluate(test_gen)
    print(f"Final Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    run_pipeline()
