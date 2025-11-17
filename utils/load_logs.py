import os
from tensorflow.keras.models import load_model

def load_minmax(subject_dir, subject_id):
    
    txt_path = os.path.join(subject_dir, f"subject-{subject_id:02d}_minmax.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Minmax file not found: {txt_path}")

    with open(txt_path, "r") as f:
        lines = f.readlines()
        min_val = float(lines[0].strip().split(": ")[1])
        max_val = float(lines[1].strip().split(": ")[1])
    return min_val, max_val, txt_path

def load_model_for_subject(subject_dir):

    model_file = "best_model.h5" 
    model_path = os.path.join(subject_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path)
    return model, model_path

