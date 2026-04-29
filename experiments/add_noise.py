import os
import sys

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.audio_processing import add_noise_to_audio, audio_to_logmel


config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_PATH = config.get("paths", "best_model_path")
MIN_SAMPLES_PER_CLASS = config.get("data", "min_samples_per_class")

NOISE_LEVELS = config.get("add_noise", "noise_levels")

kept_class_names = np.load(
    os.path.join(PROCESSED_DIR, "main_classes.npy"),
    allow_pickle=True
)


X_raw = np.load(os.path.join(PROCESSED_DIR, "X_raw.npy"), mmap_mode="r")
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
idx_val = np.load(os.path.join(PROCESSED_DIR, "main/main_val_idx.npy"))

X_val = X_raw[idx_val]

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EVALUAR PARA DIFERENTES NIVELES DE RUIDO
# =========================

results = []

X_val_noise = X_val
X_val_noise_mel = []
for audio in X_val_noise:
    X_val_noise_mel.append(audio_to_logmel(audio))
X_val_noise_mel = np.array(X_val_noise_mel)
X_val_noise_mel = X_val_noise_mel[..., np.newaxis]

probs = model.predict(X_val_noise_mel, verbose=0)
y_pred = np.argmax(probs, axis=1)

report = classification_report(
    y_val,
    y_pred,
    target_names=kept_class_names,
    output_dict=True,
    zero_division=0
)

results.append({
    "SNR (dB)": "None",
    "Accuracy": report["accuracy"],
    "Macro F1": report["macro avg"]["f1-score"],
    "Weighted F1": report["weighted avg"]["f1-score"]
})

for level in tqdm(NOISE_LEVELS, desc="Evaluando niveles de ruido"):
    X_val_noise = add_noise_to_audio(X_val, level)

    X_val_noise_mel = []
    for audio in X_val_noise:
        X_val_noise_mel.append(audio_to_logmel(audio))
    
    X_val_noise_mel = np.array(X_val_noise_mel)
    X_val_noise_mel = X_val_noise_mel[..., np.newaxis]

    probs = model.predict(X_val_noise_mel, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(
        y_val,
        y_pred,
        target_names=kept_class_names,
        output_dict=True,
        zero_division=0
    )
    
    results.append({
        "SNR (dB)": level,
        "Accuracy": report["accuracy"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Weighted F1": report["weighted avg"]["f1-score"]
    })

# =========================
# MOSTRAR TABLA COMPARATIVA
# =========================

df = pd.DataFrame(results)
print("\n" + "="*50)
print("TABLA COMPARATIVA: RENDIMIENTO VS RUIDO")
print("="*50)
print(df.to_string(index=False, float_format="%.4f"))
print("="*50)
