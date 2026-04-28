# Clasificar ruido aleatorio y ver donde lo clasifica

import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from audio_processing import (
    add_noise_to_audio,
    audio_to_logmel,
    preprocess_audio_segment,
    find_audio_file
)

config = Config()

MODEL_PATH = config.get("paths", "best_model_path")
PROCESSED_DIR = config.get("paths", "processed_dir")
CLASS_NAMES = np.load(os.path.join(PROCESSED_DIR, "main_classes.npy"))

SAMPLE_LENGTH = int(
    config.get("audio", "sample_rate") *
    config.get("audio", "segment_duration")
)


# =========================
# GENERAR RUIDO ALEATORIO
# =========================

n_noise = 100

noise_samples = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=(n_noise, SAMPLE_LENGTH)
).astype(np.float32)

# Convertir a logmel
noise_logmel = []
for audio in tqdm(noise_samples, desc="Generando ruido"):
    noise_logmel.append(audio_to_logmel(audio))

noise_logmel = np.array(noise_logmel)
noise_logmel = noise_logmel[..., np.newaxis]

# =========================
# MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# PREDECIR
# =========================

noise_pred = model.predict(noise_logmel)
y_noise_pred = np.argmax(noise_pred, axis=1)
noise_conf = np.max(noise_pred, axis=1)

# =========================
# ANALISIS
# =========================

print("Distribución de clases para ruido aleatorio:")
unique, counts = np.unique(y_noise_pred, return_counts=True)

for cls, count in zip(unique, counts):
    print(f"Clase {CLASS_NAMES[cls]}: {count} muestras")

print("\nConfianza promedio para ruido:", np.mean(noise_conf))
print("Confianza minima:", np.min(noise_conf))
print("Confianza maxima:", np.max(noise_conf))
print("Mediana confianza:", np.median(noise_conf))

