# Agregar data de que no sea parte ninguna de las clases y ver donde la clasifica
import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.audio_processing import add_noise_to_audio, audio_to_logmel


config = Config()

MODEL_PATH = config.get("paths", "best_model_path")
PROCESSED_DIR = config.get("paths", "processed_dir")


# =========================
# DATOS DE VAL
# =========================
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
val_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_val_idx.npy"))

X_val = X[val_idx]

class_names = np.load(os.path.join(PROCESSED_DIR, "main_classes.npy"))
print("Clases de datos de validacion")
print(class_names)

# =========================
# DATOS SIN RELACION
# =========================
exp_val_idx = np.load(os.path.join(PROCESSED_DIR, "exp/exp_val_idx.npy"))
exp_val = X[exp_val_idx]

# Imprimir clases de datos sin relacion 
exp_classes = np.load(os.path.join(PROCESSED_DIR, "exp_classes.npy"))
print("Clases de datos sin relacion")
print(exp_classes)

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EVALUAR
# =========================

# Datos de validacion
probs = model.predict(X_val)
y_pred = np.argmax(probs, axis=1)

# Datos sin relacion
exp_probs = model.predict(exp_val)
exp_y_pred = np.argmax(exp_probs, axis=1)

# =========================
# ANALISIS
# =========================

exp_conf = np.max(exp_probs, axis=1)
val_conf = np.max(probs, axis=1)

# Distribución de predicción en datos sin relación
unique, counts = np.unique(y_pred, return_counts=True)

for cls, count in zip(unique, counts):
    print(f"Clase {class_names[cls]}: {count} muestras")

# Confianza del modelo en datos sin relación
print("\nConfianza del modelo en datos sin relacion")
print(f"Promedio: {np.mean(exp_conf):.4f}")
print(f"Desviacion estandar: {np.std(exp_conf):.4f}")
print(f"Mediana: {np.median(exp_conf):.4f}")
print(f"Minimo: {np.min(exp_conf):.4f}")
print(f"Maximo: {np.max(exp_conf):.4f}")

# Comparación con datos de validación
print("\nConfianza del modelo en datos conocidos")
print(f"Promedio: {np.mean(val_conf):.4f}")
print(f"Desviacion estandar: {np.std(val_conf):.4f}")
print(f"Mediana: {np.median(val_conf):.4f}")
print(f"Minimo: {np.min(val_conf):.4f}")
print(f"Maximo: {np.max(val_conf):.4f}")