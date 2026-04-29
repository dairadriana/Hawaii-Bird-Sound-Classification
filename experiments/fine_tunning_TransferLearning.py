# Ver como le va al modelo con clases con las que no fue entrenado
# Hacer fine tunning a capa final con clases no vistas por el modelo anteriormente
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from sklearn.metrics import classification_report

# =========================
# CONFIGURACIÓN
# =========================

config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_DIR = config.get("paths", "model_dir")
MODEL_PATH = config.get("paths", "best_model_path")

# =========================
# CARGAR DATOS
# =========================

print("Cargando datos...")
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"), mmap_mode='r')

# Cargamos las clases de 'exp' porque son las "no vistas" según el objetivo
exp_classes = np.load(
    os.path.join(PROCESSED_DIR, "exp_classes.npy"),
    allow_pickle=True
)

num_classes = len(exp_classes)

# Corregimos los nombres de los archivos en la carpeta 'exp' (basado en el contenido del disco)
train_idx = np.load(os.path.join(PROCESSED_DIR, "exp/exp_train_idx.npy"))
val_idx   = np.load(os.path.join(PROCESSED_DIR, "exp/exp_val_idx.npy"))

y_train = np.load(os.path.join(PROCESSED_DIR, "exp/exp_train_y.npy"))
y_val   = np.load(os.path.join(PROCESSED_DIR, "exp/exp_val_y.npy"))

# =========================
# INDEXAR FEATURES
# =========================
X_train = X[train_idx]
X_val   = X[val_idx]

print("\nTamaños:")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val: {X_val.shape}, {y_val.shape}")
print(f"Número de clases nuevas: {num_classes}")

# =========================
# MODELO
# =========================

print(f"\nCargando modelo base de: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# REDIMENSIONAR CAPA FINAL
# =========================

new_model = tf.keras.Sequential(model.layers[:-1])

new_model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="new_dense_output"))

# =========================
# CONGELAR CAPAS BASE
# =========================
# Congelamos todas las capas excepto la última que acabamos de añadir
for layer in new_model.layers[:-1]:
    layer.trainable = False

# =========================
# ENTRENAR CAPA FINAL
# =========================
new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

new_model.summary()

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_model_fine_tunning.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

print("\nIniciando entrenamiento de la capa final...")
history = new_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, checkpoint]
)

# =========================
# GRÁFICAS
# =========================

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Curvas de Accuracy")

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("Curvas de Loss")

plt.tight_layout()
graph_path = os.path.join(MODEL_DIR, "fine_tuning_curves.png")
plt.savefig(graph_path, dpi=300)
print(f"\nGráficas guardadas en: {graph_path}")

# =========================
# EVALUACIÓN FINAL
# =========================
print("\nEvaluando mejor modelo en set de validación...")
best_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "best_model_fine_tunning.keras"))
y_pred_probs = best_model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nReporte de Clasificación:")
print(classification_report(y_val, y_pred, target_names=exp_classes))