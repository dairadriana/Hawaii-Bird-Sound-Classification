# Ver como le va al modelo con clases con las que no fue entrenado
# Hacer fine tunning a capa final con clases no vistas por el modelo anteriormente

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = "processed"
MODEL_PATH = "models/model_500_16k/best_cnn_model_500_samples.keras"

MAX_SAMPLES_PER_CLASS = 500

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    old_encoder = pickle.load(f)

old_class_names = old_encoder.classes_

# =========================
# FILTRAR IGUAL QUE EN TRAIN
# =========================

classes, counts = np.unique(y, return_counts=True)
valid_classes = classes[counts <= MAX_SAMPLES_PER_CLASS]

mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]

y_text = old_class_names[y]

mid_encoder = LabelEncoder()
y = mid_encoder.fit_transform(y_text)

class_names = mid_encoder.classes_

print("X filtrado:", X.shape)
print("y filtrado:", y.shape)
print("Número de clases:", len(class_names))

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)
num_classes = model.output_shape[1]

# =========================
# REDUCIR NUMERO DE CLASES A LAS QUE SOPORTA EL MODELO
# =========================
topClasses = np.argsort(counts)[-num_classes:]

mask = np.isin(y, topClasses)
X = X[mask]
y = y[mask]

y_text = mid_encoder.classes_[y]

new_encoder = LabelEncoder()
y = new_encoder.fit_transform(y_text)

class_names = new_encoder.classes_

print("X filtrado:", X.shape)
print("y filtrado:", y.shape)
print("Número de clases:", len(class_names))

# =========================
# EVALUAR
# =========================

probs = model.predict(X)
y_pred = np.argmax(probs, axis=1)

print("\nClassification report:")
print(classification_report(
    y,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot=False
)

plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión - Clasificación de cantos de aves")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=300)
plt.show()