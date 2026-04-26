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
MODEL_PATH = "models/best_cnn_model.keras"
MIN_SAMPLES_PER_CLASS = 5

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    old_encoder = pickle.load(f)

old_class_names = old_encoder.classes_

# =========================
# FILTRAR IGUAL QUE EN TRAIN
# =========================

classes, counts = np.unique(y, return_counts=True)
valid_classes = classes[counts >= MIN_SAMPLES_PER_CLASS]

mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]

y_text = old_class_names[y]

new_encoder = LabelEncoder()
y = new_encoder.fit_transform(y_text)

class_names = new_encoder.classes_

print("X filtrado:", X.shape)
print("y filtrado:", y.shape)
print("Número de clases:", len(class_names))

# =========================
# MISMO SPLIT QUE EN TRAIN
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EVALUAR
# =========================

probs = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

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