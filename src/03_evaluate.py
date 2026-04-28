import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from config import Config
from sklearn.metrics import classification_report, confusion_matrix

config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_PATH = config.get("paths", "best_model_path")
MIN_SAMPLES_PER_CLASS = config.get("dataset", "min_samples_per_class")
MODEL_DIR = config.get("paths", "model_dir")

# =========================
# DATOS DE TEST
# =========================
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
test_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_test_idx.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "main/main_test_y.npy"))

X_test = X[test_idx]

class_names = np.load(os.path.join(PROCESSED_DIR, "main_classes.npy"))

print(X_test.shape, y_test.shape)
print(class_names)

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
plt.savefig(os.path.join(MODEL_DIR, "test_confusion_matrix.png"), dpi=300)

plt.show()