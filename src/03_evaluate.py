import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from config import Config
config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_PATH = config.get("paths", "best_model_path")
MIN_SAMPLES_PER_CLASS = config.get("dataset", "min_samples_per_class")
MODEL_DIR = config.get("paths", "model_dir")

#test data
X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
test_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_test_idx.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "main/main_test_y.npy"))
X_test = X[test_idx]
class_names = np.load(os.path.join(PROCESSED_DIR, "main_classes.npy"))

print(X_test.shape, y_test.shape)
print(class_names)

# Modelo:
model = tf.keras.models.load_model(MODEL_PATH)

probs = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

# formato
print("\n VERIFICACIÓN")
print(".........................")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Número de clases encoder:", len(class_names))
print("Número de clases en y_test:", len(np.unique(y_test)))
print("Clases:", class_names)

print("\nDistribución de predicciones:")
print(np.bincount(y_pred, minlength=len(class_names)))

print("\n MÉTRICAS GENERALES")
print(".........................")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Accuracy sklearn: {accuracy_score(y_test, y_pred):.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nREPORTE POR CLASE")
print(".........................")
print(classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(class_names))
)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión - 12 clases")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(os.path.join(MODEL_DIR, "test_confusion_matrix.png"), dpi=300)

