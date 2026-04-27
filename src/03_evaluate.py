import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

PROCESSED_DIR = "processed"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_cnn_model.keras")

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

class_names = encoder.classes_

model = tf.keras.models.load_model(MODEL_PATH)

probs = model.predict(X_test)
y_pred = np.argmax(probs, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

print("\n==============================")
print("MÉTRICAS GENERALES - TEST OFICIAL")
print("==============================")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\n==============================")
print("REPORTE POR CLASE")
print("==============================")
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
plt.title("Matriz de confusión - Test oficial")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=300)
plt.show()

cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_norm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot=False
)

plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión normalizada - Test oficial")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_normalized.png"), dpi=300)
plt.show()