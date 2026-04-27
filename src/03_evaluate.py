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

PROCESSED_DIR = "processed"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "best_cnn_model_12_classes.keras")
ENCODER_PATH = os.path.join(PROCESSED_DIR, "label_encoder.pkl")

X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(ENCODER_PATH, "rb") as f:
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
print("VERIFICACIÓN")
print("==============================")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("Número de clases encoder:", len(class_names))
print("Número de clases en y_test:", len(np.unique(y_test)))
print("Clases:", class_names)

print("\nDistribución de predicciones:")
print(np.bincount(y_pred, minlength=len(class_names)))

print("\n==============================")
print("MÉTRICAS GENERALES")
print("==============================")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Accuracy sklearn: {accuracy_score(y_test, y_pred):.4f}")
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
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix_12_classes.png"), dpi=300)
plt.show()