import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks

# =========================
# CONFIGURACIÓN
# =========================

PROCESSED_DIR = "processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 5

# =========================
# CARGAR DATOS
# =========================

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    old_encoder = pickle.load(f)

old_class_names = old_encoder.classes_

print("X original:", X.shape)
print("y original:", y.shape)
print("Número original de clases:", len(np.unique(y)))

# =========================
# FILTRAR CLASES PEQUEÑAS
# =========================

classes, counts = np.unique(y, return_counts=True)

print("\nDistribución original:")
for c, count in zip(classes, counts):
    print(f"{old_class_names[c]}: {count}")

valid_classes = classes[counts >= MIN_SAMPLES_PER_CLASS]

mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]

kept_class_names = old_class_names[valid_classes]

# Reindexar etiquetas: 0, 1, 2, ...
new_encoder = LabelEncoder()
y_text = old_class_names[y]
y = new_encoder.fit_transform(y_text)

num_classes = len(np.unique(y))

print("\nDespués de filtrar clases pequeñas:")
print("X:", X.shape)
print("y:", y.shape)
print("Número de clases:", num_classes)

print("\nClases usadas:")
for cls, count in zip(*np.unique(y_text, return_counts=True)):
    print(f"{cls}: {count}")

with open(os.path.join(PROCESSED_DIR, "label_encoder_filtered.pkl"), "wb") as f:
    pickle.dump(new_encoder, f)

# =========================
# SPLIT TRAIN / VAL / TEST
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

print("\nTamaños:")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# =========================
# CLASS WEIGHTS
# =========================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights_array))

print("\nClass weights:")
print(class_weights)

# =========================
# MODELO CNN
# =========================

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model


model = build_model(X_train.shape[1:], num_classes)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, "best_cnn_model.keras"),
    monitor="val_accuracy",
    save_best_only=True
)

# =========================
# ENTRENAMIENTO
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# =========================
# EVALUACIÓN
# =========================

test_loss, test_acc = model.evaluate(X_test, y_test)

print("\n==============================")
print("RESULTADOS FINALES")
print("==============================")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# =========================
# GRÁFICAS
# =========================

plt.figure()
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy de entrenamiento y validación")
plt.savefig(os.path.join(MODEL_DIR, "accuracy_curve.png"), dpi=300)
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss de entrenamiento y validación")
plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"), dpi=300)
plt.show()