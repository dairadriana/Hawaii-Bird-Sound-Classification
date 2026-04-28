import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import Config
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import layers, models, callbacks

# =========================
# CONFIGURACIÓN
# =========================

config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_DIR = config.get("paths", "model_dir")
MODEL_PATH = config.get("paths", "best_model_path")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# CARGAR DATOS
# =========================

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
main_classes = np.load(
    os.path.join(PROCESSED_DIR, "main_classes.npy"),
    allow_pickle=True
)

num_classes = len(main_classes)

train_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_train_idx.npy"))
val_idx   = np.load(os.path.join(PROCESSED_DIR, "main/main_val_idx.npy"))

y_train = np.load(os.path.join(PROCESSED_DIR, "main/main_train_y.npy"))
y_val   = np.load(os.path.join(PROCESSED_DIR, "main/main_val_y.npy"))

# =========================
# INDEXAR FEATURES
# =========================
X_train = X[train_idx]
X_val   = X[val_idx]

np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
np.save(os.path.join(PROCESSED_DIR, "X_val.npy"), X_val)
np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)

print("\nTamaños:")
print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)


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

print("\nTamaños:")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


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

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)

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

plt.figure()
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss de entrenamiento y validación")
plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"), dpi=300)


