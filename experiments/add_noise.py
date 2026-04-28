import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

PROCESSED_DIR = "processed"
MODEL_PATH = "models/best_cnn_model_500_samples.keras"
MIN_SAMPLES_PER_CLASS = 500
# MELS CONFIG
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

X_raw = np.load(os.path.join(PROCESSED_DIR, "X_raw.npy"))
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

NOISE_LEVELS=np.array([0,0.05,0.1,0.5,1,2,4,8])

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    old_encoder = pickle.load(f)

old_class_names = old_encoder.classes_

print("X original:", X_raw.shape)
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

X_raw = X_raw[mask]
y = y[mask]

kept_class_names = old_class_names[valid_classes]

# Reindexar etiquetas: 0, 1, 2, ...
new_encoder = LabelEncoder()
y_text = old_class_names[y]
y = new_encoder.fit_transform(y_text)

num_classes = len(np.unique(y))

print("\nDespués de filtrar clases pequeñas:")
print("X:", X_raw.shape)
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
    X_raw,
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
# CREAR VAL/TEST SETS CON RUIDO
# =========================

# Generar ruido de fondo (white noise) a partir del dataset de audio crudo
# Usar el ruido generado para perturbar val y test
# Convertir a Log-Mel Spectrograms

def add_noise(X,noise_level):
    noise = np.random.randn(*X.shape)
    X = (X - X.min()) / (X.max() - X.min())
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    X = X + noise_level * noise

    return X

def audio_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    return logmel

# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# EVALUAR PARA DIFERENTES NIVELES DE RUIDO
# =========================

X_val_noise = []

for level in NOISE_LEVELS:
    x_val = add_noise(X_val, level)

    x_val_mel = []
    for audio in x_val:
        x_val_mel.append(audio_to_logmel(audio))
    
    x_val_mel = np.array(x_val_mel)
    x_val_mel = x_val_mel[..., np.newaxis]

    probs = model.predict(x_val_mel)
    y_pred = np.argmax(probs, axis=1)

    print("\nClassification report for noise level:", level)
    print(classification_report(
        y_val,
        y_pred,
        target_names=kept_class_names,
        zero_division=0
    ))
