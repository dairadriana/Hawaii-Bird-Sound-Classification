import os
import pickle
import numpy as np
import pandas as pd
import librosa

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "soundscape_data")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 32000
SEGMENT_DURATION = 3.0
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

MAX_SAMPLES_PER_CLASS = 1500
MIN_SAMPLES_PER_CLASS = 5


def load_audio_segment(filepath, start_time, end_time):
    duration = end_time - start_time

    y, _ = librosa.load(
        filepath,
        sr=SR,
        offset=start_time,
        duration=duration,
        mono=True
    )

    return y


def fix_length_audio(y):
    if len(y) < N_SAMPLES:
        y = np.pad(y, (0, N_SAMPLES - len(y)))
    else:
        y = y[:N_SAMPLES]

    return y


def audio_to_logmel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # Normalización Min-Max para CNN
    logmel = (logmel - np.min(logmel)) / (np.max(logmel) - np.min(logmel) + 1e-8)

    return logmel


def find_audio_file(filename):
    filename = str(filename)

    possible_names = [
        filename,
        filename.replace(".wav", ".flac"),
        filename.replace(".WAV", ".flac"),
        filename.replace(".FLAC", ".flac")
    ]

    for root, _, files in os.walk(AUDIO_DIR):
        for name in possible_names:
            if name in files:
                return os.path.join(root, name)

    return None


def main():
    annotations = pd.read_csv(ANNOTATIONS_FILE)

    print("Columnas encontradas:")
    print(annotations.columns)

    filename_col = "Filename"
    start_col = "Start Time (s)"
    end_col = "End Time (s)"
    label_col = "Species eBird Code"

    annotations = annotations.dropna(
        subset=[filename_col, start_col, end_col, label_col]
    )

    annotations[start_col] = annotations[start_col].astype(float)
    annotations[end_col] = annotations[end_col].astype(float)

    annotations = annotations[annotations[end_col] > annotations[start_col]]

    print("Total de anotaciones válidas:", len(annotations))
    print("Número de especies original:", annotations[label_col].nunique())

    class_counts = annotations[label_col].value_counts()
    valid_labels = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index

    annotations = annotations[annotations[label_col].isin(valid_labels)]

    print("Número de especies después de filtrar:", annotations[label_col].nunique())

    annotations = (
        annotations
        .groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(
            min(len(x), MAX_SAMPLES_PER_CLASS),
            random_state=42
        ))
        .reset_index(drop=True)
    )

    print("\nDistribución final:")
    print(annotations[label_col].value_counts())

    X = []
    y_labels = []

    missing_files = 0
    errors = 0

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
        filename = row[filename_col]
        start_time = row[start_col]
        end_time = row[end_col]
        label = row[label_col]

        filepath = find_audio_file(filename)

        if filepath is None:
            missing_files += 1
            continue

        try:
            audio = load_audio_segment(filepath, start_time, end_time)
            audio = fix_length_audio(audio)
            logmel = audio_to_logmel(audio)

            X.append(logmel)
            y_labels.append(label)

        except Exception as e:
            errors += 1
            print(f"Error procesando {filename}: {e}")

    if len(X) == 0:
        raise ValueError("No se generó ningún segmento. Revisa rutas y nombres de archivos.")

    X = np.array(X, dtype=np.float32)
    X = X[..., np.newaxis]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print("\nDataset procesado correctamente")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Clases:", encoder.classes_)
    print("Archivos no encontrados:", missing_files)
    print("Errores:", errors)


if __name__ == "__main__":
    main()