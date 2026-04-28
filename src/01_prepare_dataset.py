import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from config import Config
from audio_processing import (
    find_audio_file,
    preprocess_audio_segment,
    audio_to_logmel
)

config = Config()

DATA_DIR = config.get("paths", "data_dir")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
OUTPUT_DIR = config.get("paths", "processed_dir")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SAMPLES_PER_CLASS = config.get("prepare_dataset", "MAX_SAMPLES_PER_CLASS")
SAVE_RAW_AUDIO = config.get("prepare_dataset", "SAVE_RAW_AUDIO")
SAVE_LOGMEL = config.get("prepare_dataset", "SAVE_LOGMEL")


def main():
    annotations = pd.read_csv(ANNOTATIONS_FILE)

    filename_col = "Filename"
    start_col = "Start Time (s)"
    end_col = "End Time (s)"
    label_col = "Species eBird Code"

    annotations = annotations.dropna(
        subset=[filename_col, start_col, end_col, label_col]
    )

    annotations[start_col] = annotations[start_col].astype(np.float32)
    annotations[end_col] = annotations[end_col].astype(np.float32)

    annotations = annotations[
        annotations[end_col] > annotations[start_col]
    ]

    sampled = []
    for _, group in annotations.groupby(label_col):
        sampled.append(
            group.sample(
                min(len(group), MAX_SAMPLES_PER_CLASS),
                random_state=42
            )
        )

    annotations = pd.concat(sampled).reset_index(drop=True)

    n_samples = len(annotations)

    # Infer shapes from one sample
    sample_row = annotations.iloc[0]
    sample_path = find_audio_file(sample_row[filename_col])

    sample_audio = preprocess_audio_segment(
        sample_path,
        sample_row[start_col],
        sample_row[end_col]
    )

    if SAVE_LOGMEL:
        sample_logmel = audio_to_logmel(sample_audio)
        X = np.empty(
            (n_samples, *sample_logmel.shape, 1),
            dtype=np.float32
        )

    if SAVE_RAW_AUDIO:
        X_raw = np.empty(
            (n_samples, len(sample_audio)),
            dtype=np.float32
        )

    y_labels = []

    missing_files = 0
    errors = 0
    valid_count = 0

    for _, row in tqdm(annotations.iterrows(), total=n_samples):
        filepath = find_audio_file(row[filename_col])

        if filepath is None:
            missing_files += 1
            continue

        try:
            audio = preprocess_audio_segment(
                filepath,
                row[start_col],
                row[end_col]
            )

            if SAVE_LOGMEL:
                X[valid_count] = audio_to_logmel(audio)[..., np.newaxis]

            if SAVE_RAW_AUDIO:
                X_raw[valid_count] = audio

            y_labels.append(row[label_col])
            valid_count += 1

        except Exception as e:
            errors += 1
            print(f"Error procesando {row[filename_col]}: {e}")

    if valid_count == 0:
        raise ValueError("No se generó ningún segmento válido.")

    # Trim unused preallocated space
    if SAVE_LOGMEL:
        X = X[:valid_count]

    if SAVE_RAW_AUDIO:
        X_raw = X_raw[:valid_count]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    if SAVE_LOGMEL:
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)

    if SAVE_RAW_AUDIO:
        np.save(os.path.join(OUTPUT_DIR, "X_raw.npy"), X_raw)

    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print("Dataset procesado correctamente")

    if SAVE_LOGMEL:
        print("X shape:", X.shape)

    if SAVE_RAW_AUDIO:
        print("X_raw shape:", X_raw.shape)

    print("y shape:", y.shape)
    print("Archivos no encontrados:", missing_files)
    print("Errores:", errors)


if __name__ == "__main__":
    main()