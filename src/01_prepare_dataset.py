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

    annotations[start_col] = annotations[start_col].astype(float)
    annotations[end_col] = annotations[end_col].astype(float)

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

    X = []
    X_raw = []
    y_labels = []

    missing_files = 0
    errors = 0

    for _, row in tqdm(annotations.iterrows(), total=len(annotations)):
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
                X.append(audio_to_logmel(audio))

            if SAVE_RAW_AUDIO:
                X_raw.append(audio)

            y_labels.append(row[label_col])

        except Exception as e:
            errors += 1
            print(f"Error procesando {row[filename_col]}: {e}")

    if not X and not X_raw:
        raise ValueError("No se generó ningún segmento válido.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    if SAVE_LOGMEL:
        X = np.array(X, dtype=np.float32)[..., np.newaxis]
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)

    if SAVE_RAW_AUDIO:
        X_raw = np.array(X_raw, dtype=np.float32)
        np.save(os.path.join(OUTPUT_DIR, "X_raw.npy"), X_raw)

    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)


if __name__ == "__main__":
    main()