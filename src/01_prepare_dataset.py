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

MIN_SAMPLES_PER_CLASS = 5
MAX_SAMPLES_PER_CLASS = 1000   # muestras usadas para dataset principal


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

    # Normalización original: fue la que sí funcionó bien
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

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


def process_annotations(df, filename_col, start_col, end_col, label_col):
    X = []
    y_labels = []

    missing_files = 0
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
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

    X = np.array(X, dtype=np.float32)

    if len(X) > 0:
        X = X[..., np.newaxis]

    return X, y_labels, missing_files, errors


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

    annotations = annotations[annotations[end_col] > annotations[start_col]]

    class_counts = annotations[label_col].value_counts()
    valid_labels = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    annotations = annotations[annotations[label_col].isin(valid_labels)]

    print("Total de anotaciones válidas:", len(annotations))
    print("Número de especies:", annotations[label_col].nunique())

    main_parts = []
    extra_parts = []

    for label, group in annotations.groupby(label_col):
        group = group.sample(frac=1, random_state=42)

        main_group = group.iloc[:MAX_SAMPLES_PER_CLASS]
        extra_group = group.iloc[MAX_SAMPLES_PER_CLASS:]

        main_parts.append(main_group)

        if len(extra_group) > 0:
            extra_parts.append(extra_group)

    main_df = pd.concat(main_parts).reset_index(drop=True)
    main_df = main_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(extra_parts) > 0:
        extra_df = pd.concat(extra_parts).reset_index(drop=True)
        extra_df = extra_df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        extra_df = pd.DataFrame(columns=annotations.columns)

    print("\nDistribución dataset principal:")
    print(main_df[label_col].value_counts())

    print("\nDistribución dataset sobrante:")
    print(extra_df[label_col].value_counts())

    print("\nProcesando dataset principal...")
    X, y_labels, missing_main, errors_main = process_annotations(
        main_df, filename_col, start_col, end_col, label_col
    )

    if len(X) == 0:
        raise ValueError("No se generó ningún segmento principal.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print("\nProcesando dataset sobrante...")
    X_extra, y_extra_labels, missing_extra, errors_extra = process_annotations(
        extra_df, filename_col, start_col, end_col, label_col
    )

    if len(X_extra) > 0:
        y_extra = encoder.transform(y_extra_labels)

        np.save(os.path.join(OUTPUT_DIR, "X_extra.npy"), X_extra)
        np.save(os.path.join(OUTPUT_DIR, "y_extra.npy"), y_extra)

    print("\n==============================")
    print("PROCESAMIENTO TERMINADO")
    print("==============================")
    print("Dataset principal X:", X.shape)
    print("Dataset principal y:", y.shape)
    print("Clases:", encoder.classes_)
    print("Errores principales:", errors_main)
    print("Archivos faltantes principales:", missing_main)

    if len(X_extra) > 0:
        print("\nDataset sobrante X_extra:", X_extra.shape)
        print("Dataset sobrante y_extra:", y_extra.shape)
        print("Errores sobrante:", errors_extra)
        print("Archivos faltantes sobrante:", missing_extra)
    else:
        print("\nNo hubo dataset sobrante.")


if __name__ == "__main__":
    main()



    