import os
import pickle
import numpy as np
import pandas as pd
import librosa
import traceback

from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# =========================
# CONFIGURACIÓN
# =========================

DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "soundscape_data")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR = 16000
SEGMENT_DURATION = 3.0
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

MAX_SAMPLES_PER_CLASS = 1000

SAVE_RAW_AUDIO = False  
SAVE_LOGMEL = True

np.random.seed(42)

# =========================
# FUNCIONES
# =========================

def load_audio_segment(filepath, start_time, end_time):
    duration = end_time - start_time

    y, sr = librosa.load(
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

    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    return logmel


def find_audio_file(filename):
    """
    Busca el archivo de audio aunque venga con extensión distinta
    o aunque esté en subcarpetas.
    """
    filename = str(filename)

    possible_names = [
        filename,
        filename.replace(".wav", ".flac"),
        filename.replace(".WAV", ".flac"),
        filename.replace(".FLAC", ".flac")
    ]

    for root, dirs, files in os.walk(AUDIO_DIR):
        for name in possible_names:
            if name in files:
                return os.path.join(root, name)

    return None


# =========================
# MAIN
# =========================

def main():
    annotations = pd.read_csv(ANNOTATIONS_FILE)

    print("Columnas encontradas:")
    print(annotations.columns)

    filename_col = "Filename"
    start_col = "Start Time (s)"
    end_col = "End Time (s)"
    label_col = "Species eBird Code"

    print(f"Usando columnas: {filename_col}, {start_col}, {end_col}, {label_col}")

    annotations = annotations.dropna(
        subset=[filename_col, start_col, end_col, label_col]
    )

    annotations[start_col] = annotations[start_col].astype(float)
    annotations[end_col] = annotations[end_col].astype(float)

    annotations = annotations[
        annotations[end_col] > annotations[start_col]
    ]

    print("Total de anotaciones válidas:", len(annotations))
    print("Número de especies:", annotations[label_col].nunique())

    print("\nDistribución original por especie:")
    print(annotations[label_col].value_counts())


    sampled_list = []
    for _, group in annotations.groupby(label_col):
        sampled_list.append(
            group.sample(min(len(group), MAX_SAMPLES_PER_CLASS), random_state=42)
        )
    annotations = pd.concat(sampled_list).reset_index(drop=True)

    print("\nDistribución después de limitar muestras:")
    print(annotations[label_col].value_counts())

    X = []
    X_raw = []
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
            audio = librosa.resample(
                y=audio,
                orig_sr=32000,
                target_sr=SR
            )

            if SAVE_LOGMEL:
                logmel = audio_to_logmel(audio)
                X.append(logmel)
            
            if SAVE_RAW_AUDIO:
                X_raw.append(audio)
                
            y_labels.append(label)

        except Exception as e:
            errors += 1
            print(f"Error procesando {filename}: {e}")
            continue

    if len(X) == 0 :
        raise ValueError(
            "No se generó ningún segmento. Revisa AUDIO_DIR, nombres de archivos y estructura de carpetas."
        )


    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    if SAVE_LOGMEL:
        X = np.array(X, dtype=np.float32)
        X = X[..., np.newaxis]
        np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    
    if SAVE_RAW_AUDIO:
        X_raw = np.array(X_raw, dtype=np.float32)
        np.save(os.path.join(OUTPUT_DIR, "X_raw.npy"), X_raw)
        
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print("\n==============================")
    print("Dataset procesado correctamente")
    print("==============================")
    if SAVE_LOGMEL:
        print("X shape (Log-Mel):", X.shape)
    if SAVE_RAW_AUDIO:
        print("X_raw shape (Waveform):", X_raw.shape)
    print("y shape:", y.shape)
    print("Clases:", encoder.classes_)
    print("Archivos no encontrados:", missing_files)
    print("Errores de procesamiento:", errors)

    print("\nArchivos guardados en:")
    if SAVE_LOGMEL:
        print(os.path.join(OUTPUT_DIR, "X.npy"))
    if SAVE_RAW_AUDIO:
        print(os.path.join(OUTPUT_DIR, "X_raw.npy"))
    print(os.path.join(OUTPUT_DIR, "y.npy"))
    print(os.path.join(OUTPUT_DIR, "label_encoder.pkl"))


if __name__ == "__main__":
    main()