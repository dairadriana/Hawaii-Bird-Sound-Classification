import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

MODEL_PATH = "models/best_cnn_model_12_classes.keras"
ENCODER_PATH = "processed/label_encoder.pkl"

SR = 16000
SEGMENT_DURATION = 3.0
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)


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


def add_noise(y, noise_level):
    if noise_level <= 0:
        return y

    noise = np.random.normal(0, noise_level, size=len(y))
    return y + noise


def make_spectrogram_plot(logmel):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(logmel, aspect="auto", origin="lower")
    ax.set_title("Espectrograma Log-Mel")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Bandas Mel")
    plt.tight_layout()
    return fig


def predict(audio_path, noise_level):
    if audio_path is None:
        return "No se cargó audio.", {}, None

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = fix_length_audio(y)
    y = add_noise(y, noise_level)

    logmel = audio_to_logmel(y)
    X = logmel[np.newaxis, ..., np.newaxis]

    probs = model.predict(X, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    top_indices = np.argsort(probs)[-5:][::-1]

    top_probs = {
        encoder.classes_[i]: float(probs[i])
        for i in top_indices
    }

    result_text = (
        f"Especie predicha: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Ruido aplicado: {noise_level}"
    )

    fig = make_spectrogram_plot(logmel)

    return result_text, top_probs, fig


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type="filepath", label="Sube un audio (.wav, .flac, .mp3)"),
        gr.Slider(
            minimum=0.0,
            maximum=0.5,
            value=0.0,
            step=0.01,
            label="Nivel de ruido blanco"
        )
    ],
    outputs=[
        gr.Textbox(label="Resultado"),
        gr.Label(label="Top 5 predicciones"),
        gr.Plot(label="Espectrograma Log-Mel")
    ],
    title="Clasificador de cantos de aves de Hawai",
    description=(
        "Aplicación de RNA para clasificar cantos de aves usando espectrogramas Log-Mel "
        "y una red neuronal convolucional. Permite agregar ruido blanco para observar "
        "variaciones en el comportamiento del modelo."
    )
)

if __name__ == "__main__":
    interface.launch()