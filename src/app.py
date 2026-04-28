import os
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

from config import Config

# =========================
# CONFIG
# =========================

config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_PATH = config.get("paths", "best_model_path")

SR = config.get("audio", "sample_rate")
SEGMENT_DURATION = config.get("audio", "segment_duration")
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = config.get("audio", "n_mels")
N_FFT = config.get("audio", "n_fft")
HOP_LENGTH = config.get("audio", "hop_length")

CONFIDENCE_THRESHOLD = 0.40

# =========================
# CARGA MODELO Y DATOS
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))

test_idx = np.load(
    os.path.join(PROCESSED_DIR, "main", "main_test_idx.npy")
)

y_test = np.load(
    os.path.join(PROCESSED_DIR, "main", "main_test_y.npy")
)

X_test = X[test_idx]

class_names = np.load(
    os.path.join(PROCESSED_DIR, "main_classes.npy"),
    allow_pickle=True
)

print("Modelo cargado:", MODEL_PATH)
print("Clases MAIN:", class_names)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =========================
# FUNCIONES
# =========================

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


def plot_logmel(logmel, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(logmel, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Bandas Mel")
    plt.tight_layout()
    return fig


def top_predictions(probs, top_n=5):
    top_idx = np.argsort(probs)[-top_n:][::-1]

    return {
        str(class_names[i]): float(probs[i])
        for i in top_idx
    }


def make_result_text(pred_label, confidence, extra_text=""):
    if confidence < CONFIDENCE_THRESHOLD:
        return (
            f"Predicción de baja confianza\n"
            f"Clase más probable: {pred_label}\n"
            f"Confianza: {confidence:.4f}\n"
            f"{extra_text}"
        )

    return (
        f"Especie predicha: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"{extra_text}"
    )


# =========================
# MODO 1: TEST SET MAIN
# =========================

def predict_test_sample(index):
    index = int(index)

    if index < 0 or index >= len(X_test):
        return "Índice fuera de rango.", {}, None

    X_sample = X_test[index:index + 1]
    true_idx = int(y_test[index])
    true_label = str(class_names[true_idx])

    probs = model.predict(X_sample, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(class_names[pred_idx])
    confidence = float(probs[pred_idx])

    correct = pred_idx == true_idx

    result = (
        f"Etiqueta real: {true_label}\n"
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Resultado: {'Correcto' if correct else 'Incorrecto'}"
    )

    logmel = X_test[index, :, :, 0]
    fig = plot_logmel(logmel, "Muestra real del test set MAIN")

    return result, top_predictions(probs), fig


# =========================
# MODO 2: AUDIO EXTERNO
# =========================

def predict_uploaded_audio(audio_path, start_time, noise_level):
    if audio_path is None:
        return "No se cargó ningún audio.", {}, None

    try:
        y_full, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return (
            f"No se pudo leer el audio.\n"
            f"Error: {type(e).__name__}: {e}",
            {},
            None
        )

    start_sample = int(start_time * SR)
    end_sample = start_sample + N_SAMPLES

    if start_sample >= len(y_full):
        return (
            "El segundo de inicio está fuera de la duración del audio.",
            {},
            None
        )

    y = y_full[start_sample:end_sample]
    y = fix_length_audio(y)
    y = add_noise(y, noise_level)

    logmel = audio_to_logmel(y)
    X_input = logmel[np.newaxis, ..., np.newaxis]

    expected_shape = model.input_shape[1:]

    if X_input.shape[1:] != expected_shape:
        return (
            f"Error de forma de entrada.\n"
            f"Entrada generada: {X_input.shape[1:]}\n"
            f"Modelo espera: {expected_shape}\n"
            f"Revisa sample_rate, segment_duration, n_fft y hop_length en config.json.",
            {},
            None
        )

    probs = model.predict(X_input, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(class_names[pred_idx])
    confidence = float(probs[pred_idx])

    extra = (
        f"Inicio del segmento: {start_time:.2f} s\n"
        f"Duración usada: {SEGMENT_DURATION} s\n"
        f"Ruido blanco: {noise_level:.2f}"
    )

    result = make_result_text(pred_label, confidence, extra)
    fig = plot_logmel(logmel, "Espectrograma del audio subido")

    return result, top_predictions(probs), fig


# =========================
# APP
# =========================

with gr.Blocks(title="Clasificador de aves de Hawai") as demo:
    gr.Markdown(
        f"""
        # Clasificador de cantos de aves de Hawai

        Modelo CNN entrenado con el subconjunto **MAIN** generado por el pipeline.

        **Modelo usado:** `{MODEL_PATH}`

        **Clases del modelo:**  
        {", ".join([str(c) for c in class_names])}
        """
    )

    with gr.Tab("Evaluar muestra real del test set"):
        gr.Markdown(
            """
            Este modo usa directamente `X.npy`, `main_test_idx.npy` y `main_test_y.npy`.
            Es el modo más fiel a la evaluación del script `03_evaluate.py`.
            """
        )

        test_index = gr.Slider(
            minimum=0,
            maximum=len(X_test) - 1,
            value=0,
            step=1,
            label="Índice de muestra del test set"
        )

        test_button = gr.Button("Evaluar muestra")

        test_result = gr.Textbox(label="Resultado")
        test_top = gr.Label(label="Top predicciones")
        test_plot = gr.Plot(label="Espectrograma")

        test_button.click(
            fn=predict_test_sample,
            inputs=[test_index],
            outputs=[test_result, test_top, test_plot]
        )

    with gr.Tab("Clasificar audio externo"):
        gr.Markdown(
            """
            El modelo fue entrenado con segmentos de 3 segundos.
            Si subes una grabación completa, selecciona el segundo de inicio donde se escucha el canto.
            """
        )

        audio_input = gr.Audio(
            type="filepath",
            label="Sube audio (.wav recomendado)"
        )

        start_time = gr.Slider(
            minimum=0,
            maximum=300,
            value=0,
            step=0.5,
            label="Segundo de inicio del segmento"
        )

        noise_level = gr.Slider(
            minimum=0.0,
            maximum=0.5,
            value=0.0,
            step=0.01,
            label="Ruido blanco"
        )

        audio_button = gr.Button("Clasificar audio")

        audio_result = gr.Textbox(label="Resultado")
        audio_top = gr.Label(label="Top predicciones")
        audio_plot = gr.Plot(label="Espectrograma")

        audio_button.click(
            fn=predict_uploaded_audio,
            inputs=[audio_input, start_time, noise_level],
            outputs=[audio_result, audio_top, audio_plot]
        )


if __name__ == "__main__":
    demo.launch()