import os
import pickle
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

# =========================
# RUTAS
# =========================

MODEL_PATH = "models/best_cnn_model_12_classes.keras"
ENCODER_PATH = "processed/label_encoder.pkl"

X_TEST_PATH = "processed/X_test.npy"
Y_TEST_PATH = "processed/y_test.npy"

# =========================
# CONFIG IGUAL AL TRAIN
# =========================

SR = 16000
SEGMENT_DURATION = 3.0
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

CONFIDENCE_THRESHOLD = 0.40

# =========================
# CARGA
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

class_names = encoder.classes_

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print("Modelo cargado:", MODEL_PATH)
print("Clases:", class_names)
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


def plot_logmel(logmel, title="Espectrograma Log-Mel"):
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
        class_names[i]: float(probs[i])
        for i in top_idx
    }


def prediction_text(pred_label, confidence, extra_text=""):
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
# MODO 1: TEST SET REAL
# =========================

def predict_test_sample(index):
    index = int(index)

    if index < 0 or index >= len(X_test):
        return "Índice fuera de rango.", {}, None

    X = X_test[index:index + 1]
    true_idx = int(y_test[index])
    true_label = class_names[true_idx]

    probs = model.predict(X, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    correct = pred_idx == true_idx

    result = (
        f"Etiqueta real: {true_label}\n"
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Resultado: {'Correcto' if correct else 'Incorrecto'}"
    )

    logmel = X_test[index, :, :, 0]
    fig = plot_logmel(logmel, title="Muestra real del conjunto de prueba")

    return result, top_predictions(probs), fig


# =========================
# MODO 2: AUDIO SUBIDO
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
    X = logmel[np.newaxis, ..., np.newaxis]

    probs = model.predict(X, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    extra = (
        f"Inicio del segmento: {start_time:.2f} s\n"
        f"Duración usada: {SEGMENT_DURATION} s\n"
        f"Ruido blanco: {noise_level:.2f}"
    )

    result = prediction_text(pred_label, confidence, extra)

    fig = plot_logmel(logmel, title="Espectrograma del audio subido")

    return result, top_predictions(probs), fig


# =========================
# FRONTEND
# =========================

with gr.Blocks(title="Clasificador de aves de Hawai") as demo:
    gr.Markdown(
        """
        # Clasificador de cantos de aves de Hawai

        Modelo CNN entrenado con espectrogramas Log-Mel de segmentos de 3 segundos.

        **Modo recomendado para verificar resultados:** usar el conjunto de prueba.  
        **Modo audio externo:** seleccionar correctamente el segundo de inicio del canto.
        """
    )

    with gr.Tab("Evaluar muestra real del test set"):
        gr.Markdown(
            """
            Este modo usa directamente `X_test.npy` y `y_test.npy`, por lo que es el más fiel a las métricas obtenidas en evaluación.
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
            El modelo no clasifica grabaciones completas.  
            Selecciona el segundo de inicio para extraer un segmento de 3 segundos, igual que en el entrenamiento.
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

    gr.Markdown(
        f"""
        ## Clases del modelo

        {", ".join(class_names)}
        """
    )


if __name__ == "__main__":
    demo.launch()