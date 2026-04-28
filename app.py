import os
import base64
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

from src.audio_processing import preprocess_waveform_segment
from src.config import Config

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

BACKGROUND_IMAGE = "assets/background.jpg"
BIRD_IMAGE_DIR = "assets/birds"


# =========================
# BACKGROUND
# =========================

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


bg_base64 = image_to_base64(BACKGROUND_IMAGE)

custom_css = f"""
.gradio-container {{
    background-image:
        linear-gradient(rgba(255,255,255,0.72), rgba(255,255,255,0.72)),
        url("data:image/jpeg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.main-title {{
    color: black;
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}}

.subtitle {{
    color: black;
    text-align: center;
    font-size: 16px;
}}

.gr-button {{
    background: #7e22ce !important;
    color: white !important;
    border: none !important;
}}

.gr-button:hover {{
    background: #9333ea !important;
}}

.block, .gr-box {{
    background: rgba(255,255,255,0.9) !important;
    border-radius: 15px !important;
}}
"""

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet"
)


# =========================
# CARGA MODELO Y DATOS
# =========================

model = tf.keras.models.load_model(MODEL_PATH)

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"), mmap_mode='r')
X_raw = np.load(os.path.join(PROCESSED_DIR, "X_raw.npy"), mmap_mode='r')

test_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_test_idx.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "main/main_test_y.npy"))

X_test = X[test_idx]

class_names = np.load(
    os.path.join(PROCESSED_DIR, "main_classes.npy"),
    allow_pickle=True
)


# =========================
# FUNCIONES
# =========================


def plot_logmel(logmel, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(logmel, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Bandas Mel")
    plt.tight_layout()
    return fig


def top_predictions(probs):
    idx = np.argsort(probs)[-5:][::-1]
    return {str(class_names[i]): float(probs[i]) for i in idx}


def get_bird_image_path(label):
    possible_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in possible_extensions:
        path = os.path.join(BIRD_IMAGE_DIR, f"{label}{ext}")
        if os.path.exists(path):
            return path

    return None


# =========================
# TEST SET
# =========================

def predict_test_sample(index):
    index = int(index)

    X_sample = X_test[index:index + 1]
    true_idx = int(y_test[index])
    true_label = str(class_names[true_idx])
    audio = X_raw[test_idx[index]]

    probs = model.predict(X_sample, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(class_names[pred_idx])
    confidence = float(probs[pred_idx])

    result = (
        f"Etiqueta real: {true_label}\n"
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Resultado: {'Correcto' if pred_idx == true_idx else 'Incorrecto'}"
    )

    fig = plot_logmel(X_test[index, :, :, 0], "Muestra del test set")
    bird_img = get_bird_image_path(pred_label)

    return result, top_predictions(probs), fig, bird_img, (SR, audio)


# =========================
# AUDIO EXTERNO
# =========================

def predict_audio(audio_path, start_time):
    if audio_path is None:
        return "No se cargó ningún audio.", {}, None, None

    try:
        y_full, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return f"No se pudo leer el audio.\nError: {type(e).__name__}: {e}", {}, None, None

    start = int(start_time * SR)

    if start >= len(y_full):
        return "El segundo de inicio está fuera de la duración del audio.", {}, None, None

    y = y_full[start:start + N_SAMPLES]
    logmel = preprocess_waveform_segment(y)
    X_input = logmel[..., np.newaxis]

    probs = model.predict(X_input, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(class_names[pred_idx])
    confidence = float(probs[pred_idx])

    result = (
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Inicio del segmento: {start_time:.2f} s\n"
        f"Duración usada: {SEGMENT_DURATION} s"
    )

    fig = plot_logmel(logmel, "Audio subido")
    bird_img = get_bird_image_path(pred_label)

    return result, top_predictions(probs), fig, bird_img


# =========================
# APP
# =========================

with gr.Blocks(theme=theme, css=custom_css) as demo:

    gr.HTML(
        f"""
        <div class="main-title">Clasificador de cantos de aves de Hawai</div>
        <div class="subtitle">
            Clases: {", ".join([str(c) for c in class_names])}
        </div>
        """
    )

    with gr.Tab("Test set"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrada")

                idx = gr.Slider(
                    minimum=0,
                    maximum=len(X_test) - 1,
                    value=0,
                    step=1,
                    label="Índice de muestra"
                )

                btn = gr.Button("Evaluar muestra")

                test_audio = gr.Audio(label="Audio")

                gr.Markdown("### Resultado")
                out_text = gr.Textbox(
                    label="Predicción",
                    lines=5
                )

            with gr.Column(scale=1):
                gr.Markdown("### Visualización")

                bird_image = gr.Image(
                    label="Ave reconocida",
                    type="filepath",
                    height=260
                )

                out_label = gr.Label(
                    label="Top predicciones"
                )

                out_plot = gr.Plot(
                    label="Espectrograma"
                )

        btn.click(
            predict_test_sample,
            inputs=[idx],
            outputs=[out_text, out_label, out_plot, bird_image, test_audio]
        )

    with gr.Tab("Audio externo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrada")

                audio = gr.Audio(
                    type="filepath",
                    label="Sube audio (.wav recomendado)"
                )

                start = gr.Slider(
                    minimum=0,
                    maximum=300,
                    value=0,
                    step=0.5,
                    label="Segundo de inicio del segmento"
                )

                btn2 = gr.Button("Clasificar audio")

                gr.Markdown("### Resultado")
                out_text2 = gr.Textbox(
                    label="Predicción",
                    lines=5
                )

            with gr.Column(scale=1):
                gr.Markdown("### Visualización")

                bird_image2 = gr.Image(
                    label="Ave reconocida",
                    type="filepath",
                    height=260
                )

                out_label2 = gr.Label(
                    label="Top predicciones"
                )

                out_plot2 = gr.Plot(
                    label="Espectrograma"
                )

        btn2.click(
            predict_audio,
            inputs=[audio, start],
            outputs=[out_text2, out_label2, out_plot2, bird_image2]
        )


if __name__ == "__main__":
    demo.launch()