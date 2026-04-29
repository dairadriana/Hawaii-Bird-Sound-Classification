import os
import base64
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr
import matplotlib.pyplot as plt

from src.audio_processing import preprocess_waveform_segment, preprocess_waveform_segment, add_noise_to_audio
from src.config import Config


config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
MODEL_PATH = config.get("paths", "best_model_path")
MODEL_TL_PATH = config.get("paths", "best_model_path_transfer_learning")

SR = config.get("audio", "sample_rate")
SEGMENT_DURATION = config.get("audio", "segment_duration")
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = config.get("audio", "n_mels")
N_FFT = config.get("audio", "n_fft")
HOP_LENGTH = config.get("audio", "hop_length")

BACKGROUND_IMAGE = "assets/background.jpg"
BIRD_IMAGE_DIR = "assets/birds"

# Background - bosque(?)
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

.bird-frame {{
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 12px;
    margin-top: 10px;

    border: 2px solid #e9d5ff;
    border-radius: 12px;
    background: rgba(255,255,255,0.8);

    max-width: 100%;
    max-height: 280px;
}}

.bird-frame img {{
    max-width: 100%;
    max-height: 240px;
    object-fit: contain;
    border-radius: 8px;
}}

.section-title {{
    margin-top: 10px;
    margin-bottom: 10px !important;

    padding: 2px 8px;
    border-radius: 2px;

    background: rgb(135, 128, 145) !important;
    color: white !important;

    font-weight: 700;
    width: 100%;
    box-sizing: border-box;
}}

.section-title h3 {{
    color: white !important;
    margin: 0 !important;
    text-align: center; 
}}

.gr-label {{
    margin-top: 5px !important;
}}

.green-btn {{
    background: rgb(162, 184, 128) !important;  /* verde claro */
    color: white !important;
    font-weight: bold;
}}

.green-btn:hover {{
    background: rgb(213, 224, 195) !important;
}}
"""

theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet"
)


#Carga de modelo y datos
model = tf.keras.models.load_model(MODEL_PATH)
model_tl = tf.keras.models.load_model(MODEL_TL_PATH)

X = np.load(os.path.join(PROCESSED_DIR, "X.npy"), mmap_mode='r')
X_raw = np.load(os.path.join(PROCESSED_DIR, "X_raw.npy"), mmap_mode='r')

test_idx = np.load(os.path.join(PROCESSED_DIR, "main/main_test_idx.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "main/main_test_y.npy"))

exp_test_idx = np.load(os.path.join(PROCESSED_DIR, "exp/exp_test_idx.npy"))
exp_y_test = np.load(os.path.join(PROCESSED_DIR, "exp/exp_test_y.npy"))

X_test = X[test_idx]
exp_X_test = X[exp_test_idx]


class_names = np.load(
    os.path.join(PROCESSED_DIR, "main_classes.npy"),
    allow_pickle=True
)

exp_class_names = np.load(
    os.path.join(PROCESSED_DIR, "exp_classes.npy"),
    allow_pickle=True
)




def plot_logmel(logmel, title):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(logmel, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Bandas Mel")
    plt.tight_layout()
    return fig


def top_predictions(probs, class_names):
    idx = np.argsort(probs)[-5:][::-1]
    return {str(class_names[i]): float(probs[i]) for i in idx}


def get_bird_image_path(label):
    possible_extensions = [".jpg", ".jpeg", ".png", ".webp"]

    for ext in possible_extensions:
        path = os.path.join(BIRD_IMAGE_DIR, f"{label}{ext}")
        if os.path.exists(path):
            return path

    return None


def predict_test_sample(index):
    index = int(index)

    X_sample = exp_X_test[index:index + 1]
    true_idx = int(exp_y_test[index])
    true_label = str(exp_class_names[true_idx])
    audio = X_raw[exp_test_idx[index]]

    probs = model_tl.predict(X_sample, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(exp_class_names[pred_idx])
    confidence = float(probs[pred_idx])

    result = (
        f"Etiqueta real: {true_label}\n"
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Resultado: {'Correcto' if pred_idx == true_idx else 'Incorrecto'}"
    )

    fig = plot_logmel(exp_X_test[index, :, :, 0], "Muestra del test set")

    return (
        gr.update(visible=True),
        gr.update(value=result, visible=True),
        top_predictions(probs, exp_class_names),
        fig
    )



def predict_audio(audio_path, start_time):
    if audio_path is None:
        return gr.update(visible=True), gr.update(value="No se cargó ningún audio.", visible=True), {}, None, None

    try:
        y_full, _ = librosa.load(audio_path, sr=SR, mono=True)
    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(value=f"No se pudo leer el audio.\nError: {type(e).__name__}: {e}", visible=True),
            {},
            None,
            None
        )

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

    return (
        gr.update(visible=True),
        gr.update(value=result, visible=True),
        top_predictions(probs, class_names),
        fig,
        bird_img
    )

def predict_noisy_sample(index, snr_level):
    audio = X_raw[test_idx[index]]

    if snr_level == "None":
        X_sample = X_test[index]
    else:
        X_sample = preprocess_waveform_segment(audio, snr_db=int(snr_level))
        X_sample = X_sample[..., np.newaxis]
        audio = add_noise_to_audio(audio, snr_level)

    probs = model.predict(X_sample[np.newaxis, ...], verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = str(class_names[pred_idx])
    confidence = float(probs[pred_idx])

    true_idx = y_test[index]
    true_label = str(class_names[true_idx])

    result = (
        f"Predicción: {pred_label}\n"
        f"Confianza: {confidence:.4f}\n"
        f"Resultado: {'Correcto' if pred_idx == true_idx else 'Incorrecto'}"
    )

    fig = plot_logmel(X_sample[:, :, 0], f"Sample {index} con ruido SNR={snr_level}")
    bird_img = get_bird_image_path(pred_label)

    return (
        gr.update(visible=True),
        gr.update(value=result, visible=True),
        top_predictions(probs, class_names),
        fig,
        bird_img,
        (SR, audio)
    )

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
                gr.Markdown("### Entrada", elem_classes="section-title")

                idx = gr.Slider(
                    minimum=0,
                    maximum=len(X_test) - 1,
                    value=0,
                    step=1,
                    label="Índice de muestra"
                )

                noise_level = gr.Radio(
                    choices=config.get("add_noise", "noise_levels"), 
                    value=config.get("add_noise", "noise_levels")[0], 
                    label="Nivel de ruido SNR"
                )

                btn = gr.Button("Evaluar muestra", elem_classes="green-btn")

                result_title = gr.Markdown(
                    "### Resultado",
                    elem_classes="section-title",
                    visible=False
                )

                out_text = gr.Textbox(
                    label="Predicción",
                    lines=5,
                    visible=False
                )

            with gr.Column(scale=2):
                gr.Markdown("### Visualización", elem_classes="section-title")
                # Sonido
                audio_input = gr.Audio(
                    label="Audio con ruido"
                )
                with gr.Row():
                    # imagen de cada pájaro
                    with gr.Column(scale=1):
                        bird_image = gr.Image(
                            label="Ave reconocida",
                            type="filepath",
                            elem_classes="bird-frame"
                        )

                    with gr.Column(scale=1):
                        out_label = gr.Label(
                            label="Top 5 predicciones"
                        )

                        out_plot = gr.Plot(
                            label="Espectrograma"
                        )

        btn.click(
            predict_noisy_sample,
            inputs=[idx, noise_level],
            outputs=[result_title, out_text, out_label, out_plot, bird_image, audio_input]
        )

    with gr.Tab("Audio externo"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrada", elem_classes="section-title")

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

                btn2 = gr.Button("Clasificar audio", elem_classes="green-btn")

                result_title2 = gr.Markdown(
                    "### Resultado",
                    elem_classes="section-title",
                    visible=False
                )

                out_text2 = gr.Textbox(
                    label="Predicción",
                    lines=5,
                    visible=False
                )

            with gr.Column(scale=2):
                gr.Markdown("### Visualización", elem_classes="section-title")

                with gr.Row():
                    with gr.Column(scale=1):
                        bird_image2 = gr.Image(
                            label="Ave reconocida",
                            type="filepath",
                            elem_classes="bird-frame"
                        )
                    with gr.Column(scale=1):
                        out_label2 = gr.Label(
                            label="Top 5 predicciones"
                        )

                        out_plot2 = gr.Plot(
                            label="Espectrograma"
                        )

        btn2.click(
            predict_audio,
            inputs=[audio, start],
            outputs=[result_title2, out_text2, out_label2, out_plot2, bird_image2]
        )


    with gr.Tab("Transfer Learning"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrada", elem_classes="section-title")

                idx = gr.Slider(
                    minimum=0,
                    maximum=len(exp_X_test) - 1,
                    value=0,
                    step=1,
                    label="Índice de muestra"
                )

                btn = gr.Button("Evaluar muestra", elem_classes="green-btn")

                result_title = gr.Markdown(
                    "### Resultado",
                    elem_classes="section-title",
                    visible=False
                )

                out_text = gr.Textbox(
                    label="Predicción",
                    lines=5,
                    visible=False
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### Visualización", elem_classes="section-title")

                with gr.Row():
                    # Subcolumna izquierda: imagen
                    with gr.Column(scale=1):
                        bird_image = gr.Image(
                            label="Ave reconocida",
                            type="filepath",
                            elem_classes="bird-frame"
                        )

                    with gr.Column(scale=1):
                        out_label = gr.Label(
                            label="Top 5 predicciones"
                        )

                        out_plot = gr.Plot(
                            label="Espectrograma"
                        )

        btn.click(
            predict_test_sample,
            inputs=[idx],
            outputs=[result_title, out_text, out_label, out_plot]
        )


if __name__ == "__main__":
    demo.launch()