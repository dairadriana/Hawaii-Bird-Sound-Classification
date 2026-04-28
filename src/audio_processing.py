import os
import numpy as np
import librosa

from config import Config

config = Config()

DATA_DIR = config.get("paths", "data_dir")
AUDIO_DIR = os.path.join(DATA_DIR, "soundscape_data")

SR = config.get("audio", "sample_rate")
SEGMENT_DURATION = config.get("audio", "segment_duration")
N_SAMPLES = int(SR * SEGMENT_DURATION)

N_MELS = config.get("audio", "n_mels")
N_FFT = config.get("audio", "n_fft")
HOP_LENGTH = config.get("audio", "hop_length")


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


def load_audio_segment(filepath, start_time, end_time):
    duration = end_time - start_time

    audio, _ = librosa.load(
        filepath,
        sr=None,
        offset=start_time,
        duration=duration,
        mono=True
    )

    return audio


def fix_length_audio(audio):
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    return audio


def resample_audio(audio, orig_sr):
    return librosa.resample(
        y=audio,
        orig_sr=orig_sr,
        target_sr=SR
    )


def preprocess_audio_segment(filepath, start_time, end_time, orig_sr=32000):
    audio = load_audio_segment(filepath, start_time, end_time)
    audio = fix_length_audio(audio)
    audio = resample_audio(audio, orig_sr)
    return audio


def audio_to_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    return logmel