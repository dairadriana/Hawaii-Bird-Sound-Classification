# Ver como le va al modelo con clases con las que no fue entrenado
# Hacer fine tunning a capa final con clases no vistas por el modelo anteriormente
import os
import pickle
import sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import Config
from audio_processing import add_noise_to_audio, audio_to_logmel
from sklearn.metrics import classification_report

config = Config()

MODEL_PATH = config.get("paths", "best_model_path")
PROCESSED_DIR = config.get("paths", "processed_dir")
