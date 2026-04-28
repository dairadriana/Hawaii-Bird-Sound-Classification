import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import Config

config = Config()

PROCESSED_DIR = config.get("paths", "processed_dir")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MIN_SAMPLES_PER_CLASS = config.get("dataset", "min_samples_per_class")
RANDOM_STATE = config.get("dataset", "random_state")
TEST_SIZE = config.get("dataset", "test_size")
VAL_SPLIT = config.get("dataset", "val_split")

# =========================
# CARGAR DATOS
# =========================

# We are using indexes so we only need y
y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
original_indices = np.arange(len(y))

with open(os.path.join(PROCESSED_DIR, "label_encoder.pkl"), "rb") as f:
    class_encoder = pickle.load(f)

class_names = class_encoder.classes_

print("Número original de clases:", len(np.unique(y)))

# =========================
# FILTRAR CLASES PEQUEÑAS
# =========================

classes, counts = np.unique(y, return_counts=True)

print("\nDistribución original:")
for c, count in zip(classes, counts):
    print(f"{class_names[c]}: {count}")

valid_classes = classes[counts >= MIN_SAMPLES_PER_CLASS]

mask = np.isin(y, valid_classes)

y = y[mask]
filtered_idx = original_indices[mask]
y_text = class_names[y]
#AGREGAR UN PRINT PARA SABER SI LO HIZO BIEN

# =========================
# DIVIDIR CLASES ENTRE MAIN Y EXP
# =========================

all_filtered_classes = np.unique(y_text)

main_class_names, exp_class_names = train_test_split(
    all_filtered_classes,
    test_size=0.5,
    random_state=RANDOM_STATE
)

main_mask = np.isin(y_text, main_class_names)
exp_mask  = np.isin(y_text, exp_class_names)

main_idx = filtered_idx[main_mask]
exp_idx  = filtered_idx[exp_mask]

main_y_text = y_text[main_mask]
exp_y_text  = y_text[exp_mask]
#AGREGAR UN PRINT PARA SABER SI LO HIZO BIEN

# =========================
# REINDEXAR
# =========================
main_encoder = LabelEncoder()
main_y = main_encoder.fit_transform(main_y_text)

exp_encoder = LabelEncoder()
exp_y = exp_encoder.fit_transform(exp_y_text)

with open(os.path.join(PROCESSED_DIR, "main_label_encoder.pkl"), "wb") as f:
    pickle.dump(main_encoder, f)

with open(os.path.join(PROCESSED_DIR, "exp_label_encoder.pkl"), "wb") as f:
    pickle.dump(exp_encoder, f)

np.save(os.path.join(PROCESSED_DIR, "main_classes.npy"), main_encoder.classes_)
np.save(os.path.join(PROCESSED_DIR, "exp_classes.npy"), exp_encoder.classes_)

# =========================
# SPLIT TRAIN / VAL / TEST
# =========================

# main
main_train_idx, main_temp_idx, main_y_train, main_y_temp = train_test_split(
    main_idx,
    main_y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=main_y
)

main_val_idx, main_test_idx, main_y_val, main_y_test = train_test_split(
    main_temp_idx,
    main_y_temp,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=main_y_temp
)

#AGREGAR UN PRINT PARA SABER SI LO HIZO BIEN

# exp
exp_train_idx, exp_temp_idx, exp_y_train, exp_y_temp = train_test_split(
    exp_idx,
    exp_y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=exp_y
)


exp_val_idx, exp_test_idx, exp_y_val, exp_y_test = train_test_split(
    exp_temp_idx,
    exp_y_temp,
    test_size=VAL_SPLIT,
    random_state=RANDOM_STATE,
    stratify=exp_y_temp
)



os.makedirs(os.path.join(PROCESSED_DIR, "main"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "exp"), exist_ok=True)

np.save(os.path.join(PROCESSED_DIR, "main/main_train_idx.npy"), main_train_idx)
np.save(os.path.join(PROCESSED_DIR, "main/main_val_idx.npy"), main_val_idx)
np.save(os.path.join(PROCESSED_DIR, "main/main_test_idx.npy"), main_test_idx)

np.save(os.path.join(PROCESSED_DIR, "exp/exp_train_idx.npy"), exp_train_idx)
np.save(os.path.join(PROCESSED_DIR, "exp/exp_val_idx.npy"), exp_val_idx)
np.save(os.path.join(PROCESSED_DIR, "exp/exp_test_idx.npy"), exp_test_idx)


np.save(os.path.join(PROCESSED_DIR, "main/main_train_y.npy"), main_y_train)
np.save(os.path.join(PROCESSED_DIR, "main/main_val_y.npy"), main_y_val)
np.save(os.path.join(PROCESSED_DIR, "main/main_test_y.npy"), main_y_test)

np.save(os.path.join(PROCESSED_DIR, "exp/exp_train_y.npy"), exp_y_train)
np.save(os.path.join(PROCESSED_DIR, "exp/exp_val_y.npy"), exp_y_val)
np.save(os.path.join(PROCESSED_DIR, "exp/exp_test_y.npy"), exp_y_test)