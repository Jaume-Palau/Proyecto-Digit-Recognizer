from pathlib import Path

import torch


# Raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parents[1]

# Datos
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

COMPETITION_DATA_DIR = (
    RAW_DATA_DIR / "conquerx-b05-lec01-digit-recognizer"
)

TRAIN_CSV = COMPETITION_DATA_DIR / "train.csv"
TEST_CSV = COMPETITION_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = COMPETITION_DATA_DIR / "sample_submission.csv"

# Salidas
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
MODELS_DIR = OUTPUTS_DIR / "models"
PICKLE_DIR = OUTPUTS_DIR / "pickle"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"

# Código y experimentación
SRC_DIR = ROOT_DIR / "src"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"



TRAIN_CONFIG = {
    "name" : "modelo_prueba1",
    "train_csv": TRAIN_CSV,
    "test_csv": TEST_CSV,
    "checkpoint_dir": CHECKPOINTS_DIR/"modelo_prueba1",
    "model_dir": MODELS_DIR/"modelo_prueba1",
    "pickle_dir": PICKLE_DIR/"modelo_prueba1",
    "batch_size": 64,
    "learning_rate": 0.001,
    "val_split": 0.2,
    "num_workers": 0,
    "seed": 42,
    "n_epochs": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}